"""
agents.py — Agentes LLM para The Mind
Usa modelos pequeños (~1-3B) con LoRA para caber en 4GB VRAM o 12GB RAM.
"""
import json
import re
import logging
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

# ─── Prompt base del agente ───────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un jugador del juego The Mind. Reglas:
- Cartas del 1 al 50. No puedes decir tu número.
- El equipo debe jugar todas las cartas en orden ascendente.
- Responde SIEMPRE con JSON válido, sin texto adicional antes ni después."""

# Prompt simplificado: menos campos = menos tokens = menos errores de formato.
# Se elimina "reasoning" del JSON (se mueve fuera) para reducir la longitud generada.
ACTION_PROMPT = """Estado:
- Mis cartas: {my_hand}
- Última carta en mesa: {table_top}
- Vidas: {lives} | Estrellas: {stars}
- Mensajes (jugador [acción que tomó]: texto): {messages}

Responde SOLO con este JSON (sin markdown, sin texto extra):
{{"msg": "texto corto o vacío", "act": "wait"}}

Donde "act" puede ser: "wait" (esperar), "play" (jugar mi carta más baja), "star" (usar estrella).
No incluyas tu número en "msg". Ejemplo válido: {{"msg": "creo que es pronto", "act": "wait"}}"""


ACT_LABEL = {"play": "jugó", "wait": "esperó", "star": "usó estrella"}

def format_messages(messages: list) -> str:
    if not messages:
        return "(ninguno)"
    parts = []
    for m in messages[-5:]:
        act_str = ACT_LABEL.get(m.get("act", ""), "?")
        parts.append(f"J{m['player']} [{act_str}]: {m['text']}")
    return " | ".join(parts)


# ─── Clase agente ─────────────────────────────────────────────────────────────

class TheMindAgent:
    """
    Agente LLM con LoRA. Cada agente tiene sus propios pesos LoRA.
    El modelo base se comparte entre todos los agentes para ahorrar memoria.
    """

    def __init__(
        self,
        player_id: int,
        model=None,          # modelo base compartido
        tokenizer=None,
        lora_config=None,
        device: str = "cpu",
        max_new_tokens: int = 200,
    ):
        self.player_id = player_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.generation_history = []  # para RL

    def build_prompt(self, obs: dict) -> str:
        """Construye el prompt completo para este turno."""
        action_text = ACTION_PROMPT.format(
            my_hand=obs["my_hand"],
            table_top=obs["table_top"],
            lives=obs["lives"],
            stars=obs["stars"],
            messages=format_messages(obs.get("messages", [])),
        )
        # Formato chat — robusto ante transformers 5.x y tokenizers sin chat_template
        has_template = (
            hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None) is not None
        )
        if has_template:
            try:
                chat_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": action_text},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
            except (ValueError, KeyError):
                # Fallback si el template falla en runtime
                prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{action_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Sin chat template: formato Qwen/LLaMA manual
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{action_text}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    @torch.no_grad()
    def generate_action(self, obs: dict) -> dict:
        """
        Genera mensaje + acción dado el estado observable.
        Devuelve dict con 'message', 'action', 'reasoning', 'raw_output'.
        """
        prompt = self.build_prompt(obs)

        # Forzar que la generación empiece con '{' para guiar al modelo hacia JSON
        prompt_with_start = prompt + '{"'

        inputs = self.tokenizer(
            prompt_with_start,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,      # razonador: <think>...</think> + JSON
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        # El modelo generó a partir de '{"', así que lo reincorporamos
        generated = '{"' + self.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        parsed = self._parse_output(generated, obs)
        parsed["raw_output"] = generated
        parsed["prompt"] = prompt

        self.generation_history.append({
            "prompt": prompt,
            "output": generated,
            "obs": obs,
        })

        return parsed

    def _parse_output(self, text: str, obs: dict) -> dict:
        """
        Parsea la salida del modelo con múltiples estrategias en cascada.
        Acepta formato razonador DeepSeek-R1: <think>...</think>{"msg":...,"act":...}
        También acepta claves largas ('message'/'action') como fallback.
        """
        # Extraer bloque <think> si existe (guardarlo para logging/análisis)
        think_content = ""
        if "<think>" in text:
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                # Quitar el bloque think del texto para parsear el JSON limpio
                text = text[text.find("</think>") + len("</think>"):].strip()
        # ── Estrategia 1: buscar JSON completo (con llaves balanceadas) ──────
        # Más robusto que re.search(r'\{.*?\}') que falla con JSON anidado o truncado
        for start in [i for i, c in enumerate(text) if c == '{']:
            depth = 0
            for end, c in enumerate(text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:end + 1]
                        try:
                            data = json.loads(candidate)
                            return self._extract_fields(data)
                        except json.JSONDecodeError:
                            break  # probar siguiente '{'

        # ── Estrategia 2: el JSON está truncado — intentar repararlo ─────────
        # Buscamos un '{' y añadimos '}' al final para cerrarlo
        json_start = text.find('{')
        if json_start != -1:
            truncated = text[json_start:].rstrip()
            # Añadir comillas y llave de cierre si falta
            if not truncated.endswith('}'):
                # Cerrar strings abiertas y el objeto
                if truncated.count('"') % 2 == 1:
                    truncated += '"'
                truncated += '}'
            try:
                data = json.loads(truncated)
                logger.debug(f"Agente {self.player_id}: JSON reparado exitosamente.")
                return self._extract_fields(data)
            except json.JSONDecodeError:
                pass

        # ── Estrategia 3: extracción por regex de campos individuales ─────────
        # Funciona cuando el modelo genera algo como: msg": "espera", "act": "play"
        msg_match = re.search(r'(?:"msg"|"message")\s*:\s*"([^"]*)"', text)
        act_match = re.search(r'(?:"act"|"action")\s*:\s*"(\w+)"', text)

        if act_match:
            action = act_match.group(1).lower()
            if action not in ("wait", "play", "star"):
                action = "wait"
            message = msg_match.group(1) if msg_match else ""
            logger.debug(f"Agente {self.player_id}: extraído por regex.")
            return {"message": message, "action": action, "reasoning": ""}

        # ── Estrategia 4: heurística semántica pura ───────────────────────────
        logger.warning(
            f"Agente {self.player_id}: no pudo parsear JSON. "
            f"Salida cruda (primeros 120 chars): {repr(text[:120])}"
        )
        text_lower = text.lower()
        action = "wait"
        if any(w in text_lower for w in ["play", "juego", "jugar", "juega", '"play"']):
            action = "play"
        elif any(w in text_lower for w in ["star", "estrella", '"star"']):
            action = "star"

        lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith('{')]
        message = lines[0][:80] if lines else ""

        return {"message": message, "action": action, "reasoning": text[:150]}

    def _extract_fields(self, data: dict) -> dict:
        """Extrae los campos del dict parseado, aceptando claves cortas y largas."""
        # Clave corta 'act' tiene prioridad, luego 'action'
        action = data.get("act", data.get("action", "wait"))
        if isinstance(action, str):
            action = action.strip().lower()
        if action not in ("wait", "play", "star"):
            action = "wait"

        # Clave corta 'msg' tiene prioridad, luego 'message'
        message = data.get("msg", data.get("message", ""))
        if not isinstance(message, str):
            message = str(message)

        reasoning = data.get("reasoning", data.get("reason", ""))

        return {
            "message":   message[:150],
            "action":    action,
            "reasoning": reasoning[:200],
        }

    def get_card_to_play(self, obs: dict) -> Optional[int]:
        """Devuelve la carta mínima de la mano (la única válida a jugar)."""
        hand = obs["my_hand"]
        if not hand:
            return None
        return min(hand)


# ─── Carga del modelo base ────────────────────────────────────────────────────

def load_base_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "auto",
    use_4bit: bool = False,
    use_flash_attention: bool = False,
) -> tuple:
    """
    Carga el modelo base y el tokenizer.

    Modelos recomendados según hardware:
      - GPU 4GB:  Qwen2.5-1.5B-Instruct  (con 4bit ~1.5GB VRAM)
      - CPU 12GB: Qwen2.5-1.5B-Instruct  (float32, ~6GB RAM)
      - GPU 4GB:  Qwen2.5-0.5B-Instruct  (muy ligero, ~0.5GB VRAM)

    Args:
        model_name: nombre HuggingFace del modelo base
        device:     "auto", "cuda", "cpu", "mps"
        use_4bit:   cuantización 4-bit (bitsandbytes, solo GPU)
        use_flash_attention: Flash Attention 2 (solo GPU Ampere+)
    """
    logger.info(f"Cargando modelo: {model_name} en {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
    }

    if device == "auto":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if use_flash_attention:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    logger.info(f"Modelo cargado. Parámetros: {model.num_parameters():,}")
    return model, tokenizer


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    target_modules: list = None,
    lora_dropout: float = 0.0,  # 0 obligatorio: dropout in-place rompe autograd en training
) -> LoraConfig:
    """
    Configuración LoRA para fine-tuning eficiente.
    r=8 es un buen equilibrio para modelos pequeños.
    """
    if target_modules is None:
        # Módulos típicos para Qwen2 / LLaMA / Mistral
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )


def create_agents(
    model,
    tokenizer,
    num_players: int = 4,
    device: str = "cpu",
    lora_r: int = 8,
    shared_lora: bool = False,
) -> list:
    """
    Crea N agentes.

    shared_lora=True:  todos los agentes comparten los mismos pesos LoRA
                       (para ver comportamiento emergente colectivo)
    shared_lora=False: cada agente tiene su propio adaptador LoRA
                       (para ver especialización individual)
    """
    lora_config = create_lora_config(r=lora_r)

    if shared_lora:
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        agents = [
            TheMindAgent(i, model=peft_model, tokenizer=tokenizer, device=device)
            for i in range(num_players)
        ]
    else:
        # Cada agente con su propio adaptador — más memoria pero más diversidad
        agents = []
        for i in range(num_players):
            peft_model = get_peft_model(model, lora_config)
            agents.append(
                TheMindAgent(i, model=peft_model, tokenizer=tokenizer, device=device)
            )

    return agents