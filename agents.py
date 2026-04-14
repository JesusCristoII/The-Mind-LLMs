"""
agents.py — Agentes LLM para The Mind
Usa modelos pequeños (~1-3B) con LoRA para caber en 4GB VRAM o 12GB RAM.
"""
import json
import ast
import re
import copy
import logging
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

# ─── Prompt base del agente ───────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un jugador del juego The Mind. Las reglas son:
- Hay cartas numeradas del 1 al 50 repartidas entre los jugadores.
- NO puedes decir el número de tu carta directamente.
- Debes colaborar con los demás para jugar las cartas en orden ascendente.
- Cada turno puedes enviar un mensaje corto a los demás Y/O jugar una carta.
- Juega tu carta cuando creas que es el momento adecuado.
- Si juegas fuera de orden, el equipo pierde una vida.
- Responde SIEMPRE con JSON válido, sin texto adicional antes ni después

Tu objetivo: coordinaros para jugar TODAS las cartas en orden sin errores.
Habla libremente pero SIN revelar tu número exacto ni ninguna operación matemática que lo descifre."""

ACTION_PROMPT = """Estado actual:
- Tus cartas: {my_hand}
- Última carta jugada: {table_top}
- Cartas jugadas: {played_cards}
- Vidas restantes: {lives}
- Estrellas: {stars}
- Mensajes recientes: {messages}

Responde SIEMPRE en este formato JSON exacto con tu mensaje, acción y razonamiento que hayas deducido y no pongas nada más:
{{
  "message": "mensaje corto para los demás (o vacío si no tienes nada que decir)",
  "action": "wait" o "play" o "star",
  "reasoning": "por qué tomas esta decisión (interno, no lo ven)"
}}

Reglas: no digas tu número. Puedes expresar urgencia, duda, confianza, o lo que veas conveniente siempere
que no digas expresamente tu número ni nada que pueda llevar a una deducción exacta del mismo."""


def format_messages(messages: list) -> str:
    if not messages:
        return "(ninguno)"
    return " | ".join([f"J{m['player']}: {m['text']}" for m in messages[-10:]])


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
        max_new_tokens: int = 512,
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
            played_cards=obs["played_cards"],
            lives=obs["lives"],
            stars=obs["stars"],
            messages=format_messages(obs.get("messages", [])),
        )
        # Formato chat
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": action_text},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{action_text}\n\nRespuesta:"
        return prompt

    @torch.no_grad()
    def generate_action(self, obs: dict) -> dict:
        """
        Genera mensaje + acción dado el estado observable.
        Devuelve dict con 'message', 'action', 'reasoning', 'raw_output'.
        """
        prompt = self.build_prompt(obs)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Evita el warning de transformers cuando el modelo trae max_length
        # en su generation_config y además usamos max_new_tokens.
        gen_config = None
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            gen_config = copy.deepcopy(self.model.generation_config)
            gen_config.max_length = None

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                repetition_penalty=1.2,
                temperature=0.3,    # 0.3 para más determinismo, 0.7 para más diversidad
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decodificar solo la parte generada
        input_len = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        # print(f"Agente {self.player_id} generó: {generated}")

        parsed = self._parse_output(generated, obs)
        parsed["raw_output"] = generated
        parsed["prompt"] = prompt

        # Guardar para RL
        self.generation_history.append({
            "prompt": prompt,
            "output": generated,
            "obs": obs,
        })

        return parsed

    def _parse_output(self, text: str, obs: dict) -> dict:
        """Parsea la salida JSON del modelo. Fallback a heurísticas."""
        # Intentar parsear JSON válido aunque venga con texto extra alrededor.
        # print(text)
        data = self._extract_structured_output(text)
        # print(data)
        if data is not None:
            action = str(data.get("action", "wait")).strip().lower()
            if action not in ("wait", "play", "star"):
                action = "wait"
            return {
                "message": str(data.get("message", ""))[:200],
                "action": action,
                "reasoning": str(data.get("reasoning", ""))[:300],
            }

        # Fallback heurístico
        logger.debug(
            "Agente %s: no pudo parsear JSON. Usando heuristica.",
            self.player_id,
        )
        action = "wait"
        if "play" in text.lower() or "juego" in text.lower() or "jugar" in text.lower():
            action = "play"

        # Extraer mensaje (primera línea razonable)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        message = lines[0][:100] if lines else ""

        return {
            "message":   message,
            "action":    action,
            "reasoning": text[:200],
        }

    def _extract_structured_output(self, text: str) -> Optional[dict]:
        """Extrae un dict estilo JSON del texto generado por el modelo."""
        # 1) Bloques markdown tipo ```json ... ```
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if fenced:
            parsed = self._parse_dict_like(fenced.group(1))
            if parsed is not None:
                return parsed

        # 2) JSONDecoder sobre cualquier subcadena que empiece con '{'
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch != "{":
                continue
            snippet = text[idx:]
            try:
                parsed, _ = decoder.raw_decode(snippet)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        # 3) Fallback por regex + parser tolerante
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return self._parse_dict_like(json_match.group())

        return None

    def _parse_dict_like(self, text: str) -> Optional[dict]:
        """Parsea texto de diccionario usando JSON y luego ast.literal_eval."""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict):
                return data
        except (SyntaxError, ValueError):
            pass

        return None

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
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    model.eval()

    logger.info(f"Modelo cargado. Parámetros: {model.num_parameters():,}")
    return model, tokenizer


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    target_modules: list = None,
    lora_dropout: float = 0.1,
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
        peft_model.config.use_cache = False
        if hasattr(peft_model, "generation_config"):
            peft_model.generation_config.use_cache = False
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
            peft_model.config.use_cache = False
            if hasattr(peft_model, "generation_config"):
                peft_model.generation_config.use_cache = False
            agents.append(
                TheMindAgent(i, model=peft_model, tokenizer=tokenizer, device=device)
            )

    return agents
