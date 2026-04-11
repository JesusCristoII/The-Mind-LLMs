"""
sft_trainer.py — Entrenamiento supervisado (SFT) previo al RL

Convierte cada ejemplo del dataset en un par (prompt, respuesta JSON),
y hace fine-tuning con cross-entropy loss estándar sobre los tokens
de la respuesta (el prompt se enmascara para no calcular loss sobre él).

Uso desde el notebook:
    from sft_trainer import run_sft
    run_sft(agents, tokenizer, dataset_path="sft_dataset.json", epochs=3)
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ─── Reutilizamos la misma plantilla de prompt que agents.py ──────────────────

SYSTEM_PROMPT = """Eres un jugador del juego The Mind. Reglas:
- Cartas del 1 al 50. No puedes decir tu número.
- El equipo debe jugar todas las cartas en orden ascendente.
- Responde SIEMPRE con JSON válido, sin texto adicional antes ni después."""

ACTION_PROMPT = """Estado:
- Mis cartas: {my_hand}
- Última carta en mesa: {table_top}
- Vidas: {lives} | Estrellas: {stars}
- Mensajes: {messages}

Responde SOLO con este JSON (sin markdown, sin texto extra):
{{"msg": "texto corto o vacío", "act": "wait"}}

Donde "act" puede ser: "wait" (esperar), "play" (jugar mi carta más baja), "star" (usar estrella).
No incluyas tu número en "msg". Ejemplo válido: {{"msg": "creo que es pronto", "act": "wait"}}"""


def format_messages(messages: list) -> str:
    if not messages:
        return "(ninguno)"
    return " | ".join([f"J{m['player']}: {m['text']}" for m in messages[-5:]])


def example_to_prompt_and_target(example: dict, tokenizer) -> tuple[str, str]:
    """
    Convierte un ejemplo del dataset en (prompt, target_json).
    El target es el JSON mínimo que el modelo debe aprender a generar.
    """
    action_text = ACTION_PROMPT.format(
        my_hand=example["my_hand"],
        table_top=example["table_top"],
        lives=example["lives"],
        stars=example["stars"],
        messages=format_messages(example.get("messages", [])),
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": action_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{SYSTEM_PROMPT}\n\n{action_text}\n\nRespuesta:"

    # Target: el JSON que queremos que el modelo aprenda a generar
    target = json.dumps(
        {"msg": example["msg"], "act": example["act"]},
        ensure_ascii=False,
    )

    return prompt, target


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TheMindSFTDataset(Dataset):
    """
    Dataset de pares (prompt_ids, target_ids) con máscara sobre el prompt.
    El loss solo se calcula sobre los tokens del target JSON.
    """

    def __init__(
        self,
        examples: list,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        skipped = 0
        for ex in examples:
            prompt, target = example_to_prompt_and_target(ex, tokenizer)
            item = self._encode(prompt, target)
            if item is not None:
                self.items.append(item)
            else:
                skipped += 1

        if skipped:
            logger.warning(f"Se descartaron {skipped} ejemplos por exceder max_length={max_length}")
        logger.info(f"Dataset SFT: {len(self.items)} ejemplos listos")

    def _encode(self, prompt: str, target: str) -> Optional[dict]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(
            target + self.tokenizer.eos_token,
            add_special_tokens=False,
        )

        total = len(prompt_ids) + len(target_ids)
        if total > self.max_length:
            return None

        input_ids = prompt_ids + target_ids

        # Labels: -100 en las posiciones del prompt (no calcular loss),
        #          token_id real en las posiciones del target
        labels = [-100] * len(prompt_ids) + target_ids

        # Padding hasta max_length
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padding = self.max_length - total
        input_ids = input_ids + [pad_id] * padding
        labels    = labels    + [-100]   * padding
        attn_mask = [1] * total + [0] * padding

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,    dtype=torch.long),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ─── Función principal de SFT ─────────────────────────────────────────────────

def run_sft(
    agents: list,
    tokenizer,
    dataset_path: str = "sft_dataset.json",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_length: int = 512,
    max_grad_norm: float = 1.0,
    save_dir: str = "checkpoints/sft",
    device: str = "cpu",
    shared_lora: bool = True,
) -> dict:
    """
    Ejecuta el SFT sobre todos los agentes (o solo uno si shared_lora=True).

    Args:
        agents:       lista de TheMindAgent
        tokenizer:    tokenizer del modelo base
        dataset_path: ruta al JSON con los ejemplos
        epochs:       épocas de entrenamiento
        batch_size:   ejemplos por paso (reducir si hay OOM)
        lr:           learning rate (más alto que en RL está bien para SFT)
        max_length:   longitud máxima de tokens (prompt + target)
        max_grad_norm: clipping de gradientes
        save_dir:     dónde guardar los adaptadores tras el SFT
        device:       "cpu" o "cuda"
        shared_lora:  si True, entrena solo el agente 0 (todos comparten pesos)

    Returns:
        dict con 'losses' (lista por epoch) y 'final_loss'
    """
    # Cargar dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        examples = json.load(f)

    logger.info(f"Cargados {len(examples)} ejemplos de SFT desde {dataset_path}")

    dataset    = TheMindSFTDataset(examples, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Decidir qué agentes entrenar
    agents_to_train = [agents[0]] if shared_lora else agents

    all_losses = []

    for agent_idx, agent in enumerate(agents_to_train):
        label = "compartido" if shared_lora else f"agente {agent_idx}"
        logger.info(f"Iniciando SFT — modelo {label}")

        agent.model.train()
        optimizer = torch.optim.AdamW(
            agent.model.parameters(),
            lr=lr,
            weight_decay=0.01,
        )

        # Scheduler: cosine decay
        total_steps = epochs * len(dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr * 0.1
        )

        epoch_losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            num_batches  = 0

            for batch in dataloader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = agent.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Loss NaN/Inf detectado, skipping batch")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.model.parameters(), max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                num_batches  += 1

            avg_loss = running_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            logger.info(
                f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f} "
                f"| lr: {scheduler.get_last_lr()[0]:.2e}"
            )
            print(f"  [{label}] Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

        agent.model.eval()
        all_losses.extend(epoch_losses)

        # Guardar adaptadores SFT
        save_path = Path(save_dir) / (f"agent_{agent_idx}" if not shared_lora else "shared")
        save_path.mkdir(parents=True, exist_ok=True)
        agent.model.save_pretrained(str(save_path))
        logger.info(f"Adaptador SFT guardado en {save_path}")

    final_loss = all_losses[-1] if all_losses else float("nan")
    print(f"\nSFT completado. Loss final: {final_loss:.4f}")

    return {"losses": all_losses, "final_loss": final_loss}


def verify_sft_quality(agents: list, tokenizer, num_samples: int = 5, device: str = "cpu"):
    """
    Prueba rápida tras el SFT: genera respuestas de muestra y comprueba
    que el formato JSON es correcto. Imprime los resultados.
    """
    import re

    test_cases = [
        {"my_hand": [4],  "table_top": 0,  "lives": 3, "stars": 1, "messages": []},
        {"my_hand": [45], "table_top": 0,  "lives": 3, "stars": 1, "messages": []},
        {"my_hand": [12], "table_top": 10, "lives": 2, "stars": 1, "messages": [{"player": 1, "text": "voy pronto"}]},
        {"my_hand": [33], "table_top": 10, "lives": 2, "stars": 1, "messages": [{"player": 1, "text": "voy pronto"}]},
        {"my_hand": [7],  "table_top": 5,  "lives": 1, "stars": 0, "messages": []},
    ][:num_samples]

    print("\n" + "="*60)
    print("VERIFICACIÓN POST-SFT")
    print("="*60)

    agent = agents[0]
    agent.model.eval()
    ok = 0

    for i, ex in enumerate(test_cases):
        # Construir solo el prompt, sin necesitar msg/act (son casos de prueba)
        action_text = ACTION_PROMPT.format(
            my_hand=ex["my_hand"],
            table_top=ex["table_top"],
            lives=ex["lives"],
            stars=ex["stars"],
            messages=format_messages(ex.get("messages", [])),
        )
        if hasattr(tokenizer, "apply_chat_template"):
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": action_text},
            ]
            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{action_text}\n\nRespuesta:"

        prompt_with_start = prompt + '{\"'

        inputs = tokenizer(
            prompt_with_start, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            out = agent.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,   # greedy para evaluación
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = '{"' + tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        # Intentar parsear
        json_match = re.search(r'\{[^{}]*\}', generated)
        parsed_ok = False
        act = "?"
        msg = "?"
        if json_match:
            try:
                data = json.loads(json_match.group())
                act = data.get("act", "?")
                msg = data.get("msg", "?")
                parsed_ok = act in ("wait", "play", "star")
            except json.JSONDecodeError:
                pass

        status = "✓" if parsed_ok else "✗"
        if parsed_ok:
            ok += 1

        print(f"\nEjemplo {i+1} {status}")
        print(f"  Mano: {ex['my_hand']} | Mesa: {ex['table_top']} | Vidas: {ex['lives']}")
        print(f"  Generado: {generated[:100]}")
        print(f"  Parseado → act={act}, msg='{msg}'")

    print(f"\nResultado: {ok}/{num_samples} respuestas con JSON válido")
    print("="*60)
    return ok / num_samples