"""
rewards.py — Funciones de recompensa para The Mind RL

El diseño de reward es crucial. Usamos recompensas densas (por cada acción)
y sparse (al final de la ronda) para guiar el aprendizaje.
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class StepReward:
    """Reward desglosado para un paso de la simulación."""
    play_correctness: float = 0.0   # +1 correcto, -2 error
    timing_bonus: float = 0.0       # bonus por jugar en el momento óptimo
    communication_bonus: float = 0.0 # bonus por mensajes informativos
    wait_penalty: float = 0.0       # penalización por esperar demasiado
    total: float = 0.0

    def compute_total(self, weights: dict = None):
        w = weights or {
            "play": 1.0,
            "timing": 0.5,
            "comm": 0.3,
            "wait": 0.2,
        }
        self.total = (
            w["play"]   * self.play_correctness +
            w["timing"] * self.timing_bonus +
            w["comm"]   * self.communication_bonus -
            w["wait"]   * self.wait_penalty
        )
        return self.total


def compute_timing_bonus(
    card: int,
    table_top: int,
    total_remaining_cards: int,
    num_cards_below_mine: int,
) -> float:
    """
    Bonus por jugar en el momento "correcto".

    La lógica: si no hay cartas de otros jugadores entre table_top y mi carta,
    es el momento perfecto para jugar. Si hay muchas cartas antes, debería esperar.

    num_cards_below_mine: cuántas cartas de otros jugadores son menores que la mía
    (esto se conoce a posteriori durante el entrenamiento, no durante la inferencia)
    """
    if num_cards_below_mine == 0:
        # Jugada perfecta: no había cartas menores de otros
        return 1.0
    else:
        # Penalización proporcional a cuántas cartas "debería haber esperado"
        return -0.5 * num_cards_below_mine


def compute_communication_quality(
    message: str,
    obs: dict,
) -> float:
    """
    Evalúa heurísticamente la calidad del mensaje.
    En RL real esto sería aprendido, pero esta heurística actúa como reward shaping.

    Penaliza:
      - Revelar el número exacto (trivial y rompe el juego)
      - Mensajes vacíos en momentos de alta incertidumbre
    Premia:
      - Mensajes que expresan urgencia cuando tienes carta baja
      - Mensajes que expresan calma cuando tienes carta alta
    """
    if not message or not message.strip():
        return 0.0  # sin mensaje, sin bonus

    message_lower = message.lower()
    my_hand = obs.get("my_hand", [])
    table_top = obs.get("table_top", 0)

    if not my_hand:
        return 0.0

    min_card = min(my_hand)
    card_range = 50  # MAX_CARD
    relative_position = min_card / card_range  # 0=muy baja, 1=muy alta

    # Penalizar si revela el número exacto (heurística simple)
    for card in my_hand:
        if str(card) in message:
            return -1.0

    # Penalizar frases demasiado explícitas
    forbidden_phrases = ["tengo el", "mi carta es", "tengo un", "mi número"]
    for phrase in forbidden_phrases:
        if phrase in message_lower:
            return -0.5

    # Bonus por expresar urgencia cuando la carta es relativamente baja
    urgency_words = ["urgente", "ahora", "ya", "rápido", "espera", "pronto"]
    urgency_score = sum(1 for w in urgency_words if w in message_lower)

    if relative_position < 0.3 and urgency_score > 0:
        return 0.5  # carta baja + urgencia = coherente
    elif relative_position > 0.7 and urgency_score == 0:
        return 0.3  # carta alta + calma = coherente

    # Pequeño bonus por comunicarse (cualquier mensaje)
    return 0.1


def episode_reward(
    won: bool,
    level: int,
    mistakes: int,
    lives_remaining: int,
    total_turns: int,
) -> float:
    """
    Reward al final de la ronda/episodio completo.
    Se combina con los rewards por paso para el cálculo GRPO/PPO.
    """
    if won:
        base = 10.0 * level          # más reward por niveles difíciles
        efficiency = max(0, 3 - mistakes) * 2.0  # bonus por pocos errores
        speed = max(0, 1.0 - total_turns / 100)  # bonus por rapidez
        return base + efficiency + speed
    else:
        # Parcial: recompensar el progreso aunque se pierda
        return -5.0 + (level - 1) * 1.0 + lives_remaining * 0.5


def normalize_rewards(rewards: list, eps: float = 1e-8) -> list:
    """Normaliza los rewards (mean=0, std=1) para estabilizar el entrenamiento."""
    if len(rewards) < 2:
        return rewards
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance + eps)
    return [(r - mean) / std for r in rewards]


def compute_group_reward(
    trajectories: list,
    won: bool,
    level: int,
    mistakes: int,
    lives_remaining: int,
) -> list:
    """
    GRPO-style: calcula el reward relativo dentro de un grupo de trayectorias.
    Compara cada trayectoria contra la media del grupo.

    trajectories: lista de dicts con 'step_rewards' y metadata
    """
    episode_rewards = []
    for traj in trajectories:
        step_sum = sum(traj.get("step_rewards", [0]))
        ep = episode_reward(won, level, mistakes, lives_remaining, len(traj.get("step_rewards", [])))
        episode_rewards.append(step_sum + ep)

    # Baseline: media del grupo
    baseline = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    advantages = [r - baseline for r in episode_rewards]

    return advantages
