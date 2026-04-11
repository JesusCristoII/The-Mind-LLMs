"""
environment.py — Entorno del juego The Mind
"""
import random
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Estado completo de una partida."""
    num_players: int = 4
    level: int = 1
    hands: dict = field(default_factory=dict)       # {player_id: [cartas]}
    played_cards: list = field(default_factory=list) # cartas jugadas en orden
    table_top: int = 0                               # última carta jugada
    lives: int = 1
    stars: int = 0                                 # comodines para parar y votar
    round_over: bool = False
    game_over: bool = False
    won: bool = False
    current_turn: int = 0
    messages: list = field(default_factory=list)     # historial de mensajes
    mistakes: int = 0
    cards_per_player: int = 1

    def to_dict(self):
        return {
            "level": self.level,
            "played_cards": self.played_cards,
            "table_top": self.table_top,
            "lives": self.lives,
            "stars": self.stars,
            "round_over": self.round_over,
            "game_over": self.game_over,
            "won": self.won,
            "messages": self.messages[-10:],  # solo los últimos 10 mensajes
        }


class TheMindEnv:
    """
    Entorno del juego The Mind.

    En cada paso:
      1. Se pide a cada agente que envíe un mensaje (opcional).
      2. Se pide a cada agente que decida si juega carta (y cuál).
      3. Se calcula el reward y se actualiza el estado.

    Las cartas van del 1 al 50 (o 1-100 según variante).
    """

    MAX_CARD = 50  # versión simplificada

    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.state = None

    def reset(self, level: int = 1, lives: int = 1) -> GameState:
        """Inicia una nueva ronda."""
        deck = list(range(1, self.MAX_CARD + 1))
        random.shuffle(deck)

        cards_per_player = level
        hands = {}
        for i in range(self.num_players):
            hands[i] = sorted(deck[i * cards_per_player:(i + 1) * cards_per_player])

        self.state = GameState(
            num_players=self.num_players,
            level=level,
            hands=hands,
            played_cards=[],
            table_top=0,
            lives=lives,
            stars=0,
            cards_per_player=cards_per_player,
        )
        logger.info(f"Nueva ronda. Nivel {level}. Manos: {hands}")
        return self.state

    def get_observation(self, player_id: int) -> dict:
        """Lo que un jugador puede observar (SIN ver las cartas de los demás)."""
        state = self.state
        return {
            "player_id": player_id,
            "my_hand": state.hands[player_id],
            "played_cards": state.played_cards,
            "table_top": state.table_top,
            "lives": state.lives,
            "stars": state.stars,
            "messages": state.messages[-10:],
            "level": state.level,
            "num_players": state.num_players,
        }

    def send_message(self, player_id: int, message: str):
        """Añade un mensaje al canal de comunicación."""
        if message and message.strip():
            self.state.messages.append({
                "player": player_id,
                "text": message.strip()[:1024],  # limitar longitud
            })

    def play_card(self, player_id: int, card: int) -> dict:
        """
        Un jugador intenta jugar una carta.
        Devuelve info sobre si fue válida y el reward parcial.
        """
        state = self.state
        hand = state.hands[player_id]

        if card not in hand:
            return {"valid": False, "reason": "carta_no_en_mano", "reward": -1.0}

        min_card = min(hand)
        if card != min_card:
            # El jugador no jugó su carta más baja — penalización
            return {"valid": False, "reason": "no_es_minima", "reward": -0.5}

        if card <= state.table_top:
            # ¡Error! La carta es menor o igual que la última jugada
            state.lives -= 1
            state.mistakes += 1
            hand.remove(card)
            state.played_cards.append(card)

            # Eliminar del resto de jugadores las cartas menores que la jugada
            penalty_cards = []
            for pid in range(self.num_players):
                to_remove = [c for c in state.hands[pid] if c < card]
                for c in to_remove:
                    state.hands[pid].remove(c)
                    penalty_cards.append((pid, c))

            reward = -2.0 - (len(penalty_cards) * 0.5)
            logger.info(f"Error: jugador {player_id} jugó {card} sobre {state.table_top}. Vidas: {state.lives}")

            if state.lives <= 0:
                state.game_over = True
                state.won = False
                reward -= 5.0

            return {
                "valid": True,
                "correct": False,
                "card": card,
                "reward": reward,
                "penalty_cards": penalty_cards,
            }
        else:
            # ¡Jugada correcta!
            state.table_top = card
            hand.remove(card)
            state.played_cards.append(card)

            # Bonus por orden perfecto
            reward = 1.0

            # ¿Ronda completada?
            total_cards = sum(len(h) for h in state.hands.values())
            if total_cards == 0:
                state.round_over = True
                reward += 5.0
                logger.info(f"¡Ronda {state.level} completada!")

            return {
                "valid": True,
                "correct": True,
                "card": card,
                "reward": reward,
            }

    def use_star(self) -> dict:
        """Usar estrella: todos revelan su carta más baja."""
        state = self.state
        if state.stars <= 0:
            return {"success": False, "reason": "sin_estrellas"}
        state.stars -= 1
        revealed = {}
        for pid in range(self.num_players):
            if state.hands[pid]:
                lowest = min(state.hands[pid])
                state.hands[pid].remove(lowest)
                revealed[pid] = lowest
        logger.info(f"Estrella usada. Reveladas: {revealed}")
        return {"success": True, "revealed": revealed, "reward": -0.2}

    def is_done(self) -> bool:
        return self.state.game_over or self.state.round_over

    def all_hands_empty(self) -> bool:
        return all(len(h) == 0 for h in self.state.hands.values())
