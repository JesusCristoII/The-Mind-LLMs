"""
trainer.py — Bucle de entrenamiento RL para The Mind

Implementa GRPO (Group Relative Policy Optimization) adaptado a texto,
similar a lo usado en DeepSeek-R1. También hay soporte para PPO clásico.

GRPO es preferible aquí porque:
- No necesita un value model separado (ahorra memoria)
- Funciona bien con recompensas sparse/delayed
- Más estable que PPO para LLMs pequeños
"""
import torch
import torch.nn.functional as F
import logging
from typing import Optional
from dataclasses import dataclass

from environment import TheMindEnv, GameState
from rewards import (
    compute_timing_bonus, compute_communication_quality,
    episode_reward, normalize_rewards, StepReward
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Hiperparámetros del entrenamiento RL."""
    # General
    num_episodes: int = 500
    num_levels: int = 3              # niveles del juego a entrenar
    group_size: int = 4              # tamaño del grupo para GRPO
    checkpoint_every: int = 50

    # Optimización
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_episodes: int = 20        # solo exploración, sin actualizar pesos

    # GRPO
    kl_coeff: float = 0.01           # penalización KL divergencia
    clip_ratio: float = 0.2          # ratio de clipping (como PPO)
    entropy_bonus: float = 0.01      # fomentar exploración

    # Juego
    max_turns_per_episode: int = 200
    messages_per_turn: bool = True   # si los agentes pueden mandar mensajes

    # Hardware
    device: str = "cpu"
    accumulate_grad_steps: int = 4   # gradient accumulation (simula batch grande)


class GRPOTrainer:
    """
    Entrenador GRPO para los agentes de The Mind.

    Flujo por episodio:
    1. Ejecutar N juegos (grupo) con los agentes actuales
    2. Calcular rewards para cada juego
    3. Calcular ventajas relativas al grupo (GRPO)
    4. Actualizar pesos con policy gradient + KL penalty
    """

    def __init__(
        self,
        agents: list,
        env: TheMindEnv,
        config: TrainerConfig,
        optimizers: Optional[list] = None,
    ):
        self.agents = agents
        self.env = env
        self.config = config

        # Un optimizer por agente (o compartido si shared LoRA)
        if optimizers is None:
            optimizers_by_model = {}
            self.optimizers = []
            for agent in agents:
                model_id = id(agent.model)
                if model_id not in optimizers_by_model:
                    optimizers_by_model[model_id] = torch.optim.AdamW(
                        agent.model.parameters(),
                        lr=config.lr,
                        weight_decay=0.01,
                    )
                self.optimizers.append(optimizers_by_model[model_id])
        else:
            self.optimizers = optimizers

        self.episode_count = 0

    # ─── Ejecución de un episodio ──────────────────────────────────────────────

    def run_episode(self, level: int = 1, training: bool = True) -> dict:
        """
        Ejecuta un episodio completo del juego.
        Devuelve trajectories con prompts, outputs y rewards.
        """
        state = self.env.reset(level=level)
        trajectories = {i: [] for i in range(len(self.agents))}
        total_reward = 0.0
        turn = 0

        while not self.env.is_done() and turn < self.config.max_turns_per_episode:
            turn += 1

            # ── Fase de comunicación ──────────────────────────────────────────
            if self.config.messages_per_turn:
                for agent in self.agents:
                    if state.hands[agent.player_id]:  # si tiene cartas
                        obs = self.env.get_observation(agent.player_id)
                        decision = agent.generate_action(obs)  # always no_grad internally

                        if decision["message"]:
                            self.env.send_message(agent.player_id, decision["message"])

                        # Calcular reward de comunicación
                        comm_reward = compute_communication_quality(
                            decision["message"], obs
                        )

                        trajectories[agent.player_id].append({
                            "prompt":  decision["prompt"],
                            "output":  decision["raw_output"],
                            "reward":  comm_reward,
                            "type":    "message",
                            "obs":     obs,
                        })

            # ── Fase de acción (jugar carta) ──────────────────────────────────
            # Determinamos qué agente debería jugar (el que tiene la carta más baja global)
            # Pero en el juego real, esto es lo que los agentes deben descubrir.
            # Aquí cada agente decide independientemente.

            played_this_turn = False
            for agent in self.agents:
                hand = state.hands[agent.player_id]
                if not hand:
                    continue

                obs = self.env.get_observation(agent.player_id)
                decision = agent.generate_action(obs)  # always no_grad internally

                if decision["action"] == "play":
                    card = agent.get_card_to_play(obs)
                    if card is not None:
                        result = self.env.play_card(agent.player_id, card)
                        step_reward = result.get("reward", 0.0)

                        # Bonus de timing (calculado a posteriori en training)
                        if training and result.get("correct", False):
                            # Contar cuántas cartas de otros eran menores
                            cards_below = sum(
                                1 for pid, a in enumerate(self.agents)
                                if pid != agent.player_id
                                for c in state.hands[pid]
                                if c < card
                            )
                            timing_b = compute_timing_bonus(
                                card, state.table_top,
                                sum(len(h) for h in state.hands.values()),
                                cards_below
                            )
                            step_reward += timing_b * 0.5

                        total_reward += step_reward
                        played_this_turn = True

                        trajectories[agent.player_id].append({
                            "prompt":  decision["prompt"],
                            "output":  decision["raw_output"],
                            "reward":  step_reward,
                            "type":    "play",
                            "obs":     obs,
                            "card":    card,
                            "correct": result.get("correct", False),
                        })

                elif decision["action"] == "star" and state.stars > 0:
                    self.env.use_star()

                if self.env.is_done():
                    break

            # Si nadie jugó en varios turnos, pequeña penalización anti-stall
            if not played_this_turn:
                total_reward -= 0.1

        # ── Reward de episodio ─────────────────────────────────────────────────
        ep_r = episode_reward(
            won=state.won or state.round_over,
            level=level,
            mistakes=state.mistakes,
            lives_remaining=state.lives,
            total_turns=turn,
        )
        total_reward += ep_r

        return {
            "trajectories": trajectories,
            "total_reward":  total_reward,
            "won":          state.won or state.round_over,
            "mistakes":     state.mistakes,
            "turns":        turn,
            "messages":     state.messages,
            "level":        level,
        }

    # ─── Cálculo de loss GRPO ─────────────────────────────────────────────────

    def compute_policy_loss(
        self,
        agent,
        prompt: str,
        output: str,
        advantage: float,
    ) -> Optional[torch.Tensor]:
        """
        Calcula el policy gradient loss para una transición (GRPO simplificado).

        Loss = -advantage * log_prob(output | prompt) - entropy_bonus * H(pi)

        IMPORTANTE: No pasamos `labels` al modelo para evitar que HuggingFace
        compute la cross-entropy internamente con operaciones in-place que
        corrompen el grafo de autograd. Calculamos el loss manualmente.
        """
        tokenizer = agent.tokenizer
        model = agent.model

        # Tokenizar prompt + output por separado para conocer el corte
        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
            add_special_tokens=False,
        )["input_ids"]
        output_ids = tokenizer(
            output,
            return_tensors="pt",
            truncation=True,
            max_length=200,
            add_special_tokens=False,
        )["input_ids"]

        if output_ids.shape[1] == 0:
            return None

        prompt_len = prompt_ids.shape[1]
        input_ids = torch.cat([prompt_ids, output_ids], dim=1).to(self.config.device)
        attention_mask = torch.ones_like(input_ids)

        # Forward pass limpio SIN labels — evita operaciones in-place internas
        model.train()
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits.float().clone()  # [1, seq_len, vocab]

        # Solo los logits que predicen tokens del output (shifted by 1)
        # logits[:, prompt_len-1:-1] predice los tokens output_ids[:, 0:]
        gen_logits = logits[:, prompt_len - 1:-1, :].float()  # [1, gen_len, vocab]
        target_ids = input_ids[:, prompt_len:].to(self.config.device)   # [1, gen_len]

        if gen_logits.shape[1] == 0:
            return None

        # Log-probs manualmente (sin in-place)
        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [1, gen_len]

        seq_log_prob = token_log_probs.mean()

        # Policy gradient loss
        adv_tensor = torch.tensor(advantage, dtype=torch.float32, device=self.config.device)
        pg_loss = -adv_tensor * seq_log_prob

        # Entropy bonus
        probs = F.softmax(gen_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_bonus * entropy

        return pg_loss + entropy_loss

    # ─── Actualización de pesos ───────────────────────────────────────────────

    def update(self, episode_results: list):
        """
        Actualiza los pesos de los agentes dado un grupo de resultados de episodio.
        Implementa GRPO: advantage = reward - mean(group_rewards).
        """
        # Calcular baseline del grupo
        group_rewards = [r["total_reward"] for r in episode_results]
        baseline = sum(group_rewards) / len(group_rewards)
        advantages_group = [r - baseline for r in group_rewards]

        # Normalizar ventajas
        adv_norm = normalize_rewards(advantages_group)

        total_loss_per_agent = {i: 0.0 for i in range(len(self.agents))}
        num_updates = {i: 0 for i in range(len(self.agents))}

        for ep_idx, (result, advantage) in enumerate(zip(episode_results, adv_norm)):
            trajectories = result["trajectories"]

            for agent in self.agents:
                pid = agent.player_id
                agent_traj = trajectories.get(pid, [])

                for step in agent_traj:
                    loss = self.compute_policy_loss(
                        agent=agent,
                        prompt=step["prompt"],
                        output=step["output"],
                        advantage=advantage,
                    )
                    if loss is not None:
                        # Gradient accumulation
                        loss = loss / self.config.accumulate_grad_steps
                        loss.backward()
                        total_loss_per_agent[pid] += loss.item()
                        num_updates[pid] += 1

        # Aplicar gradientes
        stepped_optimizers = set()
        for agent in self.agents:
            pid = agent.player_id
            opt = self.optimizers[pid]

            if id(opt) in stepped_optimizers:
                continue

            torch.nn.utils.clip_grad_norm_(
                agent.model.parameters(),
                self.config.max_grad_norm,
            )
            opt.step()
            opt.zero_grad()
            stepped_optimizers.add(id(opt))

            avg_loss = (
                total_loss_per_agent[pid] / num_updates[pid]
                if num_updates[pid] > 0 else 0.0
            )
            logger.debug(f"Agente {pid} — loss: {avg_loss:.4f}")

        return {
            pid: total_loss_per_agent[pid] / max(num_updates[pid], 1)
            for pid in range(len(self.agents))
        }

    # ─── Bucle principal de entrenamiento ─────────────────────────────────────

    def train(self, metrics=None, lang_analyzer=None, verbose: bool = True):
        """
        Bucle completo de entrenamiento.

        Args:
            metrics: instancia TrainingMetrics (de utils.py)
            lang_analyzer: instancia LanguageAnalyzer (de utils.py)
            verbose: imprimir progreso
        """
        from utils import save_checkpoint

        config = self.config
        level_schedule = self._build_level_schedule()

        for episode in range(config.num_episodes):
            self.episode_count = episode
            level = level_schedule[episode]

            # Recoger un grupo de episodios para GRPO
            group_results = []
            for _ in range(config.group_size):
                result = self.run_episode(
                    level=level,
                    training=(episode >= config.warmup_episodes),
                )
                group_results.append(result)

            # Actualizar pesos (si pasó el warmup)
            losses = {}
            if episode >= config.warmup_episodes:
                losses = self.update(group_results)

            # Métricas del grupo
            avg_reward = sum(r["total_reward"] for r in group_results) / len(group_results)
            avg_won = sum(1 for r in group_results if r["won"]) / len(group_results)
            avg_mistakes = sum(r["mistakes"] for r in group_results) / len(group_results)

            # Registrar
            for r in group_results:
                if metrics:
                    metrics.record_episode(
                        won=r["won"],
                        total_reward=r["total_reward"],
                        mistakes=r["mistakes"],
                        level=level,
                        messages=r["messages"],
                        episode_num=episode,
                    )
                if lang_analyzer:
                    for msg in r["messages"]:
                        # Aproximación de la carta del jugador en el momento del mensaje
                        lang_analyzer.log_message(
                            episode=episode,
                            player=msg["player"],
                            message=msg["text"],
                            card=None,
                            table_top=0,
                        )

            if verbose and episode % 10 == 0:
                print(
                    f"Ep {episode:4d} | Nivel {level} | "
                    f"Win: {avg_won:.2f} | Reward: {avg_reward:.2f} | "
                    f"Errores: {avg_mistakes:.1f} | "
                    f"Loss: {list(losses.values())[0]:.4f}" if losses else
                    f"Ep {episode:4d} | Nivel {level} | Win: {avg_won:.2f} | "
                    f"Reward: {avg_reward:.2f} | Errores: {avg_mistakes:.1f}"
                )

            # Checkpoint
            if episode > 0 and episode % config.checkpoint_every == 0:
                if metrics:
                    metrics.plot(f"metrics_ep{episode}.png")
                save_checkpoint(self.agents, metrics or _DummyMetrics(), episode)

        if verbose:
            print("\n¡Entrenamiento completado!")
            if metrics:
                metrics.print_summary()

    def _build_level_schedule(self) -> list:
        """
        Curriculum: empieza en nivel 1, va subiendo gradualmente.
        """
        schedule = []
        n = self.config.num_episodes
        levels = self.config.num_levels

        for i in range(n):
            progress = i / n
            if progress < 0.5:
                level = 1
            elif progress < 0.75:
                level = min(2, levels)
            else:
                level = min(3, levels)
            schedule.append(level)

        return schedule


class _DummyMetrics:
    """Placeholder por si no se pasan métricas."""
    episodes = []
    win_rates = []
    avg_rewards = []
    mistake_rates = []
    def get_win_rate(self, n=50): return 0.0