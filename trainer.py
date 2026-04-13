"""
trainer.py — Bucle de entrenamiento RL para The Mind

Implementa GRPO (Group Relative Policy Optimization) adaptado a texto.

Frecuencia de actualización de pesos:
  - Cada `group_size` episodios se recoge un grupo de partidas.
  - Al final de cada grupo se llama a update() -> backward() -> optimizer.step().
  - Es decir, los pesos se actualizan CADA group_size episodios (por defecto cada 4).
  - Durante los primeros `warmup_episodes` solo se recopilan datos, sin actualizar.
"""
import torch
import torch.nn.functional as F
import logging
from typing import Optional
from dataclasses import dataclass

from environment import TheMindEnv, GameState
from rewards import (
    compute_timing_bonus, compute_communication_quality,
    episode_reward, normalize_rewards,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    num_episodes: int = 500
    num_levels: int = 3
    group_size: int = 4          # episodios por actualización GRPO
    checkpoint_every: int = 50

    lr: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_episodes: int = 20    # solo exploración, sin backprop

    kl_coeff: float = 0.01
    clip_ratio: float = 0.2
    entropy_bonus: float = 0.01

    max_turns_per_episode: int = 200
    messages_per_turn: bool = True

    device: str = "cpu"
    accumulate_grad_steps: int = 4
    checkpoint_dir: str = "checkpoints"  # ruta donde guardar checkpoints


class GRPOTrainer:
    def __init__(self, agents, env, config: TrainerConfig, optimizers=None):
        self.agents = agents
        self.env = env
        self.config = config
        self.episode_count = 0

        if optimizers is None:
            self.optimizers = [
                torch.optim.AdamW(a.model.parameters(), lr=config.lr, weight_decay=0.01)
                for a in agents
            ]
        else:
            self.optimizers = optimizers

    # ──────────────────────────────────────────────────────────────────────────
    # Ejecución de episodio — generate() SIEMPRE sin grafo
    # ──────────────────────────────────────────────────────────────────────────

    def run_episode(self, level: int = 1) -> dict:
        state = self.env.reset(level=level)
        trajectories = {i: [] for i in range(len(self.agents))}
        total_reward = 0.0
        turn = 0

        while not self.env.is_done() and turn < self.config.max_turns_per_episode:
            turn += 1

            # Fase de comunicación — siempre no_grad (generate lo garantiza)
            if self.config.messages_per_turn:
                for agent in self.agents:
                    if not state.hands[agent.player_id]:
                        continue
                    obs = self.env.get_observation(agent.player_id)
                    decision = agent.generate_action(obs)

                    if decision["message"]:
                        self.env.send_message(agent.player_id, decision["message"])

                    comm_reward = compute_communication_quality(decision["message"], obs)
                    trajectories[agent.player_id].append({
                        "prompt": decision["prompt"],
                        "output": decision["raw_output"],
                        "reward": comm_reward,
                        "type":   "message",
                        "obs":    obs,
                    })

            # Fase de acción
            played_this_turn = False
            for agent in self.agents:
                if not state.hands[agent.player_id]:
                    continue
                if self.env.is_done():
                    break

                obs = self.env.get_observation(agent.player_id)
                decision = agent.generate_action(obs)

                if decision["action"] == "play":
                    card = agent.get_card_to_play(obs)
                    if card is not None:
                        result = self.env.play_card(agent.player_id, card)
                        step_reward = result.get("reward", 0.0)

                        cards_below = sum(
                            1 for pid in range(len(self.agents))
                            if pid != agent.player_id
                            for c in state.hands[pid] if c < card
                        )
                        timing_b = compute_timing_bonus(
                            card, state.table_top,
                            sum(len(h) for h in state.hands.values()),
                            cards_below,
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

            if not played_this_turn:
                total_reward -= 0.1

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
            "won":           state.won or state.round_over,
            "mistakes":      state.mistakes,
            "turns":         turn,
            "messages":      state.messages,
            "level":         level,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Loss GRPO — forward pass limpio, separado del generate()
    # ──────────────────────────────────────────────────────────────────────────

    def compute_policy_loss(self, agent, prompt: str, output: str, advantage: float) -> Optional[torch.Tensor]:
        """
        Forward pass completamente independiente del generate() anterior.
        - Sin `labels` para evitar cross-entropy in-place de HuggingFace.
        - lora_dropout desactivado en eval temporalmente... NO: ponemos
          dropout=0 en la config desde create_lora_config.
        - Loss calculado manualmente sobre los logits del output.
        """
        tok = agent.tokenizer
        device = self.config.device

        # Tokenizar por separado para saber exactamente dónde empieza el output
        enc_prompt = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=768, add_special_tokens=True)
        enc_output = tok(output, return_tensors="pt", truncation=True,
                         max_length=150, add_special_tokens=False)

        prompt_len = enc_prompt["input_ids"].shape[1]
        out_len    = enc_output["input_ids"].shape[1]

        if out_len == 0:
            return None

        input_ids = torch.cat(
            [enc_prompt["input_ids"], enc_output["input_ids"]], dim=1
        ).to(device)
        attn_mask = torch.ones_like(input_ids).to(device)

        # Forward limpio — model.train() para que los pesos LoRA actualicen,
        # pero NO pasamos labels (evita operaciones in-place en HF)
        agent.model.train()
        with torch.enable_grad():
            out = agent.model(input_ids=input_ids, attention_mask=attn_mask)

        # Logits para predecir los tokens del output (shifted)
        # posición [prompt_len-1] predice el primer token del output
        gen_logits  = out.logits[:, prompt_len - 1 : prompt_len - 1 + out_len, :].float()
        target_ids  = enc_output["input_ids"].to(device)

        if gen_logits.shape[1] == 0:
            return None

        # Log-probs manuales — sin ninguna operación in-place
        log_probs       = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        seq_log_prob    = token_log_probs.mean()

        adv = torch.tensor(float(advantage), dtype=torch.float32, device=device)
        pg_loss = -(adv * seq_log_prob)

        # Entropy bonus
        probs        = F.softmax(gen_logits, dim=-1)
        entropy      = -(probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_bonus * entropy

        return pg_loss + entropy_loss

    # ──────────────────────────────────────────────────────────────────────────
    # Actualización de pesos
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, episode_results: list) -> dict:
        group_rewards = [r["total_reward"] for r in episode_results]
        baseline      = sum(group_rewards) / len(group_rewards)
        adv_norm      = normalize_rewards([r - baseline for r in group_rewards])

        total_loss_per_agent = {i: 0.0 for i in range(len(self.agents))}
        num_updates          = {i: 0    for i in range(len(self.agents))}

        for opt in self.optimizers:
            opt.zero_grad()

        for result, advantage in zip(episode_results, adv_norm):
            for agent in self.agents:
                pid = agent.player_id
                for step in result["trajectories"].get(pid, []):
                    loss = self.compute_policy_loss(
                        agent, step["prompt"], step["output"], advantage
                    )
                    if loss is None:
                        continue
                    (loss / self.config.accumulate_grad_steps).backward()
                    total_loss_per_agent[pid] += loss.item()
                    num_updates[pid]          += 1

        for agent in self.agents:
            pid = agent.player_id
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), self.config.max_grad_norm)
            self.optimizers[pid].step()
            self.optimizers[pid].zero_grad()

        return {
            pid: total_loss_per_agent[pid] / max(num_updates[pid], 1)
            for pid in range(len(self.agents))
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Bucle principal
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, metrics=None, lang_analyzer=None, verbose: bool = True):
        from utils import save_checkpoint

        config         = self.config
        level_schedule = self._build_level_schedule()

        for episode in range(self.episode_count, config.num_episodes):
            self.episode_count = episode
            level = level_schedule[episode]

            # Recoger group_size episodios
            group_results = [self.run_episode(level=level) for _ in range(config.group_size)]

            # Actualizar pesos tras el warmup
            losses = {}
            if episode >= config.warmup_episodes:
                losses = self.update(group_results)

            avg_reward   = sum(r["total_reward"] for r in group_results) / len(group_results)
            avg_won      = sum(1 for r in group_results if r["won"]) / len(group_results)
            avg_mistakes = sum(r["mistakes"] for r in group_results) / len(group_results)

            for r in group_results:
                if metrics:
                    metrics.record_episode(
                        won=r["won"], total_reward=r["total_reward"],
                        mistakes=r["mistakes"], level=level,
                        messages=r["messages"], episode_num=episode,
                    )
                if lang_analyzer:
                    for msg in r["messages"]:
                        lang_analyzer.log_message(
                            episode=episode, player=msg["player"],
                            message=msg["text"], card=None, table_top=0,
                        )

            if verbose and episode % 10 == 0:
                loss_str = f" | Loss: {list(losses.values())[0]:.4f}" if losses else ""
                print(
                    f"Ep {episode:4d} | Nivel {level} | "
                    f"Win: {avg_won:.2f} | Reward: {avg_reward:.2f} | "
                    f"Errores: {avg_mistakes:.1f}{loss_str}"
                )

            if episode > 0 and episode % config.checkpoint_every == 0:
                save_checkpoint(
                    self.agents,
                    metrics or _DummyMetrics(),
                    episode,
                    output_dir=getattr(config, 'checkpoint_dir', 'checkpoints'),
                )

        if verbose:
            print("\n¡Entrenamiento completado!")
            if metrics:
                metrics.print_summary()

    def _build_level_schedule(self) -> list:
        n      = self.config.num_episodes
        levels = self.config.num_levels
        schedule = []
        for i in range(n):
            p = i / n
            if p < 0.5:   level = 1
            elif p < 0.75: level = min(2, levels)
            else:          level = min(3, levels)
            schedule.append(level)
        return schedule


class _DummyMetrics:
    episodes = []; win_rates = []; avg_rewards = []; mistake_rates = []
    def get_win_rate(self, n=50): return 0.0