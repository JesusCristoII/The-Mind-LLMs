"""
utils.py — Utilidades para The Mind RL
Logging, análisis del lenguaje emergente, checkpoints, métricas.
"""
import json
import os
import re
import time
import logging
import torch
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ─── Métricas de entrenamiento ─────────────────────────────────────────────────

class TrainingMetrics:
    """Registra métricas durante el entrenamiento."""

    def __init__(self):
        self.episodes = []
        self.win_rates = []
        self.avg_rewards = []
        self.mistake_rates = []
        self.loss_history = []
        self.language_stats = defaultdict(list)  # evolución del lenguaje

    def record_episode(
        self,
        won: bool,
        total_reward: float,
        mistakes: int,
        level: int,
        messages: list,
        episode_num: int,
    ):
        self.episodes.append(episode_num)
        self.win_rates.append(1.0 if won else 0.0)
        self.avg_rewards.append(total_reward)
        self.mistake_rates.append(mistakes)
        self._analyze_language(messages, episode_num)

    def _analyze_language(self, messages: list, episode_num: int):
        """Extrae estadísticas del lenguaje usado en los mensajes."""
        if not messages:
            return
        all_words = []
        for msg in messages:
            text = msg.get("text", "")
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)

        if all_words:
            common = Counter(all_words).most_common(10)
            self.language_stats[episode_num] = {
                "common_words": common,
                "avg_msg_len":  sum(len(m.get("text","")) for m in messages) / len(messages),
                "num_messages": len(messages),
            }

    def get_win_rate(self, last_n: int = 50) -> float:
        if not self.win_rates:
            return 0.0
        return sum(self.win_rates[-last_n:]) / min(len(self.win_rates), last_n)

    def plot(self, save_path: str = "training_metrics.png"):
        """Genera gráficas de las métricas de entrenamiento."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("The Mind RL — Métricas de entrenamiento", fontsize=14)

        if self.episodes:
            # Win rate (rolling 50)
            ax = axes[0, 0]
            window = 50
            wr = np.convolve(self.win_rates, np.ones(window)/window, mode='valid')
            ax.plot(wr, color="#5563DE")
            ax.set_title("Win rate (rolling 50)")
            ax.set_ylabel("Tasa de victoria")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # Reward medio
            ax = axes[0, 1]
            rewards_smooth = np.convolve(self.avg_rewards, np.ones(window)/window, mode='valid')
            ax.plot(rewards_smooth, color="#2DB37A")
            ax.set_title("Reward medio (rolling 50)")
            ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3)

            # Errores por episodio
            ax = axes[1, 0]
            mistakes_smooth = np.convolve(self.mistake_rates, np.ones(window)/window, mode='valid')
            ax.plot(mistakes_smooth, color="#E85D30")
            ax.set_title("Errores por episodio (rolling 50)")
            ax.set_ylabel("Errores")
            ax.grid(True, alpha=0.3)

            # Longitud media de mensajes
            ax = axes[1, 1]
            if self.language_stats:
                eps = sorted(self.language_stats.keys())
                msg_lens = [self.language_stats[e]["avg_msg_len"] for e in eps]
                ax.plot(eps, msg_lens, color="#D4B033")
                ax.set_title("Longitud media de mensajes")
                ax.set_ylabel("Caracteres")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Métricas guardadas en {save_path}")

    def print_summary(self):
        print("\n" + "="*50)
        print("RESUMEN DE ENTRENAMIENTO")
        print("="*50)
        print(f"Episodios totales: {len(self.episodes)}")
        print(f"Win rate (último 50): {self.get_win_rate(50):.2%}")
        print(f"Reward medio (último 50): {np.mean(self.avg_rewards[-50:]):.3f}")
        print(f"Errores medio (último 50): {np.mean(self.mistake_rates[-50:]):.2f}")
        if self.language_stats:
            last_ep = max(self.language_stats.keys())
            stats = self.language_stats[last_ep]
            print(f"\nLenguaje más reciente (ep {last_ep}):")
            print(f"  Palabras más comunes: {stats['common_words'][:5]}")
            print(f"  Longitud media msg:   {stats['avg_msg_len']:.1f} chars")
        print("="*50)


# ─── Análisis del lenguaje emergente ──────────────────────────────────────────

class LanguageAnalyzer:
    """
    Analiza qué tipo de lenguaje emergente desarrollan los agentes.
    Detecta patrones, vocabulario recurrente, y estrategias comunicativas.
    """

    def __init__(self):
        self.message_log = []  # {episode, player, message, card, table_top}

    def log_message(self, episode: int, player: int, message: str,
                    card: Optional[int], table_top: int):
        if message:
            self.message_log.append({
                "episode": episode,
                "player":  player,
                "message": message,
                "card":    card,
                "table_top": table_top,
            })

    def get_vocabulary_evolution(self, n_bins: int = 5) -> dict:
        """Cómo cambia el vocabulario a lo largo del entrenamiento."""
        if not self.message_log:
            return {}

        eps = [m["episode"] for m in self.message_log]
        max_ep = max(eps)
        bin_size = max(1, max_ep // n_bins)

        bins = defaultdict(Counter)
        for msg in self.message_log:
            b = msg["episode"] // bin_size
            words = re.findall(r'\b\w+\b', msg["message"].lower())
            bins[b].update(words)

        return {b: counter.most_common(20) for b, counter in sorted(bins.items())}

    def detect_strategies(self) -> dict:
        """
        Detecta estrategias comunicativas emergentes.
        Clasifica mensajes en: urgencia, calma, duda, acuerdo, etc.
        """
        patterns = {
            "urgencia": ["ahora", "ya", "rápido", "urgente", "pronto", "espera"],
            "calma":    ["tranquilo", "espera", "tiempo", "paciencia", "despacio"],
            "duda":     ["creo", "quizás", "tal vez", "no sé", "hmm", "?"],
            "acuerdo":  ["ok", "vale", "bien", "entendido", "listo", "sí"],
            "señal":    ["lista", "ready", "voy", "go", "now"],
        }

        strategy_counts = defaultdict(int)
        for msg in self.message_log:
            text = msg["message"].lower()
            for strategy, keywords in patterns.items():
                if any(kw in text for kw in keywords):
                    strategy_counts[strategy] += 1

        return dict(strategy_counts)

    def print_report(self):
        print("\n" + "="*50)
        print("ANÁLISIS DEL LENGUAJE EMERGENTE")
        print("="*50)
        print(f"Total de mensajes analizados: {len(self.message_log)}")

        strategies = self.detect_strategies()
        if strategies:
            print("\nEstrategias detectadas:")
            for s, count in sorted(strategies.items(), key=lambda x: -x[1]):
                print(f"  {s}: {count} ocurrencias")

        vocab = self.get_vocabulary_evolution(n_bins=3)
        if vocab:
            print("\nEvolución del vocabulario:")
            for b, words in vocab.items():
                phase = ["inicio", "medio", "final"][min(b, 2)]
                top5 = [w for w, _ in words[:5]]
                print(f"  Fase {phase}: {top5}")
        print("="*50)

    def save_log(self, path: str = "language_log.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.message_log, f, ensure_ascii=False, indent=2)
        print(f"Log guardado en {path}")


# ─── Checkpoints ──────────────────────────────────────────────────────────────

def save_checkpoint(
    agents: list,
    metrics: TrainingMetrics,
    episode: int,
    output_dir: str = "checkpoints",
    optimizers: list = None,
):
    """
    Guarda los pesos LoRA de cada agente, las métricas y (opcionalmente)
    el estado de los optimizadores para poder reanudar el entrenamiento exacto.

    Estructura en disco:
        checkpoints/
        └── episode_N/
            ├── agent_0/
            │   ├── adapter_config.json
            │   └── adapter_model.safetensors
            ├── agent_1/ ...
            ├── optimizer_0.pt   (si se pasan optimizadores)
            ├── training_state.json
            └── metrics.json
    """
    checkpoint_dir = Path(output_dir) / f"episode_{episode}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Guardar adaptadores LoRA de cada agente
    for i, agent in enumerate(agents):
        agent_dir = checkpoint_dir / f"agent_{i}"
        agent.model.save_pretrained(str(agent_dir))

    # Guardar estado de los optimizadores (permite reanudar con el mismo momentum)
    if optimizers:
        for i, opt in enumerate(optimizers):
            opt_path = checkpoint_dir / f"optimizer_{i}.pt"
            torch.save(opt.state_dict(), str(opt_path))

    # Estado del entrenamiento (para saber desde dónde reanudar)
    state = {
        "episode":      episode,
        "num_agents":   len(agents),
        "shared_lora":  len(set(id(a.model) for a in agents)) == 1,
    }
    with open(checkpoint_dir / "training_state.json", "w") as f:
        json.dump(state, f, indent=2)

    # Métricas en ese punto
    metrics_data = {
        "episode":    episode,
        "win_rate":   metrics.get_win_rate(50),
        "avg_reward": float(np.mean(metrics.avg_rewards[-50:])) if metrics.avg_rewards else 0,
        "mistakes":   float(np.mean(metrics.mistake_rates[-50:])) if metrics.mistake_rates else 0,
        # Historial completo para poder restaurar las gráficas
        "win_rates":      metrics.win_rates,
        "avg_rewards":    metrics.avg_rewards,
        "mistake_rates":  metrics.mistake_rates,
    }
    with open(checkpoint_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Checkpoint guardado en: {checkpoint_dir}")
    for i in range(len(agents)):
        size = sum(
            p.stat().st_size for p in (checkpoint_dir / f"agent_{i}").rglob("*") if p.is_file()
        )
        print(f"  Agente {i}: {size / 1e6:.2f} MB")


def load_checkpoint(
    agents: list,
    episode: int,
    output_dir: str = "checkpoints",
    optimizers: list = None,
) -> TrainingMetrics:
    """
    Carga los pesos LoRA desde un checkpoint y restaura las métricas.
    Devuelve el objeto TrainingMetrics restaurado para continuar el historial.

    Uso típico:
        metrics = load_checkpoint(agents, episode=100, optimizers=optimizers)
        trainer.episode_count = 100  # reanudar desde aquí
    """
    checkpoint_dir = Path(output_dir) / f"episode_{episode}"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No se encontró checkpoint en: {checkpoint_dir}")

    # Leer estado guardado
    with open(checkpoint_dir / "training_state.json") as f:
        state = json.load(f)

    shared_lora = state.get("shared_lora", False)

    # Cargar adaptadores LoRA
    # set_adapter activa el adaptador ya existente con los pesos del checkpoint
    for i, agent in enumerate(agents):
        agent_dir = checkpoint_dir / f"agent_{i}"
        if not agent_dir.exists():
            print(f"  AVISO: no se encontró checkpoint para agente {i}, se mantienen pesos actuales.")
            continue

        if shared_lora and i > 0:
            # Con LoRA compartida todos los agentes usan el mismo modelo,
            # así que solo hay que cargar una vez
            continue

        # load_adapter carga los pesos en el adaptador 'default' ya inicializado
        agent.model.load_adapter(str(agent_dir), adapter_name="default")
        agent.model.set_adapter("default")
        print(f"  Agente {i} cargado desde {agent_dir}")

    if shared_lora:
        print("  (LoRA compartida: todos los agentes usan los mismos pesos)")

    # Cargar estado de los optimizadores
    if optimizers:
        for i, opt in enumerate(optimizers):
            opt_path = checkpoint_dir / f"optimizer_{i}.pt"
            if opt_path.exists():
                opt.load_state_dict(torch.load(str(opt_path), map_location="cpu"))
                print(f"  Optimizador {i} restaurado")
            else:
                print(f"  AVISO: no se encontró estado del optimizador {i}")

    # Restaurar métricas
    metrics = TrainingMetrics()
    metrics_path = checkpoint_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
        metrics.win_rates     = data.get("win_rates", [])
        metrics.avg_rewards   = data.get("avg_rewards", [])
        metrics.mistake_rates = data.get("mistake_rates", [])
        metrics.episodes      = list(range(len(metrics.win_rates)))
        print(f"  Métricas restauradas ({len(metrics.episodes)} episodios previos)")

    print(f"\nCheckpoint ep {episode} cargado. Listo para reanudar.")
    return metrics


def list_checkpoints(output_dir: str = "checkpoints") -> list:
    """Lista todos los checkpoints disponibles con sus métricas."""
    base = Path(output_dir)
    if not base.exists():
        print("No hay checkpoints guardados todavía.")
        return []

    checkpoints = []
    for ep_dir in sorted(base.glob("episode_*"), key=lambda p: int(p.name.split("_")[1])):
        ep = int(ep_dir.name.split("_")[1])
        metrics_path = ep_dir / "metrics.json"
        info = {"episode": ep, "path": str(ep_dir)}
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            info.update({
                "win_rate":   m.get("win_rate", 0),
                "avg_reward": m.get("avg_reward", 0),
                "mistakes":   m.get("mistakes", 0),
            })
        checkpoints.append(info)

    print(f"{'Episodio':>10} {'Win rate':>10} {'Reward':>10} {'Errores':>10}")
    print("-" * 44)
    for c in checkpoints:
        print(
            f"{c['episode']:>10d} "
            f"{c.get('win_rate', 0):>10.2%} "
            f"{c.get('avg_reward', 0):>10.2f} "
            f"{c.get('mistakes', 0):>10.2f}"
        )
    return checkpoints