"""
N=100 Final Benchmark — Key Policies Only.

Evaluates the 4 most important policies at N=100 episodes each (vs the
standard N=10/N=20 used in evaluate.py). This gives publishable-quality
statistics with tight confidence intervals.

Policies evaluated:
  1. Expert (rule-based)          — deterministic baseline
  2. PPO Curriculum v2 (cv2/3M)   — best speed-only policy
  3. PPO Pit v4 D36 (d36)         — full-unfreeze breakthrough (3 pits)
  4. PPO Pit v4 D37 (d37)         — project best pit policy (3 pits, 0.319 m)

Saves results to: Notes/benchmark_n100.txt
Saves plots to:   plots/benchmark_n100_fixed.png
                  plots/benchmark_n100_random.png

Runtime: ~25 minutes on CPU (4 policies × 100 eps × 2 modes × ~2s/ep)
"""

import sys
import math
from pathlib import Path
from datetime import datetime

import numpy as np

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.evaluate import run_episodes, make_ppo_policy, make_expert_policy
from rl.evaluate import print_comparison_table
from rl.evaluate import plot_bar_comparison, plot_episode_distributions
from rl.pit_aware_policy import PitAwarePolicy  # noqa: F401 — must import before PPO.load
from env.f1_env import F1Env

import numpy as np


def mean_ci95(values):
    """Return (mean, 95% CI half-width) assuming normal distribution."""
    n = len(values)
    mean = np.mean(values)
    std  = np.std(values, ddof=1)
    ci   = 1.96 * std / math.sqrt(n)
    return mean, ci


def run_benchmark(n_episodes=100):
    device = "cpu"

    # ── Environments ──────────────────────────────────────────────────────────
    env_standard = F1Env()
    env_pit      = F1Env(tyre_degradation=True, pit_stops=True,
                         voluntary_pit_reward=True)

    # ── Policy definitions ────────────────────────────────────────────────────
    policies = [
        {
            "name":    "Expert (rule-based)",
            "color":   "#FF6B35",
            "fn":      lambda: make_expert_policy(env_standard),
            "env":     env_standard,
        },
        {
            "name":    "PPO Curriculum v2 (cv2)",
            "color":   "#00D4FF",
            "fn":      lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_curriculum_v2.zip"), device, obs_dim=11),
            "env":     env_standard,
            "file":    "ppo_curriculum_v2.zip",
        },
        {
            "name":    "PPO Pit D36 (d36)",
            "color":   "#1B5E20",
            "fn":      lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_pit_v4_d36.zip"), device, obs_dim=12),
            "env":     env_pit,
            "file":    "ppo_pit_v4_d36.zip",
        },
        {
            "name":    "PPO Pit D37 (d37)",
            "color":   "#2E7D32",
            "fn":      lambda: make_ppo_policy(
                str(project_root / "rl" / "ppo_pit_v4_d37.zip"), device, obs_dim=12),
            "env":     env_pit,
            "file":    "ppo_pit_v4_d37.zip",
        },
    ]

    # Skip policies whose model file doesn't exist
    available = []
    for p in policies:
        f = p.get("file")
        if f and not (project_root / "rl" / f).exists():
            print(f"[Skip] {p['name']} — {f} not found")
            continue
        available.append(p)

    results = {}
    for mode_label, fixed_start in [("RANDOM START", False), ("FIXED START", True)]:
        print(f"\n{'═'*70}")
        print(f"  {mode_label}  (N={n_episodes} episodes each)")
        print(f"{'═'*70}")

        summaries = []
        for p in available:
            print(f"\n  Evaluating: {p['name']} ...")
            policy_fn = p["fn"]()
            summary = run_episodes(
                env=p["env"],
                policy_fn=policy_fn,
                policy_name=p["name"],
                policy_color=p["color"],
                n_episodes=n_episodes,
                record_one_trajectory=True,
                fixed_start=fixed_start,
            )
            summaries.append(summary)

            rew_mean, rew_ci = mean_ci95([r.total_reward for r in summary.all_results])
            print(f"    Reward:  {rew_mean:.1f} ± {rew_ci:.1f} (95% CI)")
            print(f"    Laps:    {summary.avg_laps_completed:.2f}")
            print(f"    Speed:   {summary.avg_speed_ms:.2f} m/s")
            print(f"    Lateral: {summary.avg_lateral_error_m:.3f} m")
            print(f"    Pits:    {summary.avg_pit_count:.2f}")
            print(f"    Compl:   {summary.lap_completion_rate*100:.1f}%")

        results[mode_label] = summaries
        print_comparison_table(summaries, title=f"{mode_label} — N={n_episodes}")

    return results


def write_report(results, n_episodes, output_path):
    lines = []
    lines.append("=" * 78)
    lines.append(f"N=100 FINAL BENCHMARK — Race Strategy Simulator")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"N={n_episodes} episodes per policy, deterministic rollouts")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Policies: Expert, PPO Curriculum v2 (cv2), PPO Pit D36, PPO Pit D37")
    lines.append("Environments:")
    lines.append("  Standard:  F1Env()  (11D obs, no tyres, no pit)")
    lines.append("  Pit:       F1Env(tyre_degradation=True, pit_stops=True,")
    lines.append("                   voluntary_pit_reward=True)  (12D obs, 3D action)")
    lines.append("Evaluation: deterministic=True (mean action, no sampling)")
    lines.append("")

    for mode_label, summaries in results.items():
        lines.append("─" * 78)
        lines.append(f"{mode_label}  (N={n_episodes})")
        lines.append("─" * 78)
        lines.append(f"  {'Policy':<28} {'Reward':>10} {'95% CI':>10} {'Laps':>6} "
                     f"{'Speed':>8} {'Lateral':>9} {'Pits':>6} {'Compl%':>7}")
        lines.append("  " + "─" * 76)
        for s in summaries:
            rew_mean, rew_ci = mean_ci95([r.total_reward for r in s.all_results])
            lines.append(
                f"  {s.name:<28} {rew_mean:>10.1f} {rew_ci:>9.1f} "
                f"{s.avg_laps_completed:>6.2f} {s.avg_speed_ms:>7.2f}m/s "
                f"{s.avg_lateral_error_m:>8.3f}m {s.avg_pit_count:>5.2f} "
                f"{s.lap_completion_rate*100:>6.1f}%"
            )
        lines.append("")

    lines.append("=" * 78)
    lines.append("KEY FINDINGS")
    lines.append("=" * 78)

    # Extract key numbers if both modes ran
    if "FIXED START" in results:
        fs = {s.name: s for s in results["FIXED START"]}
        cv2_key = next((k for k in fs if "cv2" in k.lower() or "curriculum" in k.lower()), None)
        d37_key = next((k for k in fs if "d37" in k.lower()), None)
        exp_key = next((k for k in fs if "expert" in k.lower()), None)

        if cv2_key and d37_key:
            cv2 = fs[cv2_key]
            d37 = fs[d37_key]
            d37_r_mean, d37_r_ci = mean_ci95([r.total_reward for r in d37.all_results])
            cv2_r_mean, cv2_r_ci = mean_ci95([r.total_reward for r in cv2.all_results])
            lines.append("")
            lines.append(f"Best pit policy (D37) vs speed champion (cv2):")
            lines.append(f"  D37:  {d37_r_mean:.1f} ± {d37_r_ci:.1f} reward, "
                         f"{d37.avg_laps_completed:.1f} laps, "
                         f"{d37.avg_speed_ms:.2f} m/s, "
                         f"{d37.avg_pit_count:.1f} pits")
            lines.append(f"  cv2:  {cv2_r_mean:.1f} ± {cv2_r_ci:.1f} reward, "
                         f"{cv2.avg_laps_completed:.1f} laps, "
                         f"{cv2.avg_speed_ms:.2f} m/s, "
                         f"{cv2.avg_pit_count:.1f} pits")
            lines.append(f"  Gap:  {d37_r_mean - cv2_r_mean:+.1f} reward "
                         f"({(d37_r_mean/cv2_r_mean)*100:.1f}% of cv2)")
            lines.append(f"  Note: D37's 3 pits = -600 overhead. "
                         f"Adjusted gap ≈ {d37_r_mean + 600 - cv2_r_mean:+.1f}")

        if exp_key and d37_key:
            exp = fs[exp_key]
            d37 = fs[d37_key]
            d37_r_mean, _ = mean_ci95([r.total_reward for r in d37.all_results])
            exp_r_mean, _ = mean_ci95([r.total_reward for r in exp.all_results])
            lines.append(f"  vs Expert: D37 {d37_r_mean - exp_r_mean:+.1f} reward "
                         f"({d37.avg_laps_completed:.1f} vs {exp.avg_laps_completed:.1f} laps)")

    lines.append("")
    output_path.write_text("\n".join(lines))
    print(f"\n[Benchmark] Report saved → {output_path}")


if __name__ == "__main__":
    N = 100
    print(f"\n[Benchmark] N={N} final benchmark — 4 key policies")
    print(f"[Benchmark] Estimated runtime: ~25 minutes on CPU")
    print(f"[Benchmark] Started: {datetime.now().strftime('%H:%M:%S')}\n")

    results = run_benchmark(n_episodes=N)

    # Save text report
    report_path = project_root / "Notes" / "benchmark_n100.txt"
    write_report(results, N, report_path)

    # Save plots
    if "RANDOM START" in results:
        plot_bar_comparison(
            results["RANDOM START"],
            save_path=str(project_root / "plots" / "benchmark_n100_random.png"),
            subtitle=f"Random Start | N={N} Episodes per Policy",
        )
        plot_episode_distributions(
            results["RANDOM START"],
            save_path=str(project_root / "plots" / "benchmark_n100_dist_random.png"),
        )
    if "FIXED START" in results:
        plot_bar_comparison(
            results["FIXED START"],
            save_path=str(project_root / "plots" / "benchmark_n100_fixed.png"),
            subtitle=f"Fixed Start | N={N} Episodes per Policy",
        )
        plot_episode_distributions(
            results["FIXED START"],
            save_path=str(project_root / "plots" / "benchmark_n100_dist_fixed.png"),
        )

    print(f"\n[Benchmark] Done: {datetime.now().strftime('%H:%M:%S')}")
    print(f"[Benchmark] Report: Notes/benchmark_n100.txt")
    print(f"[Benchmark] Plots:  plots/benchmark_n100_*.png")
