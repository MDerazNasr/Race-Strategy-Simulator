"""
D29 Diagnostic: Fixed-Start Ceiling Investigation for D28.

WHY THIS SCRIPT?
================
D26, D27, D28 all give exactly 1883.41 on the fixed-start evaluation.
The pit_timing_reward offers +100 for pitting at tyre_life < 0.30.
D28's pit bias has moved to -0.929 (learning to delay pitting), but
the fixed-start result is still locked at 1883.41.

KEY QUESTION: Is the +100 timing bonus geometrically reachable from the
fixed-start trajectory?

To answer this, we need to trace EXACTLY what happens:
  1. At what step / tyre_life does the pit fire?
  2. Does tyre_life ever reach < 0.30 before the pit fires?
  3. What is tyre_life at episode termination (step 1354)?
  4. How much tyre_life remains after the pit? When does post-pit
     tyre_life start to matter again?
  5. Could a LATER pit (at tyre_life < 0.30) give more reward?

If tyre_life is already below 0.30 when the pit fires → bonus already earned.
If tyre_life > 0.30 when pit fires → agent pitting too early (timing penalty zone).
If tyre_life < 0.30 is never reached before pit → the threshold is reachable
  if the agent delays (good news for more training).

WHAT THIS SCRIPT DOES:
  Run one deterministic fixed-start episode with d28.
  Log every step: tyre_life, pit_signal mean, pit fires, reward.
  Print a summary: pit timing, tyre life at termination.
  Run a what-if counterfactual: simulate delayed pit at tyre_life=0.27.
"""

import sys
from pathlib import Path

import numpy as np
import torch

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from env.f1_env import F1Env


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH    = str(project_root / "rl" / "ppo_pit_v4_d28.zip")
FIXED_OPTIONS = {"fixed_start": True}   # same as evaluate.py — places car at track waypoint 0
TYRE_TIMING_THRESHOLD = 0.30    # pit_timing_reward bonus threshold
PRINT_EVERY   = 50              # log a row every N steps (also log pit events)

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_episode_trace(model, env):
    """Run one deterministic episode and return a full trace."""
    obs, _ = env.reset(options=FIXED_OPTIONS)
    trace = []   # list of dicts, one per step

    done = False
    pit_fired_steps = []

    while not done:
        step_num = env.step_count    # BEFORE stepping
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        tyre_life = info.get("tyre_life", 1.0)
        pit_count = info.get("pit_count", 0)

        entry = {
            "step":       env.step_count,
            "tyre_life":  tyre_life,
            "pit_signal": float(action[2]),
            "reward":     reward,
            "pit_count":  pit_count,
            "terminated": terminated,
            "truncated":  truncated,
        }
        trace.append(entry)

        # Detect pit firing (count changed)
        if len(trace) > 1 and trace[-1]["pit_count"] > trace[-2]["pit_count"]:
            pit_fired_steps.append(env.step_count)

    return trace, pit_fired_steps


def run_counterfactual(model, forced_pit_tyre_life):
    """
    Run episode with a forced_pit_threshold overriding agent pit signal.
    This simulates: what if the pit fired ONLY when tyre_life < forced_pit_tyre_life?
    The agent's OWN pit signal is ignored; pit fires state-conditionally instead.
    """
    env = F1Env(
        tyre_degradation=True,
        pit_stops=True,
        forced_pit_threshold=forced_pit_tyre_life,
    )
    obs, _ = env.reset(options=FIXED_OPTIONS)
    total_reward = 0.0
    done = False
    pit_steps = []
    prev_pit_count = 0

    while not done:
        # Use agent obs but suppress pit signal (forced pit will handle it)
        action, _ = model.predict(obs, deterministic=True)
        # Override pit signal to never-pit (forced threshold does the work)
        action_no_pit = np.array([action[0], action[1], -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_no_pit)
        total_reward += reward
        done = terminated or truncated
        if info["pit_count"] > prev_pit_count:
            pit_steps.append((env.step_count, info["tyre_life"]))
            prev_pit_count = info["pit_count"]

    env.close()
    return total_reward, env.step_count, env.laps_completed, pit_steps


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Diag] Device: {device}")
    print(f"[Diag] Loading: {MODEL_PATH}")

    model = PPO.load(MODEL_PATH, device=device)

    # ── Pit row diagnostics ────────────────────────────────────────────────────
    pit_bias = model.policy.action_net.bias[2].item()
    pit_std  = model.policy.log_std[2].item()
    print(f"\n[Model] action_net.bias[2] = {pit_bias:.6f}  (d28 pit bias)")
    print(f"[Model] log_std[2]         = {pit_std:.6f}   (pit std)")
    # The pit fires when pit_signal > 0.
    # With a Gaussian mean (action_net output) and std=exp(log_std),
    # the DETERMINISTIC action equals the mean. Pit fires when mean > 0.
    # Bias alone sets the pre-activation baseline; the full mean depends on obs.
    print(f"[Model] Pit fires when mean > 0. Bias of -0.929 → strong tendency NOT to pit.\n")

    # ── Build eval env ────────────────────────────────────────────────────────
    env = F1Env(tyre_degradation=True, pit_stops=True)   # standard, no pit_timing_reward
    obs, _ = env.reset(options=FIXED_OPTIONS)

    # ── Trace episode ─────────────────────────────────────────────────────────
    print("=" * 72)
    print("FIXED-START EPISODE TRACE (D28, deterministic)")
    print("=" * 72)
    print(f"{'Step':>6}  {'TyreLife':>8}  {'PitSig':>8}  {'PitCnt':>6}  {'Reward':>8}  Note")
    print("-" * 72)

    obs, _ = env.reset(options=FIXED_OPTIONS)
    trace = []
    done = False
    total_reward = 0.0
    prev_pit_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        tyre_life = info["tyre_life"]
        pit_count = info["pit_count"]
        pit_signal = float(action[2])
        step = env.step_count

        note = ""
        if pit_count > prev_pit_count:
            note = f"<<< PIT FIRED  tyre_life={tyre_life:.4f}"
            prev_pit_count = pit_count
        elif tyre_life < TYRE_TIMING_THRESHOLD and pit_count == 0:
            note = f"*** tyre_life < {TYRE_TIMING_THRESHOLD} (timing bonus zone)"
        elif terminated:
            note = "TERMINATED (crash)"
        elif truncated:
            note = "TRUNCATED (2000 steps)"

        trace.append({
            "step":       step,
            "tyre_life":  tyre_life,
            "pit_signal": pit_signal,
            "pit_count":  pit_count,
            "reward":     reward,
            "note":       note,
        })

        if step % PRINT_EVERY == 0 or note:
            print(f"{step:>6}  {tyre_life:>8.4f}  {pit_signal:>8.4f}  {pit_count:>6}  {reward:>8.2f}  {note}")

    print("-" * 72)
    env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    final   = trace[-1]
    pit_rows = [t for t in trace if t["pit_count"] > (trace[trace.index(t)-1]["pit_count"] if trace.index(t) > 0 else 0)]

    # Find actual pit events (where pit_count increases)
    pit_events = []
    for i, t in enumerate(trace):
        if i > 0 and t["pit_count"] > trace[i-1]["pit_count"]:
            pit_events.append(t)

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  Total steps:      {final['step']}")
    print(f"  Total reward:     {total_reward:.2f}")
    print(f"  Final tyre_life:  {final['tyre_life']:.4f}")
    print(f"  Pit count:        {final['pit_count']}")

    if pit_events:
        for i, pe in enumerate(pit_events):
            print(f"\n  Pit #{i+1}:")
            print(f"    Step:        {pe['step']}")
            print(f"    Tyre life at pit: {pe['tyre_life']:.4f}")
            print(f"    Pit signal:  {pe['pit_signal']:.4f}")
            # pre-pit tyre life (step before)
            pre_idx = next((i for i, t in enumerate(trace) if t["step"] == pe["step"] - 1), None)
            pre = trace[pre_idx] if pre_idx is not None else pe
            print(f"    Pre-pit tyre_life: {pre['tyre_life']:.4f}")
    else:
        print("\n  NO PITS FIRED.")

    # Check if tyre_life ever dropped below 0.30 BEFORE first pit
    below_30_before_pit = [t for t in trace
                           if t["tyre_life"] < 0.30 and t["pit_count"] == 0]
    if below_30_before_pit:
        first = below_30_before_pit[0]
        print(f"\n  tyre_life crossed 0.30 at step {first['step']} "
              f"(tyre_life={first['tyre_life']:.4f}) BEFORE pit fired!")
        print(f"  → Timing bonus WAS reachable if pit fired here or later.")
    else:
        first_pit_step = pit_events[0]["step"] if pit_events else final["step"]
        pre_pit_idx  = next((i for i, t in enumerate(trace) if t["step"] == first_pit_step - 1), None)
        pre_pit_tyre = trace[pre_pit_idx]["tyre_life"] if pre_pit_idx is not None else None
        print(f"\n  tyre_life did NOT cross 0.30 before first pit.")
        if pre_pit_tyre is not None:
            print(f"  Pit fired at tyre_life ≈ {pre_pit_tyre:.4f} "
                  f"(still above 0.30 → timing bonus NOT yet earned)")
        print(f"  → More training could push bias negative enough to delay past 0.30.")

    # Post-pit tyre life check
    if pit_events:
        post_pit = [t for t in trace if t["step"] > pit_events[0]["step"]]
        if post_pit:
            post_pit_final = post_pit[-1]
            post_pit_min   = min(t["tyre_life"] for t in post_pit)
            print(f"\n  Post-pit tyre_life at episode end: {post_pit_final['tyre_life']:.4f}")
            print(f"  Post-pit tyre_life minimum:        {post_pit_min:.4f}")
            if post_pit_final["tyre_life"] < 0.40:
                print(f"  → Post-pit tyres also worn by episode end. A 2nd pit might help.")
            else:
                print(f"  → Post-pit tyres still healthy at episode end.")

    # Tyre wear rate estimation (first 200 steps, before pit)
    pre_pit_trace = [t for t in trace if t["pit_count"] == 0 and t["step"] <= 200]
    if len(pre_pit_trace) >= 50:
        tyre_at_50  = pre_pit_trace[49]["tyre_life"]   # step 50
        tyre_at_200 = pre_pit_trace[min(199, len(pre_pit_trace)-1)]["tyre_life"]
        wear_rate   = (tyre_at_50 - tyre_at_200) / max(1, 200 - 50)
        print(f"\n  Tyre wear rate (steps 50→200): {wear_rate*1000:.3f} per 1000 steps")
        if wear_rate > 0:
            steps_to_030 = (tyre_at_50 - 0.30) / wear_rate
            print(f"  Estimated step to reach tyre_life=0.30: {50 + steps_to_030:.0f}")
            steps_to_027 = (tyre_at_50 - 0.27) / wear_rate
            print(f"  Estimated step to reach tyre_life=0.27: {50 + steps_to_027:.0f}")

    # ── Counterfactual: what if pit is delayed to tyre_life < X? ──────────────
    print(f"\n{'='*72}")
    print("COUNTERFACTUAL: Forced pit at different tyre_life thresholds")
    print(f"{'='*72}")
    print(f"  Simulating: agent drives normally, pit fires ONLY when tyre_life < threshold")
    print(f"  (overrides agent pit signal — measures impact of delayed pit timing)")
    print()
    print(f"  {'Threshold':>10}  {'Reward':>10}  {'Steps':>8}  {'Laps':>6}  {'Pit at':>10}")

    for thresh in [0.50, 0.40, 0.35, 0.30, 0.27, 0.25, 0.20]:
        try:
            rew, steps, laps, pit_info = run_counterfactual(model, thresh)
            pit_str = f"step {pit_info[0][0]}, tl={pit_info[0][1]:.3f}" if pit_info else "no pit"
            marker = " ←" if abs(thresh - TYRE_TIMING_THRESHOLD) < 0.01 else ""
            print(f"  {thresh:>10.2f}  {rew:>10.2f}  {steps:>8}  {laps:>6.1f}  {pit_str}{marker}")
        except Exception as e:
            print(f"  {thresh:>10.2f}  ERROR: {e}")

    print(f"\n  Note: threshold=0.30 boundary = timing bonus zone (net -100 vs -200)")
    print(f"  If reward increases at threshold<0.30 vs 0.35, delayed pit is beneficial.")


if __name__ == "__main__":
    main()
