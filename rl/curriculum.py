"""
Curriculum Learning for DynamicCar PPO Training.

WHAT IS CURRICULUM LEARNING?
==============================
Curriculum learning = "start easy, get harder over time."

The name comes from education: you don't teach calculus before algebra.
In RL, presenting hard tasks too early leads to the cold-start problem
all over again -- but now at a deeper level.

OUR SPECIFIC PROBLEM:
  DynamicCar has Pacejka tyre physics. At high speeds, lateral forces
  saturate and the car spins. A policy that tries to go 20 m/s from
  step 1 will spin constantly and learn nothing useful from the crashes.

  The reward signal is too sparse: crashing gives -20 but the agent
  does not know WHY (too fast? wrong steering? bad yaw entry?).
  With random starts, the agent tries to attribute blame across many
  possible causes -- it has no signal isolating speed as the problem.

THE CURRICULUM SOLUTION:
  Stage 1 -- Stability:  cap speed at ~8 m/s. At 8 m/s, tyre slip angles
    stay small -- the car behaves almost like the kinematic model. The agent
    learns to follow the track and observe the new obs dims (v_y, r)
    without the chaos of high-speed crashes.

  Stage 2 -- Speed:  unlock to ~15 m/s. Tyres now start working -- moderate
    slip angles generate real cornering force. The agent must learn to
    manage v_y and r rather than just observe them.

  Stage 3 -- Racing:  full speed, no cap. The agent must use everything
    it learned in Stages 1 and 2 to optimise lap time.

WHY THIS WORKS (theory):
  Each stage provides a DENSE reward signal. At 8 m/s the agent can
  complete full laps and collect large positive rewards. Those completed
  laps train the value function to understand track structure -- "being
  at waypoint 50 with v=8 m/s is worth ~X reward." The value function
  carries forward when speed increases, so the agent already "knows"
  what a good lap looks like. It only has to learn the new physics.

  Analogy: a racing driving school starts students on go-karts (low speed,
  forgiving). Once the student understands racing lines and braking points,
  they graduate to faster cars. The track knowledge transfers; only the
  car behaviour is new.

  Interview version:
    "Curriculum learning addresses the credit assignment problem at scale.
     By constraining the task initially, the reward signal stays dense
     and informative. As the agent builds a stable value function
     baseline, we expand the state-action space gradually, letting the
     agent transfer prior knowledge rather than restart from scratch."

GRADUATION CRITERION:
  The callback monitors lap_completion_rate = fraction of episodes that
  survived to max_steps (did NOT crash) over a rolling window of rollouts.
  When the rolling average exceeds a threshold, the agent advances to the
  next stage. The threshold is set lower for harder stages.
"""

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# =============================================================================
# CURRICULUM STAGE DEFINITION
# =============================================================================

@dataclass
class CurriculumStage:
    """
    One stage of the curriculum. Defines the physical speed cap AND
    the reward weights active during that stage.

    When the agent graduates a stage, the next stage's parameters are
    applied to the LIVE environment -- no restarting, no new model.
    The policy and value function weights carry forward unchanged.

    Fields:
        name:          Human-readable label (logged to TensorBoard)
        max_accel:     DynamicCar.max_accel (m/s^2) -- limits top speed.
                       At max_accel=6  m/s^2, terminal speed ~  8 m/s
                       At max_accel=11 m/s^2, terminal speed ~ 15 m/s
                       At max_accel=15 m/s^2, original DynamicCar limit
        reward_kwargs: Keyword arguments forwarded to RacingReward().
                       Controls what behaviour is incentivised each stage.
        grad_lap_rate: Minimum rolling lap completion rate to graduate.
        grad_window:   Number of consecutive rollouts to average over.
                       Larger window = more stable graduation criterion.
    """
    name:          str
    max_accel:     float
    reward_kwargs: Dict   = field(default_factory=dict)
    grad_lap_rate: float  = 0.4
    grad_window:   int    = 5


# =============================================================================
# STAGE DEFINITIONS
# =============================================================================

STAGES: List[CurriculumStage] = [

    # -- Stage 1: Stability ---------------------------------------------------
    # Goal: survive. Learn the track. Learn to read v_y and r as warning signs.
    # Physics: max_accel=6 caps speed at ~8 m/s. Tyre slip stays in linear
    # regime, behaves almost like the kinematic model. No tyre saturation.
    # Reward: lateral penalty doubled, speed bonus removed.
    #   -> agent is punished for drift but NOT rewarded for rushing.
    #   -> optimal behaviour: stay centred, point at the track, survive.
    CurriculumStage(
        name          = "Stage 1 -- Stability (<=8 m/s)",
        max_accel     = 6.0,
        reward_kwargs = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.0,   # no speed bonus -- don't reward rushing
            lateral_weight    = 1.0,   # 2x normal -- survival is the priority
            heading_weight    = 0.2,   # 2x normal -- point the right way
            smoothness_weight = 0.1,   # 2x normal -- learn smooth inputs early
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.5,   # graduate when 50% of episodes survive
        grad_window   = 5,
    ),

    # -- Stage 2: Speed -------------------------------------------------------
    # Goal: go faster without crashing. Learn that v_y is controllable.
    # Physics: max_accel=11 unlocks ~15 m/s. Tyres now work in the nonlinear
    # regime -- moderate slip gives maximum cornering force, but pushing too
    # hard saturates them. The agent must learn to feel the limit.
    # Reward: speed bonus re-enabled, lateral penalty back to normal.
    #   -> agent is now rewarded for speed, but still penalised for drift.
    CurriculumStage(
        name          = "Stage 2 -- Speed (<=15 m/s)",
        max_accel     = 11.0,
        reward_kwargs = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,   # reward going faster
            lateral_weight    = 0.5,   # back to normal lateral penalty
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 0.3,   # at higher speed, some crashing is acceptable
        grad_window   = 5,
    ),

    # -- Stage 3: Racing ------------------------------------------------------
    # Goal: maximise lap time. No speed cap. Same hyperparameters as the
    # stable training run from d11/d12.
    # This stage never graduates -- it runs until total_timesteps is reached.
    CurriculumStage(
        name          = "Stage 3 -- Full Racing (no cap)",
        max_accel     = 15.0,
        reward_kwargs = dict(
            progress_weight   = 1.0,
            speed_weight      = 0.1,
            lateral_weight    = 0.5,
            heading_weight    = 0.1,
            smoothness_weight = 0.05,
            terminal_penalty  = 20.0,
        ),
        grad_lap_rate = 1.1,   # impossible threshold -- never graduates
        grad_window   = 9999,
    ),
]


# =============================================================================
# CURRICULUM CALLBACK
# =============================================================================

class CurriculumCallback(BaseCallback):
    """
    SB3 callback that drives the curriculum stage progression.

    LIFECYCLE:
      _on_training_start():  applies Stage 1 to the live environment
      _on_step():            tracks crashes vs completions per episode
      _on_rollout_end():     computes rolling lap rate, checks graduation,
                             logs to TensorBoard

    HOW IT MODIFIES THE LIVE ENVIRONMENT:
      SB3's DummyVecEnv wraps the env in two layers:
        DummyVecEnv -> Monitor -> F1Env

      To reach the actual F1Env we call .unwrapped on the Monitor:
        inner_env = self.training_env.envs[0].unwrapped

      From there we directly assign:
        inner_env.car.max_accel  = new_accel
        inner_env.reward_fn      = RacingReward(**new_kwargs)

      This modifies the running environment IN PLACE. No reset, no new
      model. PPO sees different physics and reward from the next rollout.

      Interview version:
        "We modify environment parameters via the callback interface --
         a non-destructive live update that preserves the trained
         value function and policy weights."

    CRASH DETECTION:
      F1Env.step() adds "crashed": bool(terminated) to the info dict.
      When an episode ends, Monitor adds "episode" to the same dict.
      We check both: "episode" signals an episode just ended;
      info["crashed"] tells us HOW it ended.

        crashed=True  -- terminated (went off-track) -- bad
        crashed=False -- truncated  (survived 2000 steps) -- good
    """

    def __init__(self, stages: List[CurriculumStage] = STAGES, verbose: int = 1):
        super().__init__(verbose)
        self.stages    = stages
        self.stage_idx = 0   # start at Stage 1 (index 0)

        # Rolling window of per-rollout lap completion rates
        self._lap_window: List[float] = []

        # Per-rollout episode counters (reset each _on_rollout_end)
        self._ep_completed = 0   # episodes that survived to max_steps
        self._ep_total     = 0   # total episodes that finished this rollout

    # -- Convenience property -------------------------------------------------
    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_idx]

    # -- Apply a stage to the live environment --------------------------------
    def _apply_stage(self, stage: CurriculumStage) -> None:
        """
        Push new physics + reward weights into the running F1Env.

        Navigation: DummyVecEnv -> envs[0] (Monitor) -> .unwrapped (F1Env)
        """
        from rl.rewards import RacingReward

        # Unwrap Monitor to get the bare F1Env
        inner_env = self.training_env.envs[0].unwrapped

        # Update the DynamicCar engine limit (controls terminal speed)
        inner_env.car.max_accel = stage.max_accel

        # Swap the reward function for this stage's weights
        inner_env.reward_fn = RacingReward(**stage.reward_kwargs)

        if self.verbose >= 1:
            print(f"\n[Curriculum] Applying: {stage.name}")
            print(f"             max_accel = {stage.max_accel} m/s^2")

    # -- Called once at training start ----------------------------------------
    def _on_training_start(self) -> None:
        """Activate Stage 1 immediately before the first rollout."""
        self._apply_stage(self.current_stage)

    # -- Called at every environment step -------------------------------------
    def _on_step(self) -> bool:
        """
        Track how each episode ends.

        self.locals["infos"] is a list of one dict per parallel env.
        "episode" appears in info when that episode just finished (added by Monitor).
        "crashed"  appears every step (added by F1Env.step()).

        We accumulate _ep_total and _ep_completed for the rollout-level
        calculation done in _on_rollout_end().
        """
        for info in self.locals["infos"]:
            if "episode" in info:
                # Episode just ended for this env
                self._ep_total += 1
                if not info.get("crashed", True):
                    # crashed=False = survived to max_steps = lap completed
                    self._ep_completed += 1

        return True   # returning False would abort training

    # -- Called at the end of each rollout ------------------------------------
    def _on_rollout_end(self) -> None:
        """
        Compute rolling lap rate, log to TensorBoard, check graduation.

        A "rollout" is n_steps steps (2048 by default). Multiple episodes
        may finish within one rollout, so _ep_total can be > 1.
        """
        stage = self.current_stage

        # Per-rollout lap completion rate
        rate = (self._ep_completed / self._ep_total) if self._ep_total > 0 else 0.0

        # Update rolling window -- keep only the last grad_window entries
        self._lap_window.append(rate)
        if len(self._lap_window) > stage.grad_window:
            self._lap_window.pop(0)

        rolling_rate = float(np.mean(self._lap_window))

        # Log to TensorBoard -- shows up under curriculum/ panel
        self.logger.record("curriculum/stage",        float(self.stage_idx))
        self.logger.record("curriculum/lap_rate",     rate)
        self.logger.record("curriculum/rolling_rate", rolling_rate)
        self.logger.record("curriculum/max_accel",    stage.max_accel)
        self.logger.record("curriculum/grad_target",  stage.grad_lap_rate)

        # Reset counters for the next rollout
        self._ep_completed = 0
        self._ep_total     = 0

        # -- Graduation check -------------------------------------------------
        # Conditions:
        #   1. Window is full (avoid graduating on a single lucky rollout)
        #   2. Rolling rate meets the stage's threshold
        #   3. There is a next stage to advance to
        next_idx    = self.stage_idx + 1
        window_full = len(self._lap_window) >= stage.grad_window

        if window_full and rolling_rate >= stage.grad_lap_rate and next_idx < len(self.stages):

            if self.verbose >= 1:
                print(f"\n[Curriculum] *** GRADUATED from {stage.name}! ***")
                print(f"             Rolling lap rate: {rolling_rate:.2f} "
                      f">= threshold {stage.grad_lap_rate:.2f}")
                print(f"             Advancing to: {self.stages[next_idx].name}\n")

            self.stage_idx   = next_idx
            self._lap_window = []   # reset window for fresh measurement of new stage
            self._apply_stage(self.stages[self.stage_idx])
