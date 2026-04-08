"""
EmailTriageEnv — OpenEnv-compliant environment.
Real-world task: classify and prioritize a queue of emails.
"""

import uuid
from typing import Tuple, Dict, Any
from models import Action, Observation, State
from tasks import TASKS

VALID_LABELS = {"urgent", "important", "normal", "spam"}
LABEL_SCORES = {
    # (predicted, ground_truth): reward
    ("urgent",    "urgent"):    1.0,
    ("important", "important"): 1.0,
    ("normal",    "normal"):    1.0,
    ("spam",      "spam"):      1.0,
    ("urgent",    "important"): 0.5,
    ("important", "urgent"):    0.5,
    ("normal",    "important"): 0.2,
    ("important", "normal"):    0.2,
    ("urgent",    "normal"):    0.0,
    ("spam",      "urgent"):    0.0,
    ("spam",      "important"): 0.0,
    ("urgent",    "spam"):      0.0,
    ("important", "spam"):      0.0,
    ("normal",    "spam"):      0.1,
    ("spam",      "normal"):    0.3,
}


def _label_reward(predicted: str, truth: str) -> float:
    return LABEL_SCORES.get((predicted, truth), 0.0)


def _priority_reward(predicted: int, truth: int) -> float:
    diff = abs(predicted - truth)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.6
    elif diff == 2:
        return 0.3
    else:
        return 0.0


def _reason_reward(reason: str) -> float:
    """Reward non-empty, sufficiently descriptive reasons."""
    if not reason or len(reason.strip()) < 5:
        return 0.0
    if len(reason.strip()) < 15:
        return 0.3
    return 0.6


class EmailTriageEnv:
    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"task_id must be one of {list(TASKS.keys())}")
        self.task_id = task_id
        self._task = TASKS[task_id]
        self._state: State = None
        self.reset()

    # ------------------------------------------------------------------ #
    #  OpenEnv Interface                                                   #
    # ------------------------------------------------------------------ #

    def reset(self) -> Observation:
        self._state = State(
            task_id=self.task_id,
            difficulty=self._task["difficulty"],
            emails=[dict(e) for e in self._task["emails"]],
            current_index=0,
            actions_taken=[],
            total_score=0.0,
            done=False,
        )
        return self._make_observation("Environment reset. Start triaging emails.")

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._state.done:
            obs = self._make_observation("Episode already complete.")
            return obs, 0.0, True, {"error": "already done"}

        current_email = self._state.emails[self._state.current_index]

        # Validate email_id
        if action.email_id != current_email["email_id"]:
            obs = self._make_observation(
                f"Wrong email_id. Expected {current_email['email_id']}, got {action.email_id}."
            )
            return obs, -0.1, False, {"error": "wrong email_id"}

        # Validate label
        if action.label not in VALID_LABELS:
            obs = self._make_observation(f"Invalid label '{action.label}'. Use: {VALID_LABELS}")
            return obs, -0.1, False, {"error": "invalid label"}

        # Validate priority
        if not (1 <= action.priority <= 5):
            obs = self._make_observation("Priority must be between 1 and 5.")
            return obs, -0.1, False, {"error": "invalid priority"}

        # Compute reward
        truth_label    = current_email["ground_truth_label"]
        truth_priority = current_email["ground_truth_priority"]

        r_label    = _label_reward(action.label, truth_label)
        r_priority = _priority_reward(action.priority, truth_priority)
        r_reason   = _reason_reward(action.reason)

        # Weighted reward: label matters most
        step_reward = round(0.6 * r_label + 0.3 * r_priority + 0.1 * r_reason, 4)

        # Update state
        self._state.actions_taken.append({
            "email_id":          action.email_id,
            "predicted_label":   action.label,
            "predicted_priority": action.priority,
            "reason":            action.reason,
            "truth_label":       truth_label,
            "truth_priority":    truth_priority,
            "step_reward":       step_reward,
        })
        self._state.total_score += step_reward
        self._state.current_index += 1

        total_emails = len(self._state.emails)
        done = self._state.current_index >= total_emails
        self._state.done = done

        msg = (
            f"Email '{action.email_id}' triaged. "
            f"Step reward: {step_reward:.2f} "
            f"(label: {r_label:.1f}, priority: {r_priority:.1f}, reason: {r_reason:.1f}). "
            f"Truth was label='{truth_label}', priority={truth_priority}."
        )
        if done:
            final = self.final_score()
            msg += f" | DONE. Final score: {final:.4f}"

        obs = self._make_observation(msg)
        return obs, step_reward, done, {
            "r_label": r_label,
            "r_priority": r_priority,
            "r_reason": r_reason,
        }

    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _make_observation(self, message: str) -> Observation:
        emails     = self._state.emails
        idx        = self._state.current_index
        total      = len(emails)
        remaining  = max(0, total - idx)

        if idx < total:
            email = emails[idx]
        else:
            email = emails[-1]  # last email for final obs

        return Observation(
            email_id=email["email_id"],
            subject=email["subject"],
            sender=email["sender"],
            body=email["body"],
            timestamp=email["timestamp"],
            remaining_emails=remaining,
            current_score=round(self._state.total_score, 4),
            message=message,
        )

    def final_score(self) -> float:
        """Normalized score in [0, 1]."""
        n = len(self._state.emails)
        if n == 0:
            return 0.0
        max_possible = n * 1.0  # max step reward per email = 1.0
        return round(self._state.total_score / max_possible, 4)
