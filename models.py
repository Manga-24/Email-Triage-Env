from pydantic import BaseModel
from typing import Optional, List


class Action(BaseModel):
    """Agent action: classify and prioritize one email."""
    email_id: str
    label: str          # "urgent", "important", "normal", "spam"
    priority: int       # 1 (highest) to 5 (lowest)
    reason: str         # short justification


class Observation(BaseModel):
    """What the agent sees at each step."""
    email_id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    remaining_emails: int
    current_score: float
    message: str


class State(BaseModel):
    """Full internal state of the environment."""
    task_id: str
    difficulty: str
    emails: List[dict]
    current_index: int
    actions_taken: List[dict]
    total_score: float
    done: bool
