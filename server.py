from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

from models import Action
from env import EmailTriageEnv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs = {}


def get_env(task_id: str):
    if task_id not in _envs:
        _envs[task_id] = EmailTriageEnv(task_id=task_id)
    return _envs[task_id]


# ===== REQUEST MODELS =====
class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Action


# ===== ROOT =====
@app.get("/")
def root():
    return RedirectResponse(url="/health")


@app.get("/health")
def health():
    return {"status": "ok"}


# ===== RESET (CRITICAL FIX) =====
@app.post("/reset")
def reset(req: ResetRequest | None = Body(default=None)):
    task_id = "easy"
    if req and req.task_id:
        task_id = req.task_id

    env = get_env(task_id)
    obs = env.reset()
    return obs.model_dump()


# ===== STEP (CRITICAL FIX) =====
@app.post("/step")
def step(req: StepRequest | None = Body(default=None)):
    if req is None:
        raise HTTPException(status_code=400, detail="Missing body")

    env = get_env(req.task_id)
    obs, reward, done, info = env.step(req.action)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


# ===== OPTIONAL GET (for debugging only) =====
@app.get("/reset")
def reset_get(task_id: str = Query(default="easy")):
    env = get_env(task_id)
    obs = env.reset()
    return obs.model_dump()


@app.get("/step")
def step_get(
    task_id: str = Query(default="easy"),
    email_id: str = Query(default="e1"),
    label: str = Query(default="normal"),
    priority: int = Query(default=1),
    reason: str = Query(default="auto"),
):
    env = get_env(task_id)

    action = Action(
        email_id=email_id,
        label=label,
        priority=priority,
        reason=reason,
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = Query(default="easy")):
    env = get_env(task_id)
    return env.state().model_dump()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860)