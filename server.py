"""
FastAPI server for EmailTriageEnv
OpenEnv compliant (POST + JSON body supported)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import uvicorn

from models import Action
from env import EmailTriageEnv

# ===== APP SETUP =====
app = FastAPI(
    title="EmailTriageEnv",
    description="OpenEnv-compliant RL environment for email triage.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENV STORAGE =====
_envs: dict[str, EmailTriageEnv] = {}


def get_env(task_id: str) -> EmailTriageEnv:
    if task_id not in _envs:
        try:
            _envs[task_id] = EmailTriageEnv(task_id=task_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return _envs[task_id]


# ===== REQUEST MODELS (IMPORTANT FIX) =====
class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Action


# ===== ROOT =====
@app.get("/")
def root():
    return RedirectResponse(url="/health")


# ===== HEALTH =====
@app.get("/health")
def health():
    return {"status": "ok", "env": "EmailTriageEnv"}


# ===== RESET (POST REQUIRED FOR OPENENV) =====
@app.post("/reset")
def reset_post(req: ResetRequest):
    env = get_env(req.task_id)
    obs = env.reset()
    return obs.model_dump()


# ===== OPTIONAL GET RESET (FOR DEBUG) =====
@app.get("/reset")
def reset_get(task_id: str = Query(default="easy")):
    env = get_env(task_id)
    obs = env.reset()
    return obs.model_dump()


# ===== STEP (POST REQUIRED FOR OPENENV) =====
@app.post("/step")
def step_post(req: StepRequest):
    env = get_env(req.task_id)
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


# ===== OPTIONAL GET STEP (FOR DEBUG) =====
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


# ===== STATE =====
@app.get("/state")
def state(task_id: str = Query(default="easy")):
    env = get_env(task_id)
    return env.state().model_dump()


# ===== MAIN =====
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)