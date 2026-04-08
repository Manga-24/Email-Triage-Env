"""
FastAPI server for EmailTriageEnv.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from models import Action, Observation, State
from env import EmailTriageEnv

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

# One environment instance per task (keyed by task_id)
_envs: dict[str, EmailTriageEnv] = {}


def get_env(task_id: str) -> EmailTriageEnv:
    if task_id not in _envs:
        try:
            _envs[task_id] = EmailTriageEnv(task_id=task_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return _envs[task_id]


from fastapi.responses import HTMLResponse, RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/web")

@app.get("/health")
def health():
    return {"status": "ok", "env": "EmailTriageEnv"}


@app.post("/reset")
def reset(task_id: str = Query(default="easy", description="Task difficulty: easy | medium | hard")):
    env = get_env(task_id)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(
    action: Action,
    task_id: str = Query(default="easy"),
):
    env = get_env(task_id)
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


@app.get("/web", response_class=HTMLResponse)
def web_ui():
    """Simple browser UI for manual testing."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>EmailTriageEnv — Test UI</title>
  <style>
    body { font-family: monospace; background: #0f0f0f; color: #e0e0e0; padding: 24px; }
    h1 { color: #7ee8a2; }
    label { display: block; margin-top: 12px; color: #aaa; }
    input, select, textarea { width: 100%; padding: 6px; background: #1e1e1e; color: #fff; border: 1px solid #333; border-radius: 4px; }
    button { margin-top: 12px; padding: 8px 18px; background: #7ee8a2; color: #000; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
    pre { background: #1a1a1a; padding: 16px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; color: #cfe2ff; margin-top: 16px; }
    .section { border: 1px solid #333; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
  </style>
</head>
<body>
  <h1>📧 EmailTriageEnv — Test UI</h1>

  <div class="section">
    <h2>1. Reset</h2>
    <label>Task ID:
      <select id="task_id_reset">
        <option value="easy">easy</option>
        <option value="medium">medium</option>
        <option value="hard">hard</option>
      </select>
    </label>
    <button onclick="doReset()">Reset</button>
    <pre id="reset_out">—</pre>
  </div>

  <div class="section">
    <h2>2. Step</h2>
    <label>Task ID: <select id="task_id_step">
      <option value="easy">easy</option>
      <option value="medium">medium</option>
      <option value="hard">hard</option>
    </select></label>
    <label>Email ID: <input id="email_id" value="e1"></label>
    <label>Label: <select id="label">
      <option>urgent</option><option>important</option><option>normal</option><option>spam</option>
    </select></label>
    <label>Priority (1-5): <input id="priority" type="number" min="1" max="5" value="1"></label>
    <label>Reason: <textarea id="reason" rows="2">This email indicates a critical production issue.</textarea></label>
    <button onclick="doStep()">Step</button>
    <pre id="step_out">—</pre>
  </div>

  <div class="section">
    <h2>3. State</h2>
    <label>Task ID: <select id="task_id_state">
      <option value="easy">easy</option>
      <option value="medium">medium</option>
      <option value="hard">hard</option>
    </select></label>
    <button onclick="doState()">Get State</button>
    <pre id="state_out">—</pre>
  </div>

<script>
  async function doReset() {
    const task = document.getElementById("task_id_reset").value;
    const r = await fetch(`/reset?task_id=${task}`, {method:"POST"});
    document.getElementById("reset_out").textContent = JSON.stringify(await r.json(), null, 2);
  }
  async function doStep() {
    const task = document.getElementById("task_id_step").value;
    const body = {
      email_id: document.getElementById("email_id").value,
      label: document.getElementById("label").value,
      priority: parseInt(document.getElementById("priority").value),
      reason: document.getElementById("reason").value,
    };
    const r = await fetch(`/step?task_id=${task}`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    document.getElementById("step_out").textContent = JSON.stringify(await r.json(), null, 2);
  }
  async function doState() {
    const task = document.getElementById("task_id_state").value;
    const r = await fetch(`/state?task_id=${task}`);
    document.getElementById("state_out").textContent = JSON.stringify(await r.json(), null, 2);
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
