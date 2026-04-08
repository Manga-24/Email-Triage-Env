"""
inference.py — Baseline inference script for EmailTriageEnv.
Uses OpenAI-compatible client with Hugging Face router.
Reads HF_TOKEN from environment variables.
Runs all three tasks and reports scores.
"""

import os
import json
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────── #
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
BASE_URL  = os.environ.get("BASE_URL", "https://router.huggingface.co/v1/")
MODEL     = os.environ.get("MODEL", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL   = os.environ.get("ENV_URL", "http://localhost:7860")
TASKS     = ["easy", "medium", "hard"]

client = OpenAI(api_key=HF_TOKEN, base_url=BASE_URL)

SYSTEM_PROMPT = """You are an expert email triage assistant.
For each email you receive, you must respond with ONLY a JSON object in this exact format:
{
  "email_id": "<the email_id from the observation>",
  "label": "<one of: urgent, important, normal, spam>",
  "priority": <integer 1-5, where 1=highest priority>,
  "reason": "<one sentence explaining your classification>"
}

Label definitions:
- urgent: requires immediate action (today)
- important: needs attention this week
- normal: routine, low-pressure communication
- spam: unsolicited, scam, or irrelevant

Do NOT include any text outside the JSON object."""


def build_user_prompt(obs: dict) -> str:
    return f"""Triage this email:

Email ID:   {obs['email_id']}
From:       {obs['sender']}
Subject:    {obs['subject']}
Timestamp:  {obs['timestamp']}
Body:
{obs['body']}

Remaining emails in queue: {obs['remaining_emails']}
Current session score: {obs['current_score']}"""


def call_llm(messages: list) -> dict:
    """Call LLM and parse JSON action."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=256,
        temperature=0.1,
    )
    content = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def run_task(task_id: str) -> float:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # Reset
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()
    print(f"  Reset OK. First email: [{obs['email_id']}] {obs['subject']}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    step = 0

    while True:
        # Build prompt
        user_msg = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_msg})

        # LLM inference
        try:
            action = call_llm(messages)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [Step {step}] LLM parse error or request failed: {repr(e)}. Defaulting to normal/3.")
            action = {
                "email_id": obs["email_id"],
                "label": "normal",
                "priority": 3,
                "reason": "Could not parse response, defaulting.",
            }

        messages.append({"role": "assistant", "content": json.dumps(action)})

        # Step environment
        r = requests.post(
            f"{ENV_URL}/step",
            params={"task_id": task_id},
            json=action,
        )
        r.raise_for_status()
        result = r.json()

        reward  = result["reward"]
        done    = result["done"]
        obs     = result["observation"]

        print(
            f"  [Step {step+1}] {action['email_id']} → "
            f"label={action['label']}, priority={action['priority']} | "
            f"reward={reward:.2f} | msg: {obs['message'][:80]}..."
        )

        step += 1
        if done:
            break

    # Final score
    state_r = requests.get(f"{ENV_URL}/state", params={"task_id": task_id})
    state   = state_r.json()
    n       = len(state["emails"])
    final   = round(state["total_score"] / n, 4) if n else 0.0
    print(f"\n  ✅ Task '{task_id}' complete. Final score: {final:.4f}")
    return final


def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    print("\n🚀 EmailTriageEnv — Baseline Inference")
    print(f"   Model : {MODEL}")
    print(f"   Env   : {ENV_URL}")

    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"  ❌ Task '{task_id}' failed: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*60}")
    print("  BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<10} {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}\n")
    return scores


if __name__ == "__main__":
    main()
