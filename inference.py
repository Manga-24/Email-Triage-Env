import os
import requests

# ===== ENV VARIABLES =====
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
ENV_URL = os.environ.get("ENV_URL", "https://manga-navya-email-triage-env.hf.space")

TASKS = ["easy", "medium", "hard"]


def run_task(task_id):
    step = 0
    rewards = []
    success = False

    try:
        # ===== START =====
        print(f"[START] task={task_id} env=email model={MODEL}", flush=True)

        r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
        obs = r.json()

        done = False

        while not done and step < 10:
            step += 1

            action = {
                "email_id": obs.get("email_id", "0"),
                "label": "normal",
                "priority": 3,
                "reason": "default"
            }

            r = requests.post(
                f"{ENV_URL}/step",
                params={"task_id": task_id},
                json=action
            )

            result = r.json()

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = result.get("error", None)

            obs = result.get("observation", {})

            rewards.append(reward)

            # FIX: error must be null (not None)
            error_str = "null" if error is None else str(error)

            # ===== STEP =====
            print(
                f"[STEP] step={step} action=classify_email reward={reward:.2f} done={str(done).lower()} error={error_str}",
                flush=True
            )

        success = done

    except Exception as e:
        print(
            f"[STEP] step=0 action=null reward=0.00 done=true error={str(e)}",
            flush=True
        )

    finally:
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.00"

        # ===== END =====
        print(
            f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}",
            flush=True
        )


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is required")

    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()