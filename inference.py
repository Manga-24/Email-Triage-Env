import os
import requests
from openai import OpenAI

# ===== ENV VARIABLES =====
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

ENV_URL = "https://manga-navya-email-triage-env.hf.space"

if not API_BASE_URL or not API_KEY:
    raise ValueError("API_BASE_URL and API_KEY must be set")

# Initialize OpenAI client (LiteLLM proxy)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


def run_task(task_name):
    step = 0
    rewards = []
    success = False

    try:
        # ===== START =====
        print(f"[START] task={task_name} env=email model={MODEL_NAME}", flush=True)

        # RESET ENV
        res = requests.get(f"{ENV_URL}/reset", params={"task_id": task_name})
        data = res.json()

        done = False
        error = None

        while not done and step < 5:
            step += 1

            # ===== LLM CALL (MANDATORY) =====
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": "Classify this email"}
                    ]
                )
                action = response.choices[0].message.content.strip()
                if not action:
                    action = "classify_email"
            except Exception:
                action = "classify_email"

            # CALL ENV STEP
            res = requests.get(
                f"{ENV_URL}/step",
                params={"task_id": task_name, "action": action}
            )

            data = res.json()

            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            error = data.get("error", None)

            rewards.append(f"{reward:.2f}")

            # ===== STEP OUTPUT =====
            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}",
                flush=True
            )

        success = done

    except Exception as e:
        print(
            f"[STEP] step=0 action=null reward=0.00 done=true error={str(e)}",
            flush=True
        )

    finally:
        rewards_str = ",".join(rewards) if rewards else "0.00"

        # ===== END =====
        print(
            f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}",
            flush=True
        )


if __name__ == "__main__":
    run_task("easy")