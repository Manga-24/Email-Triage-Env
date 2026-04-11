import os
import requests
from openai import OpenAI

# ===== ENV VARIABLES =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = "https://manga-navya-email-triage-env.hf.space"

client = None

# ===== SAFE CLIENT INITIALIZATION =====
try:
    if HF_TOKEN:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
except Exception:
    client = None


def run_task(task_name):
    step = 0
    rewards = []
    success = False

    try:
        # ===== START =====
        print(f"[START] task={task_name} env=email model={MODEL_NAME}", flush=True)

        # RESET ENV
        try:
            res = requests.get(f"{ENV_URL}/reset", params={"task_id": task_name}, timeout=5)
            data = res.json()
        except Exception as e:
            print(f"[STEP] step=0 action=null reward=0.00 done=true error={str(e)}", flush=True)
            return

        done = False

        while not done and step < 5:
            step += 1

            # ===== SAFE LLM CALL =====
            try:
                if client:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": "classify email"}]
                    )
                    action = response.choices[0].message.content.strip()
                else:
                    action = "classify_email"
            except Exception:
                action = "classify_email"

            # ===== ENV STEP =====
            try:
                res = requests.get(
                    f"{ENV_URL}/step",
                    params={"task_id": task_name, "action": action},
                    timeout=5
                )
                data = res.json()
            except Exception as e:
                print(
                    f"[STEP] step={step} action={action} reward=0.00 done=true error={str(e)}",
                    flush=True
                )
                break

            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            error = data.get("error", None)

            rewards.append(f"{reward:.2f}")

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

        print(
            f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}",
            flush=True
        )


if __name__ == "__main__":
    run_task("easy")