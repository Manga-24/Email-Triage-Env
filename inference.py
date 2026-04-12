import os
import requests
from openai import OpenAI

# ===== ENV VARIABLES (SAFE FOR ALL ENVIRONMENTS) =====
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY")

MODEL_NAME = "gpt-3.5-turbo"
# IMPORTANT: use localhost (HF container internal communication)
ENV_URL = "http://localhost:7860"

# ===== OPENAI CLIENT =====
client = None
if API_KEY:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
    except Exception:
        client = None


def get_action():
    if client:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": "Return an action like classify(email_id=1,label=spam)"
                }]
            )

            content = response.choices[0].message.content
            return content.strip() if content else "classify(email_id=1,label=spam)"

        except Exception:
            return "classify(email_id=1,label=spam)"

    return "classify(email_id=1,label=spam)"


def safe_request(url, params):
    """
    Safe HTTP request wrapper
    """
    try:
        res = requests.get(url, params=params, timeout=10)
        return res.json()
    except Exception as e:
        return {"error": str(e), "reward": 0.0, "done": True}


def run_task(task_name):
    step = 0
    rewards = []
    success = False

    # ===== START =====
    print(f"[START] task={task_name} env=email model={MODEL_NAME}", flush=True)

    try:
        # RESET
        data = safe_request(f"{ENV_URL}/reset", {"task_id": task_name})
        done = False

        while not done and step < 5:
            step += 1

            # ===== GET ACTION =====
            action = get_action()

            # ===== STEP =====
            data = safe_request(
                f"{ENV_URL}/step",
                {"task_id": task_name, "action": action}
            )

            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            error = data.get("error", None)

            rewards.append(f"{reward:.2f}")

            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} "
                f"done={str(done).lower()} error={error}",
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

        # ===== END (ALWAYS PRINTED) =====
        print(
            f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}",
            flush=True
        )


if __name__ == "__main__":
    run_task("easy")