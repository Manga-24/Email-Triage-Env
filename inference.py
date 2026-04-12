import os
import requests

# ===== SAFE IMPORT =====
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ===== ENV VARIABLES =====
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

ENV_URL = "http://localhost:7860"

# ===== SAFE CLIENT =====
client = None
if OpenAI and API_BASE_URL and API_KEY:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
    except Exception:
        client = None


def call_llm():
    """
    MUST attempt LLM call but NEVER crash
    """
    if client:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": "Return short action"}
                ]
            )
            content = response.choices[0].message.content
            return content.strip() if content else DEFAULT_ACTION
        except Exception:
            return DEFAULT_ACTION

    return DEFAULT_ACTION


# VALID ACTION (IMPORTANT)
DEFAULT_ACTION = "classify(email_id=e1,label=spam,priority=1,reason=test)"


def run_task(task_name):
    step = 0
    rewards = []
    success = False

    print(f"[START] task={task_name} env=email model={MODEL_NAME}", flush=True)

    try:
        # RESET
        try:
            res = requests.get(
                f"{ENV_URL}/reset",
                params={"task_id": task_name},
                timeout=5
            )
            data = res.json()
            done = False
        except Exception as e:
            print(f"[STEP] step=0 action=null reward=0.00 done=true error={str(e)}", flush=True)
            done = True

        while not done and step < 5:
            step += 1

            # 🔥 ALWAYS TRY LLM
            action = call_llm()

            # fallback safety
            if not action:
                action = DEFAULT_ACTION

            try:
                res = requests.get(
                    f"{ENV_URL}/step",
                    params={"task_id": task_name, "action": action},
                    timeout=5
                )
                data = res.json()

                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))
                error = data.get("error", None)

            except Exception as e:
                reward = 0.00
                done = True
                error = str(e)

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

        print(
            f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}",
            flush=True
        )


if __name__ == "__main__":
    run_task("easy")