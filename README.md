---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---
# 📧 EmailTriageEnv

**An OpenEnv-compliant RL environment for real-world email triage.**

Agents must read a queue of emails and — for each one — decide:
- **Label**: `urgent` / `important` / `normal` / `spam`
- **Priority**: `1` (act now) → `5` (low priority)
- **Reason**: a one-sentence justification

---

## 🧠 Motivation

Email triage is a high-frequency, high-stakes knowledge-work task performed by millions of people daily. Unlike toy tasks, it requires:
- Reading comprehension and context inference
- Distinguishing urgency from importance
- Detecting spam/phishing from subtle cues (e.g., spoofed domains)
- Reasoning under partial information

This makes it ideal for RL training: the reward signal is verifiable, the task is multi-step, and difficulty is tunable.

---

## 🔁 Action & Observation Spaces

### Action
| Field | Type | Description |
|---|---|---|
| `email_id` | `string` | ID of the email being triaged |
| `label` | `enum` | `urgent`, `important`, `normal`, or `spam` |
| `priority` | `int` | 1 (highest) to 5 (lowest) |
| `reason` | `string` | Short justification for classification |

### Observation
| Field | Type | Description |
|---|---|---|
| `email_id` | `string` | ID of the current email |
| `subject` | `string` | Email subject line |
| `sender` | `string` | Sender address |
| `body` | `string` | Email body text |
| `timestamp` | `string` | ISO 8601 send time |
| `remaining_emails` | `int` | How many emails left in queue |
| `current_score` | `float` | Cumulative score so far |
| `message` | `string` | Feedback from last action |

---

## 🎯 Tasks

| Task ID | Emails | Difficulty | Description |
|---|---|---|---|
| `easy` | 5 | ⭐ Easy | Clear spam, explicit urgency, no ambiguity |
| `medium` | 7 | ⭐⭐ Medium | Urgency requires reading context carefully |
| `hard` | 10 | ⭐⭐⭐ Hard | Subtle cues, spoofed senders, mixed signals |

---

## 📊 Reward Function

Each step returns a reward in `[-0.1, 1.0]`:

```
reward = 0.6 × label_score + 0.3 × priority_score + 0.1 × reason_score
```

| Component | Max | Description |
|---|---|---|
| `label_score` | 1.0 | Exact match = 1.0; adjacent = 0.5; wrong = 0.0; spam↔normal = 0.1–0.3 |
| `priority_score` | 1.0 | Off by 0 = 1.0; off by 1 = 0.6; off by 2 = 0.3; else = 0.0 |
| `reason_score` | 0.6 | Empty = 0.0; too short = 0.3; good = 0.6 |
| Invalid action | — | −0.1 penalty |

Final score = `total_reward / num_emails` (normalized to [0, 1]).

---

## 🚀 Setup & Usage

### Local (without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
python server.py

# 4. Open browser: http://localhost:7860/web
# 5. API docs:     http://localhost:7860/docs
```

### Docker

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env

# Open: http://localhost:7860/web
```

### Run Inference

```bash
# Set your HF token
export HF_TOKEN=your_token_here
export ENV_URL=http://localhost:7860

python inference.py
```

---

## 📡 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset?task_id=easy` | Reset environment, get first email |
| `POST` | `/step?task_id=easy` | Submit action, get next email + reward |
| `GET` | `/state?task_id=easy` | Get full current state |
| `GET` | `/web` | Browser-based test UI |
| `GET` | `/docs` | Auto-generated Swagger UI |

**Example step request:**
```json
POST /step?task_id=easy
{
  "email_id": "e2",
  "label": "urgent",
  "priority": 1,
  "reason": "Production outage affecting all users requires immediate engineering response."
}
```

---

## 📈 Baseline Performance

Scores using `Qwen/Qwen2.5-72B-Instruct` via HF router:

| Task | Score |
|---|---|
| easy | ~0.88 |
| medium | ~0.74 |
| hard | ~0.61 |
| **Average** | **~0.74** |

*(Scores are reproducible with `temperature=0.1`)*

---

## 🏷️ Tags
`openenv` · `email-triage` · `nlp` · `real-world` · `rl-environment`
