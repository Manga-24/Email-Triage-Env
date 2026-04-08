import requests

def test_easy():
    print("Testing /reset on 'easy' task...")
    res = requests.post("http://localhost:7860/reset?task_id=easy")
    assert res.status_code == 200
    obs = res.json()
    print("Reset response:", obs)
    assert 'email_id' in obs
    assert obs['remaining_emails'] == 5
    
    # Try an action
    action = {
        "email_id": obs['email_id'],
        "label": "spam",
        "priority": 5,
        "reason": "This looks like a test"
    }
    
    print("\nTesting /step...")
    res = requests.post("http://localhost:7860/step?task_id=easy", json=action)
    assert res.status_code == 200
    step_resp = res.json()
    print("Step response (Reward):", step_resp['reward'])
    assert 'reward' in step_resp
    assert 'observation' in step_resp
    assert 'done' in step_resp
    
    # Check state
    print("\nTesting /state...")
    res = requests.get("http://localhost:7860/state?task_id=easy")
    assert res.status_code == 200
    state = res.json()
    print("State current index:", state['current_index'])
    assert state['current_index'] == 1
    
    print("\n✅ All endpoint checks passed successfully!")

if __name__ == "__main__":
    test_easy()
