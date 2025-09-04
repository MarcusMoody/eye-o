import json
from datetime import datetime

def review_latest_test():
    """Interactive review of the most recent test"""
    try:
        with open("prompt_tests.jsonl", "r") as f:
            lines = f.readlines()
        
        if not lines:
            print("No tests found to review.")
            return
        
        # Get the latest test
        latest = json.loads(lines[-1])
        
        print(f"\n--- Test Review ---")
        print(f"Prompt Version: {latest['prompt_version']}")
        print(f"Idea: {latest['test_idea']}")
        print(f"Score: {latest['viability_score']}/10")
        print(f"Product Name: {latest.get('selected_name', 'N/A')}")
        
        # Get your assessment
        print("\nHow accurate was this score?")
        print("1. Too low  2. Accurate  3. Too high")
        assessment_choice = input("Choice (1-3): ").strip()
        
        assessment_map = {"1": "too_low", "2": "accurate", "3": "too_high"}
        assessment = assessment_map.get(assessment_choice, "unknown")
        
        # Get your notes
        notes = input("Notes (what was good/bad about this analysis?): ").strip()
        
        # Update the entry
        latest["your_assessment"] = assessment
        latest["notes"] = notes
        latest["reviewed_at"] = datetime.now().isoformat()
        
        # Rewrite the file with updated entry
        lines[-1] = json.dumps(latest) + "\n"
        with open("prompt_tests.jsonl", "w") as f:
            f.writelines(lines)
        
        print(f"✅ Review saved: {assessment} - {notes}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    review_latest_test()