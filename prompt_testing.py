import json
from datetime import datetime
import os

def log_prompt_test(prompt_version, test_idea, analysis_result, your_assessment="", notes=""):
    """Log prompt testing results"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt_version": prompt_version,
        "test_idea": test_idea,
        "viability_score": analysis_result.get("viability_score"),
        "selected_name": analysis_result.get("product", {}).get("selected_name"),
        "your_assessment": your_assessment,
        "notes": notes
    }
    
    # Append to log file
    with open("prompt_tests.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"Logged test: {prompt_version} - Score: {log_entry['viability_score']}")