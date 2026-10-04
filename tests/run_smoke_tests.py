"""run_smoke_tests.py
======================
Runs the 5 sample resumes through the ML backend to verify that
the system can process unseen text and produce predictions + skills.
"""
import sys
from pathlib import Path
import json

# Add project root to path so we can import app
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import ResumeRolePredictor

def run_tests():
    print("Loading Predictor...")
    try:
        predictor = ResumeRolePredictor()
    except Exception as e:
        print(f"Failed to load predictor: {e}")
        return

    print(f"Loaded successfully. Active model: {predictor.model_type}\n")

    resumes_dir = BASE_DIR / "tests" / "sample_resumes"
    for file_path in resumes_dir.glob("*.txt"):
        print("="*60)
        print(f"Testing: {file_path.name}")
        print("="*60)
        
        text = file_path.read_text(encoding="utf-8")
        result = predictor.analyze(text)
        
        print(f"Top Prediction: {result['predicted_role']} ({result['confidence']:.2%})")
        print("Other Top Predictions:")
        for r in result['top_predictions'][1:]:
            print(f"  - {r['role']} ({r['confidence']:.2%})")
            
        print(f"\nDetected Skills ({len(result['detected_skills'])}):")
        print(", ".join(result['detected_skills']))
        
        print(f"\nMissing Common Skills ({len(result['missing_common_skills'])}):")
        print(", ".join(result['missing_common_skills']))
        
        print(f"\nScreening Score: {result['screening_breakdown']['final_score_pct']}%")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    run_tests()
