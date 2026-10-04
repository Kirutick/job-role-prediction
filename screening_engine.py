"""screening_engine.py
======================
Generates a demo screening heuristic score and missing skill analysis.
"""
import json
import re
from pathlib import Path
from skills_extractor import get_skills_extractor

BASE_DIR = Path(__file__).resolve().parent

class ScreeningEngine:
    def __init__(self):
        self.role_reqs_path = BASE_DIR / "data" / "role_requirements.json"
        self.weights_path = BASE_DIR / "data" / "screening_weights.json"
        
        self.role_requirements = {}
        if self.role_reqs_path.exists():
            with open(self.role_reqs_path, "r", encoding="utf-8") as f:
                self.role_requirements = json.load(f)
                
        self.weights = {
            "role_match": 0.40,
            "skills_match": 0.40,
            "experience_match": 0.15,
            "education_match": 0.05
        }
        if self.weights_path.exists():
            with open(self.weights_path, "r", encoding="utf-8") as f:
                self.weights = json.load(f)

        self.skills_extractor = get_skills_extractor()

    def analyze(self, resume_text: str, predicted_role: str, confidence: float) -> dict:
        """
        Perform a full screening analysis for a resume given the predicted role.
        """
        # 1. Extract skills
        detected_skills = self.skills_extractor.extract_skills(resume_text)
        detected_skills_lower = {s.lower() for s in detected_skills}
        
        # 2. Determine missing skills
        req_data = self.role_requirements.get(predicted_role, {})
        required_skills = req_data.get("skills", [])
        
        missing_skills = []
        for req in required_skills:
            if req.lower() not in detected_skills_lower:
                missing_skills.append(req)
                
        # 3. Calculate scores
        # Role match is scaled confidence (demo heuristic: conf of 0.25+ gets full credit)
        # Since calibrated linear SVC might have max conf of 0.25-0.3 on this dataset.
        role_score_raw = min(1.0, confidence * 3.5) if confidence else 0.5
        
        # Skills match is % of required skills found
        if required_skills:
            skills_score_raw = 1.0 - (len(missing_skills) / len(required_skills))
        else:
            skills_score_raw = 0.5 # Default if no requirements known
            
        # Experience/Education are just dummy heuristics based on keywords for demo
        text_lower = resume_text.lower()
        exp_score_raw = 0.8 if any(w in text_lower for w in ["experience", "years", "worked", "managed"]) else 0.4
        edu_score_raw = 0.9 if any(w in text_lower for w in ["bachelor", "master", "phd", "degree", "university", "b.tech", "b.sc", "b.e", "b.a"]) else 0.3

        # Weighted final score
        final_score = (
            role_score_raw * self.weights["role_match"] +
            skills_score_raw * self.weights["skills_match"] +
            exp_score_raw * self.weights["experience_match"] +
            edu_score_raw * self.weights["education_match"]
        )

        return {
            "detected_skills": detected_skills,
            "missing_common_skills": missing_skills,
            "screening_breakdown": {
                "final_score_pct": round(final_score * 100),
                "components": {
                    "role_match": {
                        "score_pct": round(role_score_raw * 100),
                        "weight": self.weights["role_match"]
                    },
                    "skills_match": {
                        "score_pct": round(skills_score_raw * 100),
                        "weight": self.weights["skills_match"]
                    },
                    "experience_match": {
                        "score_pct": round(exp_score_raw * 100),
                        "weight": self.weights["experience_match"]
                    },
                    "education_match": {
                        "score_pct": round(edu_score_raw * 100),
                        "weight": self.weights["education_match"]
                    }
                }
            }
        }

# Singleton instance
_engine = None

def get_screening_engine():
    global _engine
    if _engine is None:
        _engine = ScreeningEngine()
    return _engine
