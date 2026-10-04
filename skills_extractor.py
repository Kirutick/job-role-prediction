"""skills_extractor.py
======================
Extracts technical and domain skills from resume text using a predefined
vocabulary. Does not hallucinate skills; only returns those present in the text.
"""
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class SkillsExtractor:
    def __init__(self):
        # We will build a set of lowercased skills from role_requirements.json
        # to act as our vocabulary.
        self.vocabulary = set()
        req_path = BASE_DIR / "data" / "role_requirements.json"
        
        if req_path.exists():
            with open(req_path, "r", encoding="utf-8") as f:
                reqs = json.load(f)
                for role, data in reqs.items():
                    for skill in data.get("skills", []):
                        self.vocabulary.add(skill.lower())
        
        # Add some common ones just in case
        extra_skills = [
            "c#", "ruby", "php", "go", "rust", "scala", "kotlin", "swift",
            "html5", "css3", "angular", "vue.js", "django", "flask", "spring boot",
            "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git",
            "machine learning", "deep learning", "nlp", "computer vision",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
            "agile", "scrum", "jira", "confluence", "linux", "unix", "bash"
        ]
        for skill in extra_skills:
            self.vocabulary.add(skill.lower())

        # Pre-compile regexes for each skill (word boundary, case insensitive)
        self.skill_patterns = {}
        for skill in self.vocabulary:
            # Escape special regex chars in skill names (like C++)
            escaped = re.escape(skill)
            # Use \b unless it starts/ends with non-word char (like C++)
            prefix = r"\b" if re.match(r"\w", skill[0]) else r"(?:^|\s)"
            suffix = r"\b" if re.match(r"\w", skill[-1]) else r"(?:$|\s)"
            
            # Special case for C++ / C#
            if skill in ["c++", "c#"]:
                pattern = re.compile(rf"{prefix}{escaped}(?!\w)", re.IGNORECASE)
            else:
                pattern = re.compile(rf"{prefix}{escaped}{suffix}", re.IGNORECASE)
                
            self.skill_patterns[skill] = pattern

        # Keep original casing mapping for display
        self.display_names = {}
        if req_path.exists():
            with open(req_path, "r", encoding="utf-8") as f:
                reqs = json.load(f)
                for role, data in reqs.items():
                    for skill in data.get("skills", []):
                        self.display_names[skill.lower()] = skill
        for skill in extra_skills:
            if skill.lower() not in self.display_names:
                # Title case heuristic
                self.display_names[skill.lower()] = skill.title()
        
        # Hardcode specific casings
        overrides = {
            "html5": "HTML5", "css3": "CSS3", "vue.js": "Vue.js",
            "postgresql": "PostgreSQL", "mongodb": "MongoDB",
            "aws": "AWS", "gcp": "GCP", "nlp": "NLP",
            "scikit-learn": "Scikit-learn", "tensorflow": "TensorFlow",
            "pytorch": "PyTorch", "jira": "Jira"
        }
        self.display_names.update(overrides)

    def extract_skills(self, text: str) -> list[str]:
        if not text:
            return []
            
        detected = []
        for skill_lower, pattern in self.skill_patterns.items():
            if pattern.search(text):
                detected.append(self.display_names.get(skill_lower, skill_lower.title()))
                
        return sorted(detected)

# Singleton instance
_extractor = None

def get_skills_extractor():
    global _extractor
    if _extractor is None:
        _extractor = SkillsExtractor()
    return _extractor
