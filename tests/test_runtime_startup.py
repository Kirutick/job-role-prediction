import unittest
from unittest.mock import patch

from app import ResumeRolePredictor


class RuntimeStartupTests(unittest.TestCase):
    def test_missing_model_artifacts_raise_runtime_error(self):
        with patch("app.joblib.load", side_effect=FileNotFoundError("missing artifact")):
            with self.assertRaises(RuntimeError):
                ResumeRolePredictor()


if __name__ == "__main__":
    unittest.main()
