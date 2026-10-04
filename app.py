"""FastAPI service for resume job-role prediction.

Run locally from the project directory:
    uvicorn app:app --reload
"""

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pypdf import PdfReader
from pydantic import BaseModel, Field

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional on serverless deployments
    pytesseract = None

from predict_role import _load_artifacts
from preprocess_resume import clean_text
import joblib
import json
from screening_engine import get_screening_engine

BASE_DIR = Path(__file__).resolve().parent

class PredictionRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)

class PredictionResponse(BaseModel):
    predicted_role: str
    confidence: float | None = None
    top_predictions: list[dict] = []

class AnalyzeResponse(PredictionResponse):
    detected_skills: list[str] = []
    missing_common_skills: list[str] = []
    screening_breakdown: dict = {}


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class ResumeRolePredictor:
    """Own loaded inference artifacts for the application lifetime."""

    def __init__(self) -> None:
        self.model_type = "tfidf"
        self.model = None
        self.vectorizer = None
        self.label_classes = []
        self.svd = None
        self.screening_engine = get_screening_engine()
        
        metadata_path = BASE_DIR / "metadata.json"
        models_dir = BASE_DIR / "models"
        
        try:
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.model_type = meta.get("active_model", "tfidf")

            if self.model_type == "semantic":
                print("Loading LSA Semantic Model...")
                self.model = joblib.load(models_dir / "semantic_model.pkl")
                self.vectorizer = joblib.load(models_dir / "semantic_vectorizer.pkl")
                self.svd = joblib.load(models_dir / "semantic_svd.pkl")
                self.label_classes = joblib.load(models_dir / "semantic_label_classes.pkl")
            else:
                print("Loading TF-IDF Baseline Model...")
                self.model = joblib.load(models_dir / "tfidf_model.pkl")
                self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
                self.label_classes = joblib.load(models_dir / "tfidf_label_classes.pkl")
        except FileNotFoundError:
            try:
                print("New models not found, loading original artifacts as fallback.")
                self.model, self.vectorizer, self.label_classes = _load_artifacts()
                self.model_type = "tfidf"
            except Exception as error:
                raise RuntimeError(
                    "Model artifacts are unavailable in this deployment. The serverless bundle is missing required inference files."
                ) from error

        if self.model is None or self.vectorizer is None:
            raise RuntimeError(
                "Model artifacts are unavailable in this deployment. The serverless bundle is missing required inference files."
            )

    def predict(self, resume_text: str) -> dict[str, object]:
        if self.model is None or self.vectorizer is None:
            raise RuntimeError(
                "Model artifacts are unavailable in this deployment. The serverless bundle is missing required inference files."
            )
        if not resume_text.strip():
            raise ValueError("resume_text cannot be empty or whitespace-only.")

        cleaned_text = clean_text(resume_text)
        if not isinstance(cleaned_text, str) or not cleaned_text.strip():
            raise ValueError("resume_text contains no usable text after preprocessing.")

        features = self.vectorizer.transform([cleaned_text])
        if self.svd:
            features = self.svd.transform(features)
            
        predicted_role = str(self.model.predict(features)[0])
        confidence = None
        top_predictions = []
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            confidence = float(probs.max())
            
            top_indices = probs.argsort()[-3:][::-1]
            for idx in top_indices:
                top_predictions.append({
                    "role": str(self.model.classes_[idx]),
                    "confidence": float(probs[idx])
                })
                
        return {
            "predicted_role": predicted_role, 
            "confidence": confidence,
            "top_predictions": top_predictions
        }
        
    def analyze(self, resume_text: str) -> dict:
        pred_result = self.predict(resume_text)
        screen_result = self.screening_engine.analyze(
            resume_text=resume_text,
            predicted_role=pred_result["predicted_role"],
            confidence=pred_result["confidence"]
        )
        return {**pred_result, **screen_result}


def extract_uploaded_text(filename: str, content: bytes) -> str:
    """Extract text from a PDF or image without changing the model pipeline."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if pytesseract is None:
        raise RuntimeError(
            "Image OCR is disabled in this deployment because the Tesseract engine is not available in the serverless environment. "
            "Use PDF or pasted text input instead."
        )

    try:
        return pytesseract.image_to_string(Image.open(BytesIO(content)))
    except Exception as error:
        if hasattr(pytesseract, "TesseractNotFoundError"):
            tesseract_error = pytesseract.TesseractNotFoundError
            if isinstance(error, tesseract_error):
                raise RuntimeError(
                    "Image OCR requires the Tesseract engine to be installed and available on PATH."
                ) from error
        raise ValueError("The uploaded image could not be read.") from error


def get_predictor(application_request: Request) -> ResumeRolePredictor:
    """Ensure startup-loaded prediction artifacts exist even in lightweight serverless/test execution."""
    predictor = getattr(application_request.app.state, "predictor", None)
    if predictor is None:
        try:
            predictor = ResumeRolePredictor()
        except RuntimeError as error:
            raise RuntimeError(
                "Model artifacts are unavailable in this deployment. The serverless bundle is missing required inference files."
            ) from error
        application_request.app.state.predictor = predictor
    return predictor


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.predictor = None
    try:
        application.state.predictor = ResumeRolePredictor()
    except Exception as error:
        print(f"Startup model load failed: {error}")
    yield


app = FastAPI(
    title="Resume Job-Role Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, application_request: Request) -> PredictionResponse:
    """Predict a job role from resume text using startup-loaded artifacts."""
    if not request.resume_text.strip():
        raise HTTPException(status_code=422, detail="resume_text cannot be empty or whitespace-only.")

    predictor: ResumeRolePredictor = get_predictor(application_request)
    try:
        result = predictor.predict(request.resume_text)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed.") from error
    return PredictionResponse(**result)
    
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: PredictionRequest, application_request: Request) -> AnalyzeResponse:
    """Predict role, extract skills, and generate screening breakdown."""
    if not request.resume_text.strip():
        raise HTTPException(status_code=422, detail="resume_text cannot be empty or whitespace-only.")

    predictor: ResumeRolePredictor = get_predictor(application_request)
    try:
        result = predictor.analyze(request.resume_text)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Analysis failed.") from error
    return AnalyzeResponse(**result)


@app.post("/predict-file", response_model=PredictionResponse)
async def predict_file(application_request: Request, upload: UploadFile = File(...)) -> PredictionResponse:
    """Extract text from a PDF/image upload and predict its job role."""
    filename = upload.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload a PDF or supported image file.")

    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large; maximum size is 10 MB.")
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    try:
        extracted_text = extract_uploaded_text(filename, content)
        predictor: ResumeRolePredictor = get_predictor(application_request)
        result = predictor.predict(extracted_text)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="File prediction failed.") from error
    return PredictionResponse(**result)


@app.get("/health")
def health(application_request: Request) -> dict[str, Any]:
    """Return a simple readiness response for local checks."""
    predictor = getattr(application_request.app.state, "predictor", None)
    if predictor is None:
        try:
            predictor = ResumeRolePredictor()
            application_request.app.state.predictor = predictor
        except Exception:
            predictor = None
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "message": (
            "Model artifacts unavailable in this serverless deployment."
            if predictor is None else "Model ready."
        ),
    }


app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="frontend")