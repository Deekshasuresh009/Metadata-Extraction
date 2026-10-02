"""
RESTful API Web Service Module
Provides high-performance endpoints for real-time document metadata extraction.
Uses the production pipeline: DocumentLoader -> OCR -> Preprocessor -> SemanticExtractor
"""

import os
import sys
import shutil
import tempfile
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import MetadataPredictor

app = FastAPI(
    title="USEReady Contract Metadata Extraction API",
    description="Automated AI/ML service for extracting key metadata from rental agreements (.docx and .png).",
    version="1.0.0"
)

# Initialize predictor singleton
predictor = MetadataPredictor()

class ExtractionResponse(BaseModel):
    file_name: str = Field(..., description="Uploaded document name")
    agreement_value: Optional[int] = Field(None, description="Monthly rent amount (numeric integer)")
    agreement_start_date: Optional[str] = Field(None, description="Agreement commencement date (DD.MM.YYYY)")
    agreement_end_date: Optional[str] = Field(None, description="Agreement termination date (DD.MM.YYYY)")
    renewal_notice_days: Optional[int] = Field(None, description="Renewal/termination notice period in integer days")
    party_one: Optional[str] = Field(None, description="Lessor / Landlord / Property owner entity name")
    party_two: Optional[str] = Field(None, description="Lessee / Tenant / Renting party entity name")
    raw_spans: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-extracted raw token spans")
    confidence_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="QA model logit confidence scores")

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "USEReady Contract Metadata Extraction API",
        "status": "healthy",
        "model": "deepset/minilm-uncased-squad2 (Extractive QA Transformer)",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "service_ready": True}

@app.get("/info", tags=["Metadata"])
def info():
    return {
        "supported_formats": [".docx", ".png", ".jpg", ".jpeg"],
        "target_fields": [
            "Agreement Value",
            "Agreement Start Date",
            "Agreement End Date",
            "Renewal Notice (Days)",
            "Party One",
            "Party Two"
        ],
        "extraction_methodology": "Pretrained Transformer Machine Reading Comprehension (SQuAD 2.0)"
    }

@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_metadata(file: UploadFile = File(...)):
    """
    Extracts 6 target metadata fields from an uploaded contract file (.docx or .png).
    """
    filename = file.filename or "uploaded_document"
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in [".docx", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: .docx, .png, .jpg, .jpeg"
        )

    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process document through production pipeline
        result = predictor.process_document(temp_path)
        
        return ExtractionResponse(
            file_name=filename,
            agreement_value=result.get("Aggrement Value"),
            agreement_start_date=result.get("Aggrement Start Date"),
            agreement_end_date=result.get("Aggrement End Date"),
            renewal_notice_days=result.get("Renewal Notice (Days)"),
            party_one=result.get("Party One"),
            party_two=result.get("Party Two"),
            raw_spans=result.get("_raw_spans", {}),
            confidence_scores=result.get("_confidence_scores", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
