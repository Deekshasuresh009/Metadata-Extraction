# Document Metadata Extraction

An automated machine learning system for extracting structured metadata from unstructured rental agreements and commercial contracts in native DOCX and scanned image (PNG/JPG) formats.

---

## Overview

In legal and commercial workflows, rental agreements are often drafted across varied layouts, formats, and phrasing styles without fixed structural schemas. This project implements an end-to-end extraction pipeline that extracts the six key contract metadata fields:

- **Agreement Value**: Monthly rental amount (integer).
- **Agreement Start Date**: Tenancy commencement / start date (`DD.MM.YYYY`).
- **Agreement End Date**: Tenancy expiration / end date (`DD.MM.YYYY`).
- **Renewal Notice (Days)**: Required prior termination/vacation notice period (integer days).
- **Party One**: Primary landlord / lessor / first party name.
- **Party Two**: Primary tenant / lessee / second party name.

---

## Solution Approach

The system uses a 100% semantic Machine Reading Comprehension (MRC) architecture built on a pretrained Transformer Question Answering model (`deepset/minilm-uncased-squad2`). The pipeline operates without rule-based metadata extraction (no regex field extraction, no static keyword-to-value dictionaries, and no document-specific conditions).

```
Document (.docx / .png)
       │
       ▼
Document Loading / OCR  (python-docx for DOCX; Windows Media OCR / pytesseract for images)
       │
       ▼
Text Preprocessing      (Unicode NFKC normalization and whitespace standardization)
       │
       ▼
Semantic Question Answering  (deepset/minilm-uncased-squad2 with sliding-window QA)
       │
       ▼
Metadata Normalization  (General-purpose dateutil parsing, word2number conversion)
       │
       ▼
Structured Output       (CSV / JSON format)
```

### Key Components:
1. **Document Loading & OCR (`src/document_loader.py`, `src/ocr_processor.py`)**: Ingests `.docx` files by extracting paragraphs and table cells, and processes scanned `.png` images using high-accuracy OCR with Lanczos image enhancement.
2. **Text Preprocessing (`src/text_preprocessor.py`)**: Performs generic NFKC Unicode normalization and whitespace cleaning.
3. **Semantic Question Answering (`src/semantic_extractor.py`)**: Uses multi-query extractive question answering with sliding context windows (`max_length=512`, `stride=256`). Answer candidate spans across windows are scored and ranked using joint start/end logits.
4. **General-Purpose Type Normalization**: Standard libraries (`python-dateutil`, `word2number`) normalize unstructured answer spans into standard formats (e.g., converting "two months" to `60` days, formatting dates to `DD.MM.YYYY`).
5. **Pipeline Coordinator (`src/predictor.py`)**: Connects all stages from raw file input to structured dictionary output.

---

## Project Structure

```
USEReady-Metadata-Extraction/
│
├── api/
│   ├── __init__.py
│   └── app.py
│
├── data/
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── test.csv
│
├── outputs/
│   ├── predictions.csv
│   └── recall_results.json
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── evaluator.py
│   ├── ocr_processor.py
│   ├── predictor.py
│   ├── semantic_extractor.py
│   └── text_preprocessor.py
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

## Requirements

- Python 3.9, 3.10, or 3.11
- Windows 10/11 (for Windows Native OCR) or Linux/macOS (with Tesseract OCR)
- Dependencies listed in `requirements.txt`:
  - `torch>=2.0.0`
  - `transformers>=4.35.0`
  - `python-docx>=1.1.0`
  - `pandas>=2.0.0`
  - `pillow>=10.0.0`
  - `pytesseract>=0.3.10`
  - `winocr>=0.0.15` (on Windows)
  - `fastapi>=0.100.0`
  - `uvicorn>=0.23.0`
  - `pydantic>=2.0.0`
  - `python-dateutil>=2.8.2`
  - `word2number>=1.1`

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/USEReady-Metadata-Extraction.git
   cd USEReady-Metadata-Extraction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*(Note: The pretrained model `deepset/minilm-uncased-squad2` is automatically downloaded from Hugging Face Hub on first execution.)*

---

## Run Predictions

To run batch inference on the test dataset and generate predictions:

```bash
# Run extraction on test documents and save to outputs/predictions.csv
python main.py --input data/test --output outputs/predictions.csv
```

To extract metadata from a single document:

```bash
python main.py --file data/test/24158401-Rental-Agreement.png
```

To run inference on test documents and evaluate exact-match recall against test ground truth:

```bash
python main.py --input data/test --evaluate data/test.csv --output outputs/predictions.csv
```

---

## Evaluation

The assignment evaluates metadata extraction using exact-match recall per field:

$$\text{Recall} = \frac{\text{True (Exact Matches)}}{\text{True} + \text{False}}$$

### Official Test Set Results (4 Test Documents, 24 Target Fields):

| Target Field | True Matches | False Matches | Total Fields | Exact-Match Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Agreement Value** | 4 | 0 | 4 | **100.0%** |
| **Agreement Start Date** | 3 | 1 | 4 | **75.0%** |
| **Agreement End Date** | 1 | 3 | 4 | **25.0%** |
| **Renewal Notice (Days)** | 1 | 3 | 4 | **25.0%** |
| **Party One** | 1 | 3 | 4 | **25.0%** |
| **Party Two** | 0 | 4 | 4 | **0.0%** |
| **OVERALL RECALL** | **10** | **14** | **24** | **41.67%** |

---

## REST API

A FastAPI web service is provided in `api/app.py` for real-time document extraction.

### Start the API Server:

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

### Interactive Documentation (Swagger UI):

Open your browser and navigate to:
```
http://127.0.0.1:8000/docs
```

### Upload and Extract via `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/24158401-Rental-Agreement.png"
```

### Example API Response:

```json
{
  "file_name": "24158401-Rental-Agreement.png",
  "agreement_value": 12000,
  "agreement_start_date": "01.04.2008",
  "agreement_end_date": "01.03.2009",
  "renewal_notice_days": 60,
  "party_one": "herby",
  "party_two": "The Tenant : Lessee",
  "raw_spans": {
    "value": "Rs. 12000/-",
    "start_date": "1st day of April 2008",
    "end_date": "1st day of March 2009",
    "notice": "two months",
    "party_one": "herby",
    "party_two": "The Tenant : Lessee"
  },
  "confidence_scores": {
    "value": 13.26,
    "start_date": 13.98,
    "end_date": 14.66,
    "notice": 12.05,
    "party_one": 13.58,
    "party_two": 9.30
  }
}
```

---

## Output

The final results are stored in:
- `outputs/predictions.csv`: Predictions table for all test documents.
- `outputs/recall_results.json`: Quantitative per-field and overall recall metrics.
