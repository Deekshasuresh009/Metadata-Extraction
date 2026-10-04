# Document Metadata Extraction

An AI/ML system that extracts important metadata from rental agreements and similar documents in DOCX and scanned image formats.

---

## Overview

Rental agreements and contracts often come in different formats, such as native Word documents (`.docx`) or scanned images (`.png`, `.jpg`). Because formatting, clause ordering, and phrasing vary across documents, simple rule-based methods like regex or fixed-coordinate matching frequently break.

This project implements an end-to-end extraction pipeline using a pretrained Transformer Question Answering model (`deepset/minilm-uncased-squad2`). The system asks semantic questions about each field, identifies candidate answer spans, scores them using field-specific logic, and normalizes the results into structured data.

---

## Metadata Fields

The pipeline extracts six key metadata fields from each agreement:

| Field | Description | Standardized Output |
|---|---|---|
| **Agreement Value** | Monthly rental amount | Integer (e.g., `12000`) |
| **Agreement Start Date** | Commencement date of the agreement | `DD.MM.YYYY` (e.g., `01.04.2008`) |
| **Agreement End Date** | Termination/expiry date of the agreement | `DD.MM.YYYY` (e.g., `31.03.2009`) |
| **Renewal Notice (Days)** | Required notice period for termination/vacation | Integer days (e.g., `30`, `60`) |
| **Party One** | Landlord / lessor / property owner | Entity / Name (e.g., `Hanumaiah`) |
| **Party Two** | Tenant / lessee / second party | Entity / Name (e.g., `Vishal Bhardwaj`) |

---

## How It Works

Instead of relying on rigid templates or keyword offsets, the pipeline frames metadata discovery as an extractive Question Answering task.

```
Document (DOCX / Scanned Image)
   │
   ▼
Document Loading / OCR
   │
   ▼
Text Preprocessing
   │
   ▼
Semantic Question Answering (deepset/minilm-uncased-squad2)
   │
   ▼
Candidate Selection & Scoring
   │
   ▼
Metadata Normalization
   │
   ▼
Structured Output (CSV / JSON API)
```

---

## Main Components

### 1. Document Loading and OCR (`src/document_loader.py`, `src/ocr_processor.py`)
- **DOCX Documents:** Paragraphs and table cell contents are extracted in reading order using `python-docx`.
- **Scanned Images:** Image quality is enhanced before OCR (rescaled using `LANCZOS` to a minimum width of 1600px, with contrast and sharpness adjustments).
- **OCR Hierarchy:** Primary text extraction uses Windows Native Media OCR (`winocr`), with automatic fallback to Tesseract (`pytesseract`) or EasyOCR (`easyocr`).

### 2. Text Preprocessing (`src/text_preprocessor.py`)
- Standardizes text using Unicode NFKC normalization.
- Cleans up invisible characters, extra whitespace, and inconsistent line breaks.

### 3. Semantic Extraction (`src/semantic_extractor.py`)
- Uses `deepset/minilm-uncased-squad2` to locate target spans in the text.
- Poses multiple natural language questions (probes) for each field to handle different phrasing styles.
- Splits long documents into overlapping sliding context windows (`max_tokens = 384`, `stride = 128`) with question-token masking to prevent false extractions from prompt text.

### 4. Candidate Selection & Scoring (`src/semantic_extractor.py`)
- Evaluates candidate spans using their start/end logit scores and domain-aware ranking:
  - **Dates:** Scores commencement markers (*"commencing from"*, *"with effect from"*) higher than execution preamble dates (*"made on"*).
  - **Renewal Notice:** Disambiguates notice periods (*"1 month notice to vacate"*) from lease term duration (*"period of 11 months"*) and deposits.
  - **Parties:** Truncates entity spans at legal clause boundaries (*"S/o"*, *"residing at"*, *"hereinafter"*) and ensures Party One and Party Two are distinct.

### 5. Metadata Normalization (`src/semantic_extractor.py`)
- **Values:** Strips currency symbols and converts word-form numbers (e.g., *"twelve thousand"*) or digit strings to integers via `word2number`.
- **Dates:** Parses dates to `DD.MM.YYYY` format using `python-dateutil`. Dynamically calculates month-end dates (e.g., *"end of March 2009"* $\rightarrow$ `31.03.2009`) and falls back to calendar-based duration arithmetic when end dates are implicit ($\text{Start} + \text{Months} - 1\text{d}$).
- **Notice Periods:** Normalizes durations into total days ($N \times 30$ for months, $N \times 7$ for weeks, $N$ for days).
- **Parties:** Strips honorifics (*"Mr."*, *"Mrs."*, *"Dr."*, *"Sri"*) and filters out legal role boilerplate (*"lessor"*, *"lessee"*, *"witness"*).

---

## Model

- **Model:** `deepset/minilm-uncased-squad2`
- **Description:** It is a pretrained extractive Question Answering model based on MiniLM-L12-H384 and fine-tuned on SQuAD 2.0.
- **Usage:** The project uses the pretrained model for inference and does not perform additional fine-tuning on the assignment documents.

---

## Technology Used

- **Python 3.9+**
- **PyTorch** (`torch`)
- **Hugging Face Transformers** (`transformers`)
- **python-docx** (DOCX parsing)
- **winocr / pytesseract / easyocr** (OCR engines)
- **Pillow** (Image preprocessing)
- **pandas** (Data processing & evaluation)
- **FastAPI & Uvicorn** (REST API)
- **python-dateutil** (Date parsing)
- **word2number** (Number conversion)

---

## Results

Evaluation is measured using **Exact-Match Recall** against the official test set:

| Target Field | Correct | Total | Recall |
|---|:---:|:---:|:---:|
| Agreement Value | 4 | 4 | **100.0%** |
| Agreement Start Date | 4 | 4 | **100.0%** |
| Agreement End Date | 3 | 4 | **75.0%** |
| Renewal Notice (Days) | 4 | 4 | **100.0%** |
| Party One | 4 | 4 | **100.0%** |
| Party Two | 2 | 4 | **50.0%** *(75.0% Semantic)* |
| **Overall** | **21** | **24** | **87.5%** |

### Output Summary

```text
================================================================================
 PREDICTION RESULTS SUMMARY
================================================================================
                                   File Name  Aggrement Value Aggrement Start Date Aggrement End Date  Renewal Notice (Days)           Party One           Party Two
156155545-Rental-Agreement-Kns-Home.pdf.docx            12000           15.12.2012         14.11.2013                     30         V.K.NATARAJ       RAJESH CHAVDA
         228094620-Rental-Agreement.pdf.docx            15000           07.07.2013         06.06.2014                     30      KAPIL MEHROTRA           B.Kishore
               24158401-Rental-Agreement.png            12000           01.04.2008         31.03.2009                     60           Hanumaiah     Vishal Bhardwaj
               95980236-Rental-Agreement.png             9000           01.04.2010         28.02.2011                     30        S.Sakunthala       V.V.Ravi Kian
================================================================================
```

---

## Project Structure

```text
USEReady-Metadata-Extraction/
├── api/
│   ├── __init__.py
│   └── app.py                     # FastAPI REST API endpoints
├── data/
│   ├── train/                     # Training documents (DOCX & PNG)
│   ├── test/                      # Test documents (DOCX & PNG)
│   ├── train.csv                  # Training ground truth annotations
│   └── test.csv                   # Test ground truth annotations
├── models/                        # Pretrained model cache
├── outputs/
│   └── predictions.csv            # Output predictions CSV
├── src/
│   ├── __init__.py
│   ├── document_loader.py         # Document reader (DOCX & images)
│   ├── ocr_processor.py           # OCR text extractor
│   ├── text_preprocessor.py       # Text cleaning & NFKC normalization
│   ├── semantic_extractor.py      # Transformer QA extraction & normalization
│   ├── predictor.py               # Pipeline orchestrator
│   └── evaluator.py               # Exact-match recall evaluation
├── main.py                        # CLI runner
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## Installation & How to Run

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Deekshasuresh009/Metadata-Extraction.git
cd Metadata-Extraction

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run from Command Line

```bash
# Run evaluation on the test set
python main.py --evaluate data/test.csv

# Run batch extraction on a directory
python main.py --input data/test --output outputs/predictions.csv

# Run extraction on a single file
python main.py --file data/test/24158401-Rental-Agreement.png

# Run evaluation on training set
python main.py --input data/train --evaluate data/train.csv
```

### 3. Run the REST API

```bash
# Start the FastAPI server
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive API Docs:** `http://127.0.0.1:8000/docs`
- **Endpoints:**
  - `GET /` & `GET /health`: Service health check
  - `GET /info`: Metadata fields and supported formats
  - `POST /extract`: Upload a DOCX or image file to get extracted JSON metadata

**Sample API Request:**
```bash
curl -X POST "http://127.0.0.1:8000/extract" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/24158401-Rental-Agreement.png"
```

**Sample API Response:**
```json
{
  "success": true,
  "filename": "24158401-Rental-Agreement.png",
  "metadata": {
    "agreement_value": 12000,
    "agreement_start_date": "01.04.2008",
    "agreement_end_date": "31.03.2009",
    "renewal_notice_days": 60,
    "party_one": "Hanumaiah",
    "party_two": "Vishal Bhardwaj"
  },
  "extraction_time_seconds": 1.24
}
```


