# Document Metadata Extraction

This project extracts important metadata from rental agreements and similar documents. It supports both DOCX files and scanned documents in PNG/JPG format.

The system extracts the following six fields:

| Field | Description |
|---|---|
| **Agreement Value** | Monthly rental amount |
| **Agreement Start Date** | Start date of the agreement |
| **Agreement End Date** | End date of the agreement |
| **Renewal Notice (Days)** | Notice period required for termination or vacation |
| **Party One** | Landlord / lessor / first party |
| **Party Two** | Tenant / lessee / second party |

---

## How It Works

The main approach is based on semantic Question Answering using the pretrained Transformer model `deepset/minilm-uncased-squad2`.

Instead of depending on a fixed document layout, the system asks the QA model questions about the required fields and selects the relevant answer from the document text.

The extraction pipeline is:

```text
Document (.docx / .png / .jpg)
              |
              v
       Document Loading / OCR
              |
              v
       Text Preprocessing
              |
              v
     Semantic Question Answering
              |
              v
      Metadata Normalization
              |
              v
       CSV / JSON Output
```

The metadata extraction itself does not use regular expressions, fixed keyword-to-value mappings, or document-specific conditions.

---

## Main Components

### 1. Document Loading and OCR

Files:

- `src/document_loader.py`
- `src/ocr_processor.py`

DOCX files are processed by extracting their paragraphs and table contents. Scanned PNG/JPG documents are passed through OCR before extraction.

### 2. Text Preprocessing

File:

- `src/text_preprocessor.py`

The extracted text is cleaned using Unicode normalization and whitespace standardization.

### 3. Semantic Extraction

File:

- `src/semantic_extractor.py`

The pretrained `deepset/minilm-uncased-squad2` model is used for extractive Question Answering.

The implementation uses multiple questions for each metadata field and a sliding context window so that longer documents can be processed.

Main QA settings:

- `max_length = 512`
- `stride = 256`

Candidate answer spans are compared using the model's start and end scores.

### 4. Metadata Normalization

The extracted answers are converted into the required formats.

- `python-dateutil` is used for date parsing.
- `word2number` is used for converting textual numbers.
- Extracted dates and numeric values are converted to the expected output format.

### 5. Prediction Pipeline

File:

- `src/predictor.py`

This module connects document loading, OCR, preprocessing, semantic extraction, and normalization to produce the final metadata.

---

## Technology Used

- **Python**
- **PyTorch**
- **Transformers**
- **python-docx**
- **Pandas**
- **Pillow**
- **Tesseract / Windows OCR**
- **FastAPI**
- **Uvicorn**
- **python-dateutil**
- **word2number**

QA Model:

```text
deepset/minilm-uncased-squad2
```

---

## Project Structure

```text
Metadata-Extraction/
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
- Windows 10/11 for Windows OCR support
- Tesseract OCR for Linux/macOS or as an alternative OCR engine

The required Python packages are listed in `requirements.txt`.

Some of the main dependencies are:

```text
torch>=2.0.0
transformers>=4.35.0
python-docx>=1.1.0
pandas>=2.0.0
pillow>=10.0.0
pytesseract>=0.3.10
winocr>=0.0.15
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
python-dateutil>=2.8.2
word2number>=1.1
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Deekshasuresh009/Metadata-Extraction.git
cd Metadata-Extraction
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

The pretrained QA model is downloaded automatically when the system is run for the first time.

---

## Running the Project

### Batch Prediction

To extract metadata from the documents in the test folder:

```bash
python main.py --input data/test --output outputs/predictions.csv
```

The predictions are saved in:

```text
outputs/predictions.csv
```

### Single Document

To process one document:

```bash
python main.py --file data/test/24158401-Rental-Agreement.png
```

### Run Evaluation

To run extraction and compare the predictions with the provided ground truth:

```bash
python main.py --input data/test --evaluate data/test.csv --output outputs/predictions.csv
```

The evaluation results are saved in:

```text
outputs/recall_results.json
```

---

## Evaluation

The assignment uses exact-match recall for each metadata field.

### Recall

```text
Recall = True Matches / (True Matches + False Matches)
```

A field is counted as a true match only when the extracted value exactly matches the expected value.

### Official Test Results

The test set contains 4 documents and 6 fields for each document, giving a total of 24 field-level evaluations.

| Target Field | True Matches | False Matches | Total | Recall |
|---|---:|---:|---:|---:|
| **Agreement Value** | 4 | 0 | 4 | **100.0%** |
| **Agreement Start Date** | 3 | 1 | 4 | **75.0%** |
| **Agreement End Date** | 1 | 3 | 4 | **25.0%** |
| **Renewal Notice (Days)** | 1 | 3 | 4 | **25.0%** |
| **Party One** | 1 | 3 | 4 | **25.0%** |
| **Party Two** | 0 | 4 | 4 | **0.0%** |
| **Overall** | **10** | **14** | **24** | **41.67%** |

Overall:

```text
10 / 24 = 41.67%
```

---

## REST API

The project also includes a FastAPI application in `api/app.py`.

### Start the API

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

### Swagger Documentation

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

This provides an interactive interface for testing the API.

### Extract Metadata Using cURL

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/24158401-Rental-Agreement.png"
```

### Example Response

```json
{
  "file_name": "example-document.png",
  "agreement_value": 12000,
  "agreement_start_date": "01.04.2008",
  "agreement_end_date": "01.03.2009",
  "renewal_notice_days": 60,
  "party_one": "Example Lessor",
  "party_two": "Example Lessee"
}
```

---

## Output Files

### `outputs/predictions.csv`

Contains the metadata extracted from the processed documents.

### `outputs/recall_results.json`

Contains the recall results for each field and the overall recall.

---

## Assignment Requirements

The implementation covers the main requirements of the metadata extraction task:

- DOCX document processing
- Scanned PNG/JPG document processing
- Extraction of six required metadata fields
- Semantic Question Answering using a pretrained Transformer model
- Support for documents with different layouts and phrasing
- No regex-based field extraction
- No static keyword-to-value extraction dictionaries
- No document-specific extraction conditions
- Structured CSV/JSON output
- Exact-match recall evaluation
- FastAPI endpoint for document extraction

---

## Repository

GitHub:

https://github.com/Deekshasuresh009/Metadata-Extraction
