"""
Predictor Module
Coordinates the end-to-end document metadata extraction pipeline:
File -> DocumentLoader -> OCR (if PNG) -> TextPreprocessor -> SemanticExtractor -> Metadata Dict
"""

import os
from typing import Dict, Any
from src.document_loader import DocumentLoader, Document
from src.text_preprocessor import TextPreprocessor
from src.semantic_extractor import SemanticExtractor, ExtractedMetadata

class MetadataPredictor:
    """Coordinates the document metadata extraction workflow."""

    def __init__(self, extractor: SemanticExtractor = None):
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.extractor = extractor or SemanticExtractor()

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Executes the full pipeline for a single document:
        File -> Document Loader -> OCR (if PNG) -> Preprocessor -> Semantic Extractor -> Metadata.
        
        Args:
            file_path (str): Path to input document (.docx or .png).
            
        Returns:
            dict: Structured metadata matching target CSV schema.
        """
        # 1. Ingest document & perform OCR if image
        doc: Document = self.loader.load_document(file_path)
        
        # 2. Text normalization & hygiene
        cleaned_text = self.preprocessor.preprocess(doc.extracted_text)
        
        # 3. AI/NLP Semantic metadata extraction
        extracted_metadata: ExtractedMetadata = self.extractor.extract_metadata(cleaned_text)
        
        # 4. Format according to assignment CSV specification
        res = extracted_metadata.to_csv_dict(file_name=doc.file_name)
        res["_raw_spans"] = extracted_metadata.raw_spans
        res["_confidence_scores"] = extracted_metadata.confidence_scores
        return res
