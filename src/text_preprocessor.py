"""
Text Preprocessor Module
Applies general cleaning and normalization to raw document text.
Strictly non-rule-based: performs generic text hygiene without metadata extraction.
"""

import re
import unicodedata

class TextPreprocessor:
    """Generic text hygiene and normalization processor."""

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Standardizes Unicode representations (NFKC normalization)."""
        if not text:
            return ""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """
        Normalizes excessive whitespace, carriage returns, and blank lines
        while preserving paragraph structure.
        """
        if not text:
            return ""
        # Standardize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace non-breaking spaces and tabs with standard space
        text = text.replace("\xa0", " ").replace("\t", " ")
        # Collapse multiple inline spaces to a single space
        text = re.sub(r"[ ]{2,}", " ", text)
        # Collapse three or more consecutive newlines to two
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def preprocess(cls, raw_text: str) -> str:
        """
        Performs full end-to-end text hygiene on raw document or OCR text.
        
        Args:
            raw_text (str): Unprocessed extracted text.
            
        Returns:
            str: Cleaned and normalized text ready for NLP processing.
        """
        if not raw_text:
            return ""
        text = cls.normalize_unicode(raw_text)
        text = cls.clean_whitespace(text)
        return text
