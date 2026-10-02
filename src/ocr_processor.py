"""
OCR Processor Module
Extracts raw text from image files (e.g., PNG) using OCR engines.
Strictly non-rule-based: extracts text only, does not perform metadata extraction.
"""

import os
import logging
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Applies image preprocessing (contrast enhancement, sharpening, resolution scaling)
    to optimize OCR text recognition accuracy.
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
        
    w, h = img.size
    if w < 1600:
        scale_factor = 1600.0 / w
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    enhancer_contrast = ImageEnhance.Contrast(img)
    img = enhancer_contrast.enhance(1.4)
    
    enhancer_sharpness = ImageEnhance.Sharpness(img)
    img = enhancer_sharpness.enhance(1.3)
    
    return img

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from a scanned image (PNG/JPG) using an available OCR engine.
    Supports Windows Media OCR (winocr) and Tesseract (pytesseract).
    
    Args:
        image_path (str): Absolute or relative path to the image file.
        
    Returns:
        str: Raw extracted OCR text.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    raw_img = Image.open(image_path)
    img = preprocess_image_for_ocr(raw_img)

    # Method 1: Windows Native Media OCR (winocr) - high accuracy on Windows
    try:
        import asyncio
        import winocr
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    result = loop.run_until_complete(winocr.recognize_pil(img, lang="en"))
                except ImportError:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: asyncio.run(winocr.recognize_pil(img, lang="en")))
                        result = future.result()
            else:
                result = loop.run_until_complete(winocr.recognize_pil(img, lang="en"))
        except RuntimeError:
            result = asyncio.run(winocr.recognize_pil(img, lang="en"))
            
        if hasattr(result, "text") and result.text:
            return result.text
        elif isinstance(result, dict) and "text" in result:
            return result["text"]
    except Exception as e:
        logger.warning(f"winocr engine failed or unavailable: {e}. Falling back to pytesseract.")

    # Method 2: Fallback to pytesseract
    try:
        import pytesseract
        text = pytesseract.image_to_string(img)
        if text and text.strip():
            return text
    except Exception as e:
        logger.warning(f"pytesseract failed or unavailable: {e}.")

    # Method 3: Fallback to EasyOCR if installed
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(image_path, detail=0)
        return "\n".join(results)
    except Exception as e:
        logger.warning(f"EasyOCR failed or unavailable: {e}.")

    raise RuntimeError(
        f"Unable to perform OCR on {image_path}. Please ensure winocr, pytesseract, or easyocr is installed."
    )
