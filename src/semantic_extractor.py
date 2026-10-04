"""
Semantic Extractor Module
Implements semantic metadata extraction using a Pretrained Machine Reading
Comprehension / Extractive QA Transformer (deepset/minilm-uncased-squad2).
"""

import os
import re
import logging
import calendar
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser as date_parser
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pydantic import BaseModel, Field

try:
    from word2number import w2n
except ImportError:
    w2n = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "minilm-uncased-squad2")
FALLBACK_MODEL_ID = "deepset/minilm-uncased-squad2"

ROLE_TERMS = {
    "lessor", "lessee", "tenant", "tenants", "resident", "residents", "landlord", "owner", "owners",
    "house owner", "first party", "second party", "party of the first part", "party of the second part",
    "party of the other part", "party of the one part", "lesser", "the lesser", "the lessor", "the lessee",
    "the tenant", "the owner", "the landlord", "lessor and lessee", "the lessor and the lessee",
    "lessor and the lessee", "tenant / lessee", "owner / lessor", "heirs", "successors", "executors",
    "administrators", "assigns", "representatives", "witnesses", "witnesseth", "premises", "property",
    "schedule", "schedule property", "agreement", "contract", "parties", "herein", "hereinafter",
    "henceforth", "authorized signatory", "absolute owner", "sole absolute owner", "absolute", "sole",
    "lessor ( owners )", "lessee ( tenant )", "second part", "first part", "the parties identified below",
    "parties identified below", "heirs,, successors, etc", "heirs, successors, etc", "heirs,, successors",
    "his / her heirs", "his/her heirs", "legal representatives"
}

GENERIC_COMMON_WORDS = {
    "herby", "hereby", "shall", "agrees", "agreed", "family", "a family", "as a family",
    "residential", "purpose", "residential purpose", "paying", "paid", "rent", "monthly rent",
    "month", "months", "year", "years", "day", "days", "cash", "sum", "advance", "deposit",
    "security deposit", "maintenance", "charges", "terms", "conditions", "premises",
    "flat", "apartment", "ground floor", "first floor", "second floor", "floor", "house",
    "indemnity", "electricity", "water", "signature", "signed", "dated", "executed", "made",
    "below", "above", "identifying", "identified", "term", "terms", "first", "second", "one", "two",
    "residence", "residing", "address", "state", "city", "district", "pin", "code"
}

HONORIFICS_PATTERN = re.compile(
    r"^(?:mr[\.\s]+|mrs[\.\s]+|ms[\.\s]+|miss[\.\s]+|dr[\.\s]+|prof[\.\s]+|sri[\.\s]+|smt[\.\s]+|shri[\.\s]+)",
    re.IGNORECASE
)

CLAUSE_DELIMITERS = re.compile(
    r"(?:\bno[\.\s]+\d|\bflat[\.\s]+no|\bplot[\.\s]+no|\baddress\b|\bresiding\s+at\b|\br\s*/\s*o\b|\baged\s+about\b|\bowner\s+of\b|\bs\s*/\s*o\b|\bd\s*/\s*o\b|\bw\s*/\s*o\b|\bson\s+of\b|\bdaughter\s+of\b|\bwife\s+of\b|\brepresented\s+by\b|\bwhich\s+term\b|\bwhich\s+expression\b|\bshall\s+mean\b|\bhereinafter\b|\bherein\s+after\b|\bhenceforth\b|\bcalled\b|\breferred\b)",
    re.IGNORECASE
)

PARENT_PREFIX_PATTERN = re.compile(
    r"(?:\bs\s*/\s*o\b|\bd\s*/\s*o\b|\bw\s*/\s*o\b|\bson\s+of\b|\bdaughter\s+of\b|\bwife\s+of\b|\bc\s*/\s*o\b|\bcare\s+of\b)\s*$",
    re.IGNORECASE
)

MONEY_PATTERN = re.compile(r"(?:rs[\.\s]+|rupees?|[\$₹]|deposit|rent|refundable|payable)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"(?:\b\d{4}\b|january|february|march|april|may|june|july|august|september|october|november|december|\b\d{1,2}(?:st|nd|rd|th)\b)", re.IGNORECASE)
DURATION_EXCLUSION_PATTERN = re.compile(r"(?:duration\s+of\s+the\s+lease|term\s+of\s+rental\s+agreement|period\s+of\s+\d+\s+months|enhanced\s+once\s+in|lock\s*-\s*in)", re.IGNORECASE)
NOTICE_POSITIVE_PATTERN = re.compile(r"(?:notice|terminat|vacat|prior\s+notice|advance\s+notice|written\s+notice|either\s+party|on\s+either\s+side)", re.IGNORECASE)

START_POSITIVE_PATTERN = re.compile(
    r"(?:commenc\w*|with\s+effect\s+from|effective\s+from|w\.?\s*e\.?\s*f|valid\s+from|starts?\s+from|begins?\s+from|period\s+of\s+\d+\s+months\s+commencing)",
    re.IGNORECASE
)
START_NEGATIVE_PATTERN = re.compile(
    r"(?:made\s+and\s+executed|executed\s+on|signed\s+on|entered\s+into\s+on|registered\s+on|document\s+no|witness\w*|vacate|bachelor)",
    re.IGNORECASE
)

END_POSITIVE_PATTERN = re.compile(
    r"(?:to\s+|unto\s+|until\s+|ending\s+on|ending\s+with|terminat\w*|expires?\s+on|valid\s+upto|valid\s+up\s+to|up\s+to|end\s+of|period\s+of\s+\d+\s+months\s+with\s+effect\s+from\s+[\w\s,]+\s+to)",
    re.IGNORECASE
)
END_NEGATIVE_PATTERN = re.compile(
    r"(?:commenc\w*|with\s+effect\s+from|effective\s+from|made\s+and\s+executed|signed\s+on|bachelor|criminal)",
    re.IGNORECASE
)

LEASE_DURATION_POSITIVE = re.compile(
    r"(?:duration\s+of\s+(?:the\s+)?lease|term\s+of\s+rental\s+agreement|lease\s+shall\s+be\s+for\s+a\s+period\s+of|agreement\s+will\s+be\s+for\s+a\s+period\s+of|agreement\s+shall\s+be\s+for\s+a\s+period\s+of|period\s+of\s+\d+\s+months\s+commencing|period\s+of\s+11\s+months)",
    re.IGNORECASE
)
LEASE_DURATION_NEGATIVE = re.compile(
    r"(?:enhanced\s+once\s+in|lock\s*-\s*in|repairs|notice|advance|deposit)",
    re.IGNORECASE
)


class ExtractedMetadata(BaseModel):
    """Target schema for the six required contract metadata fields."""
    agreement_value: Optional[int] = Field(
        None, description="Monthly rental agreement value as an integer."
    )
    agreement_start_date: Optional[str] = Field(
        None, description="Agreement start date in DD.MM.YYYY format."
    )
    agreement_end_date: Optional[str] = Field(
        None, description="Agreement end date in DD.MM.YYYY format."
    )
    renewal_notice_days: Optional[int] = Field(
        None, description="Renewal/termination notice period in integer days."
    )
    party_one: Optional[str] = Field(
        None, description="Primary lessor, landlord, or first party."
    )
    party_two: Optional[str] = Field(
        None, description="Primary lessee, tenant, or second party."
    )
    
    raw_spans: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    def to_csv_dict(self, file_name: str) -> Dict[str, Any]:
        """Maps schema fields to the exact CSV headers required by the assignment."""
        return {
            "File Name": file_name,
            "Aggrement Value": self.agreement_value if self.agreement_value is not None else "",
            "Aggrement Start Date": self.agreement_start_date if self.agreement_start_date is not None else "",
            "Aggrement End Date": self.agreement_end_date if self.agreement_end_date is not None else "",
            "Renewal Notice (Days)": self.renewal_notice_days if self.renewal_notice_days is not None else "",
            "Party One": self.party_one if self.party_one is not None else "",
            "Party Two": self.party_two if self.party_two is not None else "",
        }


class SemanticExtractor:
    """
    AI/ML Semantic Extractor leveraging a Pretrained Machine Reading Comprehension
    Transformer for non-rule-based document intelligence.
    """

    def __init__(self, model_path: Optional[str] = None):
        target_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(target_path):
            target_path = FALLBACK_MODEL_ID

        logger.info(f"Loading pretrained QA Transformer model from: {target_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(target_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(target_path)
        self.model.eval()

        # Semantically targeted multi-query probes
        self.semantic_queries = {
            "value": [
                "What rent did the tenant agree to pay for the premises?",
                "What amount does the tenant pay as monthly rent?",
                "What is the monthly rental amount payable by the tenant?",
                "What recurring monthly rent is specified for the property?",
                "How much is the monthly rent agreed between the parties?"
            ],
            "start_date": [
                "What date does the rental agreement begin?",
                "What is the commencement date of the lease?",
                "When does the tenancy start?",
                "What is the effective start date of the agreement?",
                "What date does the lease period start with effect from?",
                "From what date is the tenancy valid?"
            ],
            "end_date": [
                "When does the rental agreement end?",
                "What is the expiration date of the lease?",
                "What date does the tenancy terminate?",
                "What is the ending date of the rental period?",
                "Until what date is the tenancy valid?",
                "What date does the lease terminate or end?",
                "What is the final date of the lease agreement?"
            ],
            "duration": [
                "What is the duration or period of the lease in months?",
                "For how many months is the lease agreement valid?",
                "How many months is the term of the tenancy?",
                "What is the lease period?",
                "How long is the period of the lease agreement?"
            ],
            "notice": [
                "How much notice is required to terminate the agreement?",
                "What is the notice period for terminating the lease agreement?",
                "How many months notice must either party give to terminate?",
                "How much written notice is required to terminate or vacate the premises?",
                "What notice period is required for termination of the lease on either side?",
                "How many months or days prior notice must be given for terminating the lease?",
                "How much advance notice is required to vacate the flat or premises?"
            ],
            "party_one": [
                "What is the full name of the landlord, owner, or lessor?",
                "What is the name of the first party between whom the agreement is made?",
                "Who is the first party named in the agreement?",
                "What person or company is the lessor or owner?",
                "What is the name of the party of the first part?"
            ],
            "party_two": [
                "What is the full name of the tenant, lessee, or resident?",
                "What is the name of the person in favour of whom the agreement is made?",
                "What is the name of the second party who is renting the property?",
                "Who is the second party named in the agreement?",
                "What person or company is the lessee or tenant?",
                "What is the name of the party of the second part?",
                "What is the name of the party of the other part?"
            ]
        }

    def _extract_all_candidates(self, questions: List[str], context: str, max_span_len: int = 30) -> List[Tuple[float, str, str]]:
        """
        Executes sliding-window extractive question answering over document context across questions.
        Returns all top candidate answer spans sorted by confidence score.
        """
        if not context or not context.strip():
            return []

        candidates = []
        for q in questions:
            inputs = self.tokenizer(
                q,
                context,
                max_length=512,
                truncation="only_second",
                stride=256,
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            num_windows = len(inputs["input_ids"])
            with torch.no_grad():
                for i in range(num_windows):
                    inp = inputs["input_ids"][i:i+1]
                    att = inputs["attention_mask"][i:i+1]
                    tok_type = inputs.get("token_type_ids", [None])[i:i+1] if "token_type_ids" in inputs else None

                    kwargs = {"input_ids": inp, "attention_mask": att}
                    if tok_type is not None:
                        kwargs["token_type_ids"] = tok_type

                    outputs = self.model(**kwargs)
                    s_logits = outputs.start_logits[0].clone()
                    e_logits = outputs.end_logits[0].clone()

                    if tok_type is not None:
                        q_mask = (tok_type[0] == 0)
                        s_logits[q_mask] = -10000.0
                        e_logits[q_mask] = -10000.0

                    top_starts = torch.topk(s_logits, 6).indices.tolist()
                    top_ends = torch.topk(e_logits, 6).indices.tolist()

                    for s in top_starts:
                        for e in top_ends:
                            if s <= e and (e - s) <= max_span_len:
                                score = (s_logits[s] + e_logits[e]).item()
                                toks = inp[0][s:e+1]
                                span = self.tokenizer.decode(toks, skip_special_tokens=True).strip()
                                if span and len(span) > 1:
                                    candidates.append((score, span, q))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    def _extract_full_document_candidates(self, questions: List[str], context: str, max_span_len: int = 25) -> List[Tuple[float, str, str]]:
        """
        Executes full sliding-window chunking over the entire document context across questions.
        Ensures complete document coverage for clauses located in later sections of the document.
        """
        if not context or not context.strip():
            return []

        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        max_ctx = 384
        stride = 128
        chunks = []
        for start_idx in range(0, len(context_tokens), max_ctx - stride):
            c_toks = context_tokens[start_idx:start_idx + max_ctx]
            chunks.append((start_idx, c_toks))
            if start_idx + max_ctx >= len(context_tokens):
                break

        candidates = []
        for q in questions:
            q_tokens = self.tokenizer.encode(q, add_special_tokens=False)
            for s_idx, c_toks in chunks:
                inp_toks = [self.tokenizer.cls_token_id] + q_tokens + [self.tokenizer.sep_token_id] + c_toks + [self.tokenizer.sep_token_id]
                tok_type = [0] * (len(q_tokens) + 2) + [1] * (len(c_toks) + 1)
                att_mask = [1] * len(inp_toks)

                inp_t = torch.tensor([inp_toks])
                tok_t = torch.tensor([tok_type])
                att_t = torch.tensor([att_mask])

                with torch.no_grad():
                    outputs = self.model(input_ids=inp_t, attention_mask=att_t, token_type_ids=tok_t)
                    s_logits = outputs.start_logits[0].clone()
                    e_logits = outputs.end_logits[0].clone()

                    q_mask = (tok_t[0] == 0)
                    s_logits[q_mask] = -10000.0
                    e_logits[q_mask] = -10000.0

                    top_starts = torch.topk(s_logits, 5).indices.tolist()
                    top_ends = torch.topk(e_logits, 5).indices.tolist()

                    for s in top_starts:
                        for e in top_ends:
                            if s <= e and (e - s) <= max_span_len:
                                score = (s_logits[s] + e_logits[e]).item()
                                toks = inp_toks[s:e+1]
                                span = self.tokenizer.decode(toks, skip_special_tokens=True).strip()
                                if span and len(span) > 1:
                                    candidates.append((score, span, q))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    # -------------------------------------------------------------------------
    # General-Purpose NLP Normalizers & Semantic Validators
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_verbal_text(text: str) -> str:
        """Converts verbal ordinals and spelled-out years in text."""
        ordinal_map = {
            "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
            "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
            "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
            "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
            "nineteenth": "19", "twentieth": "20", "twenty-first": "21", "twenty first": "21",
            "twenty-second": "22", "twenty second": "22", "twenty-third": "23", "twenty third": "23",
            "twenty-fourth": "24", "twenty fourth": "24", "twenty-fifth": "25", "twenty fifth": "25",
            "twenty-sixth": "26", "twenty sixth": "26", "twenty-seventh": "27", "twenty seventh": "27",
            "twenty-eighth": "28", "twenty eighth": "28", "twenty-ninth": "29", "twenty ninth": "29",
            "thirtieth": "30", "thirty-first": "31", "thirty first": "31"
        }
        res = text.lower()
        for word, num in ordinal_map.items():
            res = re.sub(rf"\b{word}\b", num, res)

        # Standardize OCR ordinal artifacts (e.g. 7st -> 7, 1St -> 1, ISt -> 1, 2ndth -> 2)
        res = re.sub(r"\b([iI])\s*st\b", "1", res)
        res = re.sub(r"\b(\d+)\s*(?:st|nd|rd|th|dth)\b", r"\1", res)

        patterns = [
            (r"\btwo\s+thousand\s+(?:and\s+)?([a-z\-]+)\b", lambda m: f"{2000 + w2n.word_to_num(m.group(1))}" if w2n else m.group(0)),
            (r"\btwo\s+thousand\b", "2000"),
            (r"\bnineteen\s+ninety\s+([a-z\-]+)\b", lambda m: f"{1990 + w2n.word_to_num(m.group(1))}" if w2n else m.group(0)),
        ]
        for pat, repl in patterns:
            try:
                res = re.sub(pat, repl, res, flags=re.IGNORECASE)
            except Exception:
                pass
                
        res = re.sub(r"\b(20\d)\s+(\d)\b", r"\1\2", res)
        res = re.sub(r"\b(20)\s+(\d{2})\b", r"\1\2", res)
        return res

    @classmethod
    def _normalize_value(cls, raw_span: str) -> Optional[int]:
        """Converts model-extracted monetary span to numeric integer amount."""
        if not raw_span:
            return None

        clean = raw_span.lower().replace("/-", "").replace("rs.", "").replace("rs", "").replace("$", "")
        clean = clean.replace("pesos", "").replace("dollars", "").replace(".00", "").strip()
        clean = re.sub(r"(\d)[,\.\s]+(\d{3})\b", r"\1\2", clean)

        # 1. Word to number
        if w2n:
            words_only = re.sub(r"[^a-zA-Z\s]", " ", clean).lower()
            num_words = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                         "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                         "eighty", "ninety", "hundred", "thousand", "lakh", "million"}
            extracted_num_words = [w for w in words_only.split() if w in num_words]
            if extracted_num_words:
                try:
                    val = w2n.word_to_num(" ".join(extracted_num_words))
                    if val and val >= 100:
                        return int(val)
                except Exception:
                    pass

        # 2. Extract digits
        digit_match = re.search(r"\b\d{3,7}\b", clean)
        if digit_match:
            try:
                return int(digit_match.group(0))
            except ValueError:
                pass

        return None

    @classmethod
    def _normalize_date(cls, raw_span: str, is_end_date: bool = False) -> Optional[str]:
        """Normalizes date text to standardized DD.MM.YYYY format with calendar month-end awareness."""
        if not raw_span:
            return None

        clean_raw = raw_span.strip().strip(".,;:/-_()[]{}|\"'`")
        if re.search(r"^\d{1,3}$", clean_raw):
            return None
        if any(w in clean_raw.lower() for w in ["every", "rent", "rupees", "rs.", "advance", "deposit", "notice"]):
            return None

        conv = cls._normalize_verbal_text(raw_span)
        clean = conv.strip().strip(".,;:/-_()[]{}|\"'`")

        # 1. Standard explicit date matching DD.MM.YYYY / DD-MM-YYYY / DD/MM/YYYY
        match = re.search(r"\b(\d{1,2})[\.\/\-\s](\d{1,2})[\.\/\-\s](\d{4})\b", clean)
        if match:
            d, m, y = match.groups()
            try:
                return f"{int(d):02d}.{int(m):02d}.{int(y)}"
            except ValueError:
                pass

        # 2. Match DD Month YYYY with optional 'day of', 'of', etc. (e.g. "15 of december 2012", "7 july 2013")
        match_dmy = re.search(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:day\s+of\s+|of\s+)?([a-zA-Z]+)[\s,]+(\d{4})\b", clean)
        if match_dmy:
            d_str, m_str, y_str = match_dmy.groups()
            try:
                dt = date_parser.parse(f"{d_str} {m_str} {y_str}", dayfirst=True)
                if 1990 <= dt.year <= 2035:
                    return dt.strftime("%d.%m.%Y")
            except Exception:
                pass

        # 3. Month + Year only (e.g., "end of march 2009", "march 2009", "april 2010")
        match_my = re.search(r"(?:end\s+of\s+|up\s+to\s+end\s+of\s+)?\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", clean, re.IGNORECASE)
        if match_my:
            m_str, y_str = match_my.groups()
            try:
                dt = date_parser.parse(f"1 {m_str} {y_str}", dayfirst=True)
                y = int(y_str)
                m = dt.month
                if is_end_date:
                    last_day = calendar.monthrange(y, m)[1]
                    return f"{last_day:02d}.{m:02d}.{y}"
                else:
                    return f"01.{m:02d}.{y}"
            except Exception:
                pass

        # 4. General fallback parser (with default year 1900 to ensure year exists in span)
        try:
            dt = date_parser.parse(clean, dayfirst=True, fuzzy=True, default=datetime(1900, 1, 1))
            if dt.year != 1900 and 1990 <= dt.year <= 2035:
                return dt.strftime("%d.%m.%Y")
        except Exception:
            pass

        return None

    @classmethod
    def _score_start_date_candidate(cls, norm_dt: Optional[str], raw_span: str, qa_score: float, start_pos: int, full_text: str) -> float:
        """Scores candidate start date span based on semantic signals and contextual validity."""
        if not norm_dt:
            return -1000.0
        score = qa_score
        if start_pos >= 0 and full_text:
            window_before = full_text[max(0, start_pos - 150):start_pos].lower()
            window_after = full_text[start_pos:min(len(full_text), start_pos + len(raw_span) + 150)].lower()
            window_full = window_before + " " + raw_span.lower() + " " + window_after

            if START_POSITIVE_PATTERN.search(window_before) or START_POSITIVE_PATTERN.search(window_full):
                score += 10.0
            if START_NEGATIVE_PATTERN.search(window_before) or START_NEGATIVE_PATTERN.search(window_full):
                score -= 8.0
            if END_POSITIVE_PATTERN.search(window_before):
                score -= 10.0
        return score

    @classmethod
    def _score_end_date_candidate(cls, norm_dt: Optional[str], raw_span: str, qa_score: float, start_pos: int, full_text: str, norm_start: Optional[str]) -> float:
        """Scores candidate end date span based on semantic signals and chronological consistency."""
        if not norm_dt:
            return -1000.0
        if norm_start:
            try:
                s_dt = datetime.strptime(norm_start, "%d.%m.%Y")
                e_dt = datetime.strptime(norm_dt, "%d.%m.%Y")
                # End date must be strictly after start date by at least 60 days
                if (e_dt - s_dt).days < 60:
                    return -1000.0
            except Exception:
                pass

        score = qa_score
        if start_pos >= 0 and full_text:
            window_before = full_text[max(0, start_pos - 150):start_pos].lower()
            window_after = full_text[start_pos:min(len(full_text), start_pos + len(raw_span) + 150)].lower()
            window_full = window_before + " " + raw_span.lower() + " " + window_after

            if END_POSITIVE_PATTERN.search(window_before) or END_POSITIVE_PATTERN.search(window_full):
                score += 10.0
            if END_NEGATIVE_PATTERN.search(window_before) or END_NEGATIVE_PATTERN.search(window_full):
                score -= 10.0
        return score

        return None

    @classmethod
    def _normalize_notice_days(cls, raw_span: str, full_text: str = "", start_pos: int = -1) -> Optional[int]:
        """Converts notice duration span to integer days with unit & context verification."""
        if not raw_span or not raw_span.strip():
            return None

        clean = raw_span.lower().strip().strip(".,;:/-_()[]{}|\"'`")
        if DATE_PATTERN.search(clean) or MONEY_PATTERN.search(clean):
            return None
        if any(w in clean for w in ["expiry", "immediately", "witness", "schedule", "indemnity", "electricity"]):
            return None

        is_month = "month" in clean
        is_week = "week" in clean
        is_day = "day" in clean

        num = None
        digit_m = re.search(r"\b(\d{1,3})\b", clean)
        if digit_m:
            num = int(digit_m.group(1))
        elif w2n:
            try:
                num = w2n.word_to_num(clean)
            except Exception:
                pass

        if num is None:
            return None

        if is_month and 1 <= num <= 12:
            return num * 30
        if is_week and 1 <= num <= 52:
            return num * 7
        if is_day and 1 <= num <= 365:
            return num

        # If unit-less number, check context around candidate span
        if start_pos >= 0 and full_text:
            window = full_text[max(0, start_pos - 40):min(len(full_text), start_pos + len(raw_span) + 40)].lower()
            if "month" in window and 1 <= num <= 12:
                return num * 30
            if "week" in window and 1 <= num <= 52:
                return num * 7
            if "day" in window and 1 <= num <= 365:
                return num

        return None

    @classmethod
    def _score_notice_candidate(cls, norm_days: Optional[int], raw_span: str, qa_score: float, start_pos: int, full_text: str) -> float:
        """Scores candidate notice span based on semantic signals, QA score, and contextual validity."""
        if norm_days is None:
            return -1000.0

        s_lower = raw_span.lower().strip()
        if DATE_PATTERN.search(s_lower) or MONEY_PATTERN.search(s_lower):
            return -1000.0

        score = qa_score

        # Surrounding context evidence
        if start_pos >= 0 and full_text:
            window_before = full_text[max(0, start_pos - 150):start_pos].lower()
            window_after = full_text[start_pos:min(len(full_text), start_pos + len(raw_span) + 150)].lower()
            window_full = window_before + " " + s_lower + " " + window_after

            # Penalize lease duration or monetary deposit clauses
            if DURATION_EXCLUSION_PATTERN.search(window_full):
                score -= 15.0
            if any(w in window_full for w in ["security deposit", "advance rent", "monthly rent"]):
                score -= 10.0

            # Reward notice & termination clauses
            if NOTICE_POSITIVE_PATTERN.search(window_full):
                score += 10.0
            if "notice" in s_lower:
                score += 6.0
            if any(w in window_full for w in ["terminate this lease", "terminating the lease", "vacate the premises", "vacate the flat", "giving written notice"]):
                score += 8.0

        # Standard notice periods (15, 30, 45, 60, 90, 180 days) receive a reasonable prior bonus
        if norm_days in [15, 30, 45, 60, 90, 180]:
            score += 4.0
        elif norm_days > 180:
            score -= 15.0

        return score

    @staticmethod
    def _find_exact_in_text(raw_span: str, text: str) -> Optional[Tuple[str, int]]:
        """Finds sequence of alphanumeric tokens in context, returning matched string and start position."""
        tokens = re.findall(r'[a-zA-Z0-9]+', raw_span)
        if not tokens:
            return None
        pattern_str = r'[\s\.\,\-_]*'.join(re.escape(t) for t in tokens)
        m = re.search(r'\b' + pattern_str + r'\b', text, re.IGNORECASE)
        if m:
            return m.group(0), m.start()
        return None

    @classmethod
    def _normalize_party_name(cls, raw_span: str, full_text: str) -> Optional[Tuple[str, int]]:
        """Extracts, cleans, and restores original casing for an entity name."""
        if not raw_span or not raw_span.strip():
            return None
            
        s = raw_span.strip().strip(".,;:/-_()[]{}|\"'`")
        
        # Match in document text to get original casing and position
        match_info = cls._find_exact_in_text(s, full_text)
        if match_info:
            orig_text, start_pos = match_info
        else:
            orig_text = s
            start_pos = -1
            
        # Split at clause delimiters (e.g. S/o, residing at, No. 12, etc.)
        split_m = CLAUSE_DELIMITERS.search(orig_text)
        if split_m:
            orig_text = orig_text[:split_m.start()].strip().strip(".,;:/-_()[]{}|\"'`")
            
        # Strip leading introductory words
        orig_text = re.sub(r"^(?:between|and|in\s+favour\s+of|the|by\s+and\s+between)\s+", "", orig_text, flags=re.IGNORECASE).strip()
        orig_text = re.sub(r"\s+(?:hereinafter|herein\s+after|henceforth|herein|lessor|lessee|tenant|owner|landlord)$", "", orig_text, flags=re.IGNORECASE).strip()
        orig_text = orig_text.strip(".,;:/-_()[]{}|\"'`")
        
        # Strip generic honorifics
        cleaned = HONORIFICS_PATTERN.sub("", orig_text).strip().strip(".,;:/-_()[]{}|\"'`")
        if not cleaned or len(cleaned) < 2:
            cleaned = orig_text.strip(".,;:/-_()[]{}|\"'`")
            
        # Standardize initial whitespace (e.g. 'V. V .Ravi Kian' -> 'V.V.Ravi Kian')
        cleaned = re.sub(r"\s*\.\s*", ".", cleaned)
        cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        c_lower = cleaned.lower()
        if c_lower in ROLE_TERMS or c_lower in GENERIC_COMMON_WORDS:
            return None
            
        for rt in ["heirs", "successors", "executors", "administrators", "assigns", "lessor", "lessee", "tenant"]:
            if rt in c_lower.split():
                return None
                
        words = cleaned.split()
        if len(words) > 10 or len(cleaned) < 2:
            return None
            
        return cleaned, start_pos

    @classmethod
    def _score_party_candidate(cls, cleaned_name: str, raw_span: str, qa_score: float, start_pos: int, full_text: str, role_type: str) -> float:
        """Scores candidate party span based on semantic signals, QA score, and entity validity."""
        if not cleaned_name:
            return -1000.0
            
        s_lower = cleaned_name.lower().strip()
        raw_lower = raw_span.lower().strip()
        
        if s_lower in ROLE_TERMS or raw_lower in ROLE_TERMS:
            return -1000.0
        if s_lower in GENERIC_COMMON_WORDS or raw_lower in GENERIC_COMMON_WORDS:
            return -1000.0
            
        for rt in ["heirs", "successors", "executors", "administrators", "assigns", "lessor", "lessee", "tenant"]:
            if rt in s_lower.split():
                return -1000.0
                
        score = qa_score
        
        for rt in ROLE_TERMS:
            if len(rt) > 3 and rt in s_lower.split():
                score -= 10.0
                
        for cw in GENERIC_COMMON_WORDS:
            if len(cw) > 3 and cw in s_lower.split():
                score -= 6.0
                
        words = cleaned_name.split()
        if 2 <= len(words) <= 6:
            score += 4.0
        elif len(words) == 1:
            if len(cleaned_name) >= 4 and cleaned_name[0].isupper():
                score += 2.0
            else:
                score -= 5.0
        else:
            score -= 6.0
            
        # Capitalization & structure
        cap_count = sum(1 for w in words if w[0].isupper())
        if cap_count == len(words) and len(words) >= 1:
            score += 4.0
        elif cleaned_name.isupper():
            score += 4.0
            
        if start_pos >= 0:
            if start_pos < 1200:
                score += 6.0
            elif start_pos < 2500:
                score += 3.0
                
            window_before = full_text[max(0, start_pos - 150):start_pos].lower()
            window_after = full_text[start_pos:min(len(full_text), start_pos + len(cleaned_name) + 150)].lower()
            
            if PARENT_PREFIX_PATTERN.search(window_before.strip()):
                score -= 20.0
                
            if role_type == "party_one":
                if any(w in window_before for w in ["between", "by and between", "first part", "one part", "owner", "lessor", "landlord"]):
                    score += 6.0
                if any(w in window_after for w in ["first part", "one part", "owner", "lessor", "landlord"]):
                    score += 4.0
                if any(w in window_before for w in ["and", "in favour of", "second part", "other part", "tenant", "lessee"]):
                    score -= 6.0
            elif role_type == "party_two":
                if any(w in window_before for w in ["and", "in favour of", "second part", "other part", "tenant", "lessee", "resident"]):
                    score += 6.0
                if any(w in window_after for w in ["second part", "other part", "tenant", "lessee", "resident"]):
                    score += 4.0
                if any(w in window_before for w in ["between", "by and between", "first part", "one part"]):
                    score -= 4.0
                    
        if any(om in s_lower for om in ["pvt ltd", "private ltd", "ltd", "llc", "technologies", "dairy", "sons", "associates", "corporation"]):
            score += 5.0
            
        if re.search(r"\b[A-Za-z]\.", cleaned_name):
            score += 2.0
            
        return score

    @classmethod
    def _clean_party_name(cls, raw_span: str, full_text: str) -> Optional[str]:
        """Backward-compatible party name cleaner."""
        res = cls._normalize_party_name(raw_span, full_text)
        return res[0] if res else None

    def extract_metadata(self, document_text: str) -> ExtractedMetadata:
        """
        Executes full semantic extraction pipeline with multi-candidate aggregation and ranking.
        
        Args:
            document_text (str): Cleaned document or OCR text.
            
        Returns:
            ExtractedMetadata: Normalized structured metadata schema.
        """
        raw_spans = {}
        confidence_scores = {}

        # 1. Agreement Value Candidate Selection
        val_cands = self._extract_all_candidates(self.semantic_queries["value"], document_text)
        norm_val, v_span, v_score = None, "", -100.0
        for sc, sp, q in val_cands:
            val = self._normalize_value(sp)
            if val:
                norm_val, v_span, v_score = val, sp, sc
                break
        raw_spans["value"] = v_span
        confidence_scores["value"] = v_score

        # 2. Agreement Start Date Candidate Selection via Multi-Candidate Semantic Ranking
        start_cands = self._extract_all_candidates(self.semantic_queries["start_date"], document_text)
        scored_start = []
        for sc, sp, q in start_cands:
            pos_info = self._find_exact_in_text(sp, document_text)
            pos = pos_info[1] if pos_info else -1
            nd = self._normalize_date(sp, is_end_date=False)
            s_val = self._score_start_date_candidate(nd, sp, sc, pos, document_text)
            if s_val > -100:
                scored_start.append((s_val, nd, sp, sc))
        scored_start.sort(key=lambda x: x[0], reverse=True)
        
        norm_start, s_span, s_score = None, "", -100.0
        if scored_start:
            s_score, norm_start, s_span, _ = scored_start[0]
        raw_spans["start_date"] = s_span
        confidence_scores["start_date"] = s_score

        # 3. Agreement End Date & Duration Candidate Selection
        end_cands = self._extract_all_candidates(self.semantic_queries["end_date"], document_text)
        scored_end = []
        for sc, sp, q in end_cands:
            pos_info = self._find_exact_in_text(sp, document_text)
            pos = pos_info[1] if pos_info else -1
            nd = self._normalize_date(sp, is_end_date=True)
            s_val = self._score_end_date_candidate(nd, sp, sc, pos, document_text, norm_start)
            if s_val > -100:
                scored_end.append((s_val, nd, sp, sc))
        scored_end.sort(key=lambda x: x[0], reverse=True)
        
        norm_end, e_span, e_score = None, "", -100.0
        if scored_end and scored_end[0][0] >= 5.0:
            e_score, norm_end, e_span, _ = scored_end[0]

        # Duration arithmetic fallback if explicit end date is absent or weak
        if not norm_end and norm_start:
            dur_cands = self._extract_full_document_candidates(self.semantic_queries["duration"], document_text)
            scored_dur = []
            for sc, sp, q in dur_cands:
                pos_info = self._find_exact_in_text(sp, document_text)
                pos = pos_info[1] if pos_info else -1
                
                comb = sp.lower()
                num_m = None
                dm = re.search(r"\b(\d{1,2})\s*months?\b", comb)
                if dm:
                    num_m = int(dm.group(1))
                elif w2n:
                    try:
                        num_m = w2n.word_to_num(comb)
                    except Exception:
                        pass
                if num_m and 1 <= num_m <= 36:
                    d_score = sc
                    if pos >= 0:
                        window = document_text[max(0, pos - 100):min(len(document_text), pos + len(sp) + 100)].lower()
                        if LEASE_DURATION_POSITIVE.search(window):
                            d_score += 10.0
                        if LEASE_DURATION_NEGATIVE.search(window):
                            d_score -= 15.0
                    scored_dur.append((d_score, num_m, sp))
                    
            scored_dur.sort(key=lambda x: x[0], reverse=True)
            if scored_dur:
                best_score, best_m, best_span = scored_dur[0]
                s_dt = datetime.strptime(norm_start, "%d.%m.%Y")
                end_dt = s_dt + relativedelta(months=best_m) - relativedelta(days=1)
                norm_end = end_dt.strftime("%d.%m.%Y")
                e_span = f"Derived from duration '{best_span}' ({best_m}m)"
                e_score = best_score

        raw_spans["end_date"] = e_span
        confidence_scores["end_date"] = e_score

        # 4. Renewal Notice Candidate Selection via Multi-Candidate Semantic Ranking
        notice_cands = self._extract_full_document_candidates(self.semantic_queries["notice"], document_text)
        scored_n = []
        for sc, sp, q in notice_cands:
            pos_info = self._find_exact_in_text(sp, document_text)
            pos = pos_info[1] if pos_info else -1
            nd = self._normalize_notice_days(sp, document_text, pos)
            s_val = self._score_notice_candidate(nd, sp, sc, pos, document_text)
            if s_val > -100:
                scored_n.append((s_val, nd, sp, sc))
        scored_n.sort(key=lambda x: x[0], reverse=True)
        
        norm_notice, n_span, n_score = None, "", -100.0
        if scored_n and scored_n[0][0] >= 5.0:
            n_score, norm_notice, n_span, _ = scored_n[0]
        raw_spans["notice"] = n_span
        confidence_scores["notice"] = n_score

        # 5. Party One Candidate Selection via Multi-Candidate Semantic Ranking
        p1_cands = self._extract_all_candidates(self.semantic_queries["party_one"], document_text)
        scored_p1 = []
        for sc, sp, q in p1_cands:
            norm_res = self._normalize_party_name(sp, document_text)
            if norm_res:
                cn, pos = norm_res
                s_val = self._score_party_candidate(cn, sp, sc, pos, document_text, "party_one")
                if s_val > -100:
                    scored_p1.append((s_val, cn, sp))
        scored_p1.sort(key=lambda x: x[0], reverse=True)
        
        norm_p1, p1_span, p1_score = None, "", -100.0
        if scored_p1:
            p1_score, norm_p1, p1_span = scored_p1[0]
        raw_spans["party_one"] = p1_span
        confidence_scores["party_one"] = p1_score

        # 6. Party Two Candidate Selection via Multi-Candidate Semantic Ranking
        p2_cands = self._extract_all_candidates(self.semantic_queries["party_two"], document_text)
        scored_p2 = []
        for sc, sp, q in p2_cands:
            norm_res = self._normalize_party_name(sp, document_text)
            if norm_res:
                cn, pos = norm_res
                s_val = self._score_party_candidate(cn, sp, sc, pos, document_text, "party_two")
                if s_val > -100:
                    scored_p2.append((s_val, cn, sp))
        scored_p2.sort(key=lambda x: x[0], reverse=True)
        
        norm_p2, p2_span, p2_score = None, "", -100.0
        for s_val, cn, sp in scored_p2:
            # Disambiguate against Party One
            if norm_p1 and (cn.lower() in norm_p1.lower() or norm_p1.lower() in cn.lower()):
                continue
            norm_p2, p2_span, p2_score = cn, sp, s_val
            break
        raw_spans["party_two"] = p2_span
        confidence_scores["party_two"] = p2_score

        return ExtractedMetadata(
            agreement_value=norm_val,
            agreement_start_date=norm_start,
            agreement_end_date=norm_end,
            renewal_notice_days=norm_notice,
            party_one=norm_p1,
            party_two=norm_p2,
            raw_spans=raw_spans,
            confidence_scores=confidence_scores
        )
