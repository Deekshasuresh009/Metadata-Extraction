"""
Semantic Extractor Module
Implements semantic metadata extraction using a Pretrained Machine Reading
Comprehension / Extractive QA Transformer (deepset/minilm-uncased-squad2).
"""

import os
import re
import logging
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

ROLE_WORDS = {
    "lessor", "lessee", "tenant", "tenants", "resident", "resident ( s )", "resident(s)",
    "owner", "owners", "the owner", "the lessor", "the lessee", "the tenant",
    "house owner", "house owner and tenant", "landlord", "first party", "second party",
    "party of the first part", "party of the second part", "lessees", "the absolute owner",
    "absolute owner", "owner ( s )", "the party of the first part", "the party of the second part",
    "the landlord", "lessor ( owners )", "lessee ( tenant )", "lessee (tenant)",
    "the owner ( s )", "the resident ( s )", "lessor (owner)", "party of the first part and between",
    "the lesser", "lessor and lessee", "the lessor and the lessee", "family"
}

CLAUSE_VERBS = [
    r"\bshall\b", r"\bagrees?\b", r"\bgrants?\b", r"\bis\s+the\b", r"\buse\s+and\s+occupy\b",
    r"\bpurpose\b", r"\bpaying\b", r"\breceived\b", r"\bdeposited\b", r"\bhereinafter\b",
    r"\bwitnesseth\b", r"\bhereby\b", r"\bto\s+use\b", r"\bconsideration\b", r"\bpremises\b",
    r"\bproprietor\b", r"\bground\s+floor\b", r"\bapartment\b", r"\bflat\b", r"\brent\b"
]


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
                "On what date was this agreement made and executed?"
            ],
            "end_date": [
                "When does the rental agreement end?",
                "What is the expiration date of the lease?",
                "What date does the tenancy terminate?",
                "What is the ending date of the rental period?",
                "Until what date is the tenancy valid?"
            ],
            "duration": [
                "What is the duration or period of the lease in months?",
                "For how many months is the lease agreement valid?",
                "How many months is the term of the tenancy?",
                "What is the lease period?"
            ],
            "notice": [
                "How much advance notice is required to terminate the agreement?",
                "How many days notice is required to vacate the premises?",
                "What notice period is required for termination of the lease?",
                "What notice period is specified for ending the agreement?",
                "How many days or months written notice must either party give to terminate?",
                "How many months or days prior notice must be given?"
            ],
            "party_one": [
                "Who is the first party named in the agreement?",
                "What is the full name of the landlord, owner, or lessor?",
                "What is the name of the party of the first part who owns the property?",
                "What person or organization is letting out the premises?",
                "Who is the first person or party named between whom the agreement is made?",
                "What is the full name of the lessor?"
            ],
            "party_two": [
                "Who is the second party named in the agreement?",
                "What is the full name of the tenant, lessee, or resident?",
                "What is the name of the party of the second part who is renting the property?",
                "What person or organization is renting or occupying the premises?",
                "Who is the second person or party named between whom the agreement is made?",
                "What is the full name of the lessee?"
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
    def _normalize_date(cls, raw_span: str) -> Optional[str]:
        """Normalizes date text to standardized DD.MM.YYYY format via python-dateutil."""
        if not raw_span:
            return None

        conv = cls._normalize_verbal_text(raw_span)
        clean = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", conv).strip()

        # Standard explicit date matching DD.MM.YYYY / DD-MM-YYYY / DD/MM/YYYY
        match = re.search(r"\b(\d{1,2})[\.\/\-\s](\d{1,2})[\.\/\-\s](\d{4})\b", clean)
        if match:
            d, m, y = match.groups()
            try:
                return f"{int(d):02d}.{int(m):02d}.{int(y)}"
            except ValueError:
                pass

        # General dateutil parser (with default year 1900 to ensure year exists in span)
        try:
            dt = date_parser.parse(clean, dayfirst=True, fuzzy=True, default=datetime(1900, 1, 1))
            if dt.year != 1900 and 1990 <= dt.year <= 2035:
                return dt.strftime("%d.%m.%Y")
        except Exception:
            pass

        return None

    @classmethod
    def _normalize_notice_days(cls, raw_span: str) -> Optional[int]:
        """Converts notice duration span to integer days."""
        if not raw_span:
            return None

        clean = raw_span.lower().strip()
        # Reject dates
        if re.search(r"\b(19\d\d|20\d\d)\b", clean):
            return None
        if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", clean):
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

        if num is not None and 1 <= num <= 365:
            if is_month:
                return num * 30
            elif is_week:
                return num * 7
            else:
                return num

        return None

    @classmethod
    def _clean_party_name(cls, raw_span: str, full_text: str) -> Optional[str]:
        """Cleans and validates party entity names against boilerplate clause text."""
        if not raw_span:
            return None

        s = raw_span.strip().strip(".,;:/-_")
        if s.lower() in ROLE_WORDS:
            return None

        for cv in CLAUSE_VERBS:
            if re.search(cv, s, flags=re.IGNORECASE):
                return None

        s = re.sub(r"\b(hereinafter|herein\s+after|henceforth|herein)\s+called.*$", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\b(hereinafter|herein\s+after|henceforth|herein)\s+referred.*$", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\b(residing\s+at|r/o|r\s*/\s*o).*$", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^the\s+owner\s+of\s+the\s+premises\s+at.*$", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^lessor\s+hereby\s+.*$", "", s, flags=re.IGNORECASE).strip()
        s = s.strip(".,;:/-_")

        words = s.split()
        if not (1 <= len(words) <= 12):
            return None
        if not re.search(r"[a-zA-Z]", s) or len(s) < 3:
            return None

        pattern = re.compile(re.escape(s), re.IGNORECASE)
        match = pattern.search(full_text)
        if match:
            return match.group(0).strip()

        return s.title() if s.islower() else s

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

        # 2. Agreement Start Date Candidate Selection
        start_cands = self._extract_all_candidates(self.semantic_queries["start_date"], document_text)
        norm_start, s_span, s_score = None, "", -100.0
        for sc, sp, q in start_cands:
            dt = self._normalize_date(sp)
            if dt:
                norm_start, s_span, s_score = dt, sp, sc
                break
        raw_spans["start_date"] = s_span
        confidence_scores["start_date"] = s_score

        # 3. Agreement End Date & Duration Candidate Selection
        end_cands = self._extract_all_candidates(self.semantic_queries["end_date"], document_text)
        norm_end, e_span, e_score = None, "", -100.0
        for sc, sp, q in end_cands:
            dt = self._normalize_date(sp)
            if dt:
                # Validate that explicit end date is strictly after start date
                if norm_start:
                    try:
                        s_dt = datetime.strptime(norm_start, "%d.%m.%Y")
                        e_dt = datetime.strptime(dt, "%d.%m.%Y")
                        if e_dt <= s_dt:
                            continue
                    except Exception:
                        pass
                norm_end, e_span, e_score = dt, sp, sc
                break

        # Fallback to duration arithmetic if explicit end date is not specified
        if not norm_end and norm_start:
            dur_cands = self._extract_all_candidates(self.semantic_queries["duration"], document_text)
            for sc, sp, q in dur_cands:
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
                    s_dt = datetime.strptime(norm_start, "%d.%m.%Y")
                    end_dt = s_dt + relativedelta(months=num_m) - relativedelta(days=1)
                    norm_end = end_dt.strftime("%d.%m.%Y")
                    e_span = f"Derived from duration '{sp}' ({num_m}m)"
                    e_score = sc
                    break
        raw_spans["end_date"] = e_span
        confidence_scores["end_date"] = e_score

        # 4. Renewal Notice Candidate Selection
        notice_cands = self._extract_all_candidates(self.semantic_queries["notice"], document_text)
        norm_notice, n_span, n_score = None, "", -100.0
        for sc, sp, q in notice_cands:
            days = self._normalize_notice_days(sp)
            if days:
                norm_notice, n_span, n_score = days, sp, sc
                break
        raw_spans["notice"] = n_span
        confidence_scores["notice"] = n_score

        # 5. Party One & Party Two Candidate Selection
        p1_cands = self._extract_all_candidates(self.semantic_queries["party_one"], document_text)
        norm_p1, p1_span, p1_score = None, "", -100.0
        for sc, sp, q in p1_cands:
            cleaned = self._clean_party_name(sp, document_text)
            if cleaned:
                norm_p1, p1_span, p1_score = cleaned, sp, sc
                break
        raw_spans["party_one"] = p1_span
        confidence_scores["party_one"] = p1_score

        p2_cands = self._extract_all_candidates(self.semantic_queries["party_two"], document_text)
        norm_p2, p2_span, p2_score = None, "", -100.0
        for sc, sp, q in p2_cands:
            cleaned = self._clean_party_name(sp, document_text)
            if cleaned:
                # Disambiguate against Party One
                if norm_p1 is None or (cleaned.lower() not in norm_p1.lower() and norm_p1.lower() not in cleaned.lower()):
                    norm_p2, p2_span, p2_score = cleaned, sp, sc
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
