"""
chunking.py — PDF extraction + chunking pipeline
"""

from __future__ import annotations

import os
import re
import html
import ftfy
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tiktoken
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter

# ──────────────────────────────────────────────────────────────────────────────
# Text Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    try:
        text = text.encode("latin1").decode("utf-8")
    except Exception:
        pass
    text = ftfy.fix_text(text)
    replacements = {"\uf0b7": "•", "\xa0": " ", "•  ": "• ", "  ": " "}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^[ \t]*[•▪◦]\s*", "• ", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_chunk_id(pdf_name: str, idx: int) -> str:
    pdf_stem = Path(pdf_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9]+", "_", pdf_stem).strip("_")
    return f"{safe_stem}_{idx:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# Domain Knowledge: Brand Aliases & Patterns
# ──────────────────────────────────────────────────────────────────────────────

BRAND_ALIASES: Dict[str, List[str]] = {
    "TREMFYA":   ["tremfya", "guselkumab"],
    "STELARA":   ["stelara", "ustekinumab"],
    "SKYRIZI":   ["skyrizi", "risankizumab"],
    "COSENTYX":  ["cosentyx", "secukinumab"],
    "TALTZ":     ["taltz", "ixekizumab"],
    "SILIQ":     ["siliq", "brodalumab"],
    "ENBREL":    ["enbrel", "etanercept"],
    "HUMIRA":    ["humira", "adalimumab"],
    "AMJEVITA":  ["amjevita"],
    "OTEZLA":    ["otezla", "apremilast"],
    "BIMZELX":   ["bimzelx", "bimekizumab"],
    "ILUMYA":    ["ilumya", "tildrakizumab"],
    "REMICADE":  ["remicade", "infliximab"],
    "YESINTEK":  ["yesintek"],
    "CIMZIA":    ["cimzia", "certolizumab", "Certolizumab pegol"],
    "INFLECTRA": ["inflectra", "infliximab-dyyb"],
    "OTULFI":    ["otulfi", "ustekinumab-aauz"],
    "ACITRETIN": ["acitretin", "soriatane"],
}

PARAM_PATTERNS: Dict[str, List[str]] = {
    "Age": [
        r"\b(?:member|patient|individual)\s+is\s*(?:≥|≤|>=|<=|>|<)?\s*\d+\s*(?:years?|yrs?)\s+of\s+age\b",
        r"\b\d+\s*(?:years?|yrs?)\s+of\s+age\s+and\s+older\b",
        r"\b\d+\s*(?:years?|yrs?)\s+and\s+older\b",
        r"\bage\s*(?:≥|≤|>=|<=|>|<)\s*\d+\b",
        r"\b(?:adult|pediatric|adolescent|children)\s+(?:member|members|patient|patients)\b",
        r"(?:&gt;=|&lt;=|&gt;|&lt;)\s*\d+\s*(?:years?|yrs?)\b",
    ],
    "Step Therapy Requirements Documented in Policy": [
        r"\bstep\s*therapy\b", r"\bfail[- ]?first\b", r"\btrial(?:ed)?\b",
        r"\binadequate\s+response\b", r"\bintoler(?:ant|ance)\b",
        r"\bcontraindicat(?:ion|ed)\b", r"\bunable\s+to\s+take\b",
        r"\bclinical\s+reason\s+to\s+avoid\b",
        r"\bpreviously\s+received\s+a\s+biologic\b",
        r"\bpreviously\s+received\s+a\s+targeted\s+synthetic\s+drug\b",
        r"\bdocumentation\s+(?:is\s+)?required\s+for\s+approval\b",
        r"\bpharmacologic\s+treatment\s+with\b",
    ],
    "Number of Steps through Brands": [
        r"\bbiologic\b", r"\btargeted\s+synthetic\s+drug\b", r"\btargeted\b",
        r"\b(?:tnf|il[- ]?17|il[- ]?23|jak)\b",
        r"\bpreviously\s+received\s+a\s+biologic\b",
    ],
    "Number of Steps through Generic": [
        r"\bnon[- ]biologic\b", r"\bconventional\b", r"\btopical\b",
        r"\bpharmacologic\s+treatment\s+with\b",
        r"\b(?:methotrexate|cyclosporine|acitretin)\b",
    ],
    "Step through-Phototherapy": [
        r"\bphototherap(?:y|ies)\b", r"\bpuva\b", r"\buvb\b", r"\blight\s+therapy\b",
    ],
    "TB Test required": [
        r"\btuberculosis\b", r"\blatent\s+tb\b",
        r"\btb\s*(?:test|screen|screening)\b",
        r"\bnegative\s+tuberculosis\s+\(tb\)\s+test\b",
        r"\bppd\b", r"\btst\b", r"\bigra\b", r"\bquantiferon\b",
        r"\bwithin\s+\d+\s+months?\s+of\s+initiating\s+therapy\b",
    ],
    "Quantity Limits": [
        r"\bquantity\s+level\s+limit\b", r"\bquantity\s+limit(?:s)?\b",
        r"\bexception\s+limit\b", r"\b\d+\s+vials?\b",
        r"\b\d+\s+(?:syringes?|pens?)\b", r"\bper\s+\d+\s+days?\b",
        r"\bprefilled\s+syringe\b", r"\bql\b",
    ],
    "Specialist Types": [
        r"\bprescribed\s+by\s+or\s+in\s+consultation\s+with\b",
        r"\bdermatolog(?:ist|y)\b", r"\brheumatolog(?:ist|y)\b",
        r"\bgastroenterolog(?:ist|y)\b", r"\bspecialist\b",
        r"\bprescriber\s+restrictions?\b",
    ],
    "Initial Authorization Duration(in-months)": [
        r"\binitial\s*:\s*\d+\s*months?\b",
        r"\binitial\b.*\b(?:authorization|approval)\b",
        r"\bauthorization\s+of\s+\d+\s*months?\s+may\s+be\s+granted\b",
        r"\bapproved?\s+for\s+\d+\s*(?:months?|mos?)\b",
        r"\bcoverage\s+duration\b.*\binitial\b",
    ],
    "Reauthorization Duration(in-months)": [
        r"\brenewal\s*:\s*\d+\s*months?\b",
        r"\breauthori[sz]ation\b.*\b(?:duration|period)\b",
        r"\brenewal\b.*\b(?:duration|period)\b",
        r"\breauthori[sz]ation\s+of\s+\d+\s*months?\b",
        r"\brenewal\s+of\s+\d+\s*months?\b",
        r"\bcoverage\s+duration\b.*\brenewal\b",
    ],
    "Reauthorization Required": [
        r"\breauthori[sz]ation\b", r"\brenewal\b",
        r"\bcontinuation\s+requests?\b", r"\brenewal\s+criteria\b",
        r"\bcontinued\s+therapy\b",
    ],
    "Reauthorization Requirements Documented in Policy": [
        r"\brenewal\s+criteria\b",
        r"\bcriteria\s+for\s+continuation\s+of\s+therapy\b",
        r"\breauthori[sz]ation\b.*\bcriteria\b",
        r"\bcontinued\b.*\bbenefit\b", r"\bclinical\s+response\b",
        r"\bpositive\s+clinical\s+response\b",
        r"\breduction\s+in\s+(?:the\s+)?body\s+surface\s+area\b",
        r"\bimprovement\s+in\s+symptoms?\b", r"\bfrom\s+baseline\b",
        r"\breauthorization\s+request\b",
    ],
}

COMPILED_PARAM_PATTERNS = {
    k: [re.compile(p, re.I) for p in v] for k, v in PARAM_PATTERNS.items()
}

PSO_PATTERNS = [
    r"\bpsoriasis\s*\(pso\)\b", r"\bpso\b", r"\bplaque psoriasis\b",
    r"\bmoderate to severe plaque psoriasis\b", r"\bpsoriasis\b",
]
COMPILED_PSO_PATTERNS = [re.compile(p, re.I) for p in PSO_PATTERNS]


def detect_brands(text: str) -> List[str]:
    t = normalize_text(text).lower()
    found = []
    for brand, aliases in BRAND_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", t):
                found.append(brand)
                break
    return sorted(set(found))


def detect_policy_params(text: str) -> List[str]:
    text = normalize_text(text)
    matched = []
    for pname, patterns in COMPILED_PARAM_PATTERNS.items():
        for rgx in patterns:
            if rgx.search(text):
                matched.append(pname)
                break
    return sorted(set(matched))


def detect_pso_indication(text: str) -> str:
    text = normalize_text(text)
    return "Yes" if any(rgx.search(text) for rgx in COMPILED_PSO_PATTERNS) else "No"


# ──────────────────────────────────────────────────────────────────────────────
# PDF → Markdown Extraction
# ──────────────────────────────────────────────────────────────────────────────

def clean_table_cell(x) -> str:
    if x is None:
        return ""
    return normalize_text(str(x)).replace("\n", " ").strip()


def table_to_markdown(table) -> str:
    if not table:
        return ""
    rows = [[clean_table_cell(c) for c in row] for row in table]
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    header = rows[0]
    body = rows[1:]
    md = ["| " + " | ".join(header) + " |",
          "| " + " | ".join(["---"] * max_cols) + " |"]
    for row in body:
        md.append("| " + " | ".join(row) + " |")
    return "\n".join(md)


def word_overlaps_bbox(word, bbox) -> bool:
    wx0, wtop, wx1, wbottom = word["x0"], word["top"], word["x1"], word["bottom"]
    bx0, btop, bx1, bbottom = bbox
    return not (wx1 < bx0 or wx0 > bx1) and not (wbottom < btop or wtop > bbottom)


def words_to_lines(words, y_tolerance: int = 3) -> List[Dict]:
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines, current, current_top = [], [words[0]], words[0]["top"]
    for w in words[1:]:
        if abs(w["top"] - current_top) <= y_tolerance:
            current.append(w)
        else:
            lines.append(current)
            current, current_top = [w], w["top"]
    lines.append(current)
    out = []
    for line_words in lines:
        line_words = sorted(line_words, key=lambda w: w["x0"])
        out.append({
            "top": min(w["top"] for w in line_words),
            "text": normalize_text(" ".join(w["text"] for w in line_words)),
        })
    return out


def extract_pdf_to_markdown_with_tables(pdf_path: str) -> str:
    all_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = []
            try:
                for tbl in page.find_tables():
                    extracted = tbl.extract()
                    if not extracted:
                        continue
                    filled = sum(
                        1 for row in extracted if row
                        for cell in row if cell and str(cell).strip()
                    )
                    total = sum(len(row) for row in extracted if row)
                    if total == 0 or filled / total < 0.7:
                        continue
                    meaningful_cols = max(
                        sum(1 for c in row if c and str(c).strip())
                        for row in extracted if row
                    )
                    if meaningful_cols < 3:
                        continue
                    tables.append(tbl)
            except Exception:
                tables = []

            table_blocks = []
            for tbl in tables:
                try:
                    md_table = table_to_markdown(tbl.extract())
                    if md_table.strip():
                        table_blocks.append({
                            "top": tbl.bbox[1], "bottom": tbl.bbox[3],
                            "type": "table", "text": md_table,
                        })
                except Exception:
                    continue

            try:
                words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            except Exception:
                words = []

            safe_words = [
                w for w in words
                if not any(word_overlaps_bbox(w, tbl.bbox) for tbl in tables)
            ]
            text_lines = words_to_lines(safe_words)
            text_blocks = [
                {"top": ln["top"], "type": "text", "text": ln["text"]}
                for ln in text_lines if ln["text"]
            ]
            merged = sorted(text_blocks + table_blocks, key=lambda x: x["top"])
            page_parts = [f"*Page {page_num}*"] + [b["text"] for b in merged]
            all_pages.append(normalize_text("\n".join(page_parts)))

    return "\n\n".join(all_pages)


def load_all_pdfs(folder_path: str) -> List[Dict[str, str]]:
    documents = []
    for file in sorted(os.listdir(folder_path)):
        if not file.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(folder_path, file)
        try:
            markdown_text = extract_pdf_to_markdown_with_tables(file_path)
            documents.append({"pdf_name": file, "text": markdown_text})
            print(f"  ✓ Loaded: {file}")
        except Exception as e:
            print(f"  ✗ Failed: {file} -> {e}")
    return documents


# ──────────────────────────────────────────────────────────────────────────────
# Markdown Chunker (Pass 1)
# ──────────────────────────────────────────────────────────────────────────────

class MarkdownChunker:
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )

    def split(self, markdown_text: str) -> List[Dict[str, Any]]:
        markdown_text = normalize_text(markdown_text)
        docs = self.splitter.split_text(markdown_text)
        out = []
        for i, d in enumerate(docs):
            header_path = " > ".join(
                d.metadata.get(k, "")
                for _, k in self.headers_to_split_on
                if d.metadata.get(k)
            )
            out.append({
                "md_chunk_id": i,
                "content": normalize_text(d.page_content),
                "header_path": header_path or "(no-header)",
            })
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Medical Semantic Chunker (Pass 2)
# ──────────────────────────────────────────────────────────────────────────────

class MedicalSemanticChunker:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_chunk_size: int = 500,
        max_chunk_size: int = 800,
        merge_similarity_threshold: float = 0.80,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.merge_similarity_threshold = merge_similarity_threshold
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
        self._embedding_model_name = embedding_model_name
        self.embed_model = None
        self._embed_disabled = False
        self.embedding_cache: Dict[int, List[float]] = {}
        self.policy_headers = [
            "coverage criteria", "authorization", "step therapy", "quantity limits",
            "specialist", "reauthorization", "initial authorization", "continuation",
            "approval criteria", "prior authorization",
            "criteria for continuation of therapy", "criteria for initial therapy",
            "coverage policy",
        ]
        self.noise_patterns = [
            "table of contents", "policy number", "effective date",
            "last reviewed", "medical benefit policy", "trade name",
        ]
        self._all_brand_aliases = {
            a.lower() for aliases in BRAND_ALIASES.values() for a in aliases
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def token_overlap_text(self, text: str, overlap_tokens: int = 150) -> str:
        if self.tokenizer is None:
            return " ".join(text.split()[-overlap_tokens:])
        tokens = self.tokenizer.encode(text)
        return self.tokenizer.decode(tokens[-overlap_tokens:])

    def count_tokens(self, text: str) -> int:
        text = normalize_text(text)
        if self.tokenizer is None:
            return max(1, len(text or "") // 4)
        return len(self.tokenizer.encode(text or ""))

    def get_embedding(self, text: str) -> List[float]:
        text = normalize_text(text)
        h = hash(text)
        if h in self.embedding_cache:
            return self.embedding_cache[h]
        if self._embed_disabled:
            return [0.0]
        if self.embed_model is None:
            try:
                self.embed_model = SentenceTransformer(self._embedding_model_name)
            except Exception:
                self._embed_disabled = True
                return [0.0]
        emb = self.embed_model.encode(text).tolist()
        self.embedding_cache[h] = emb
        return emb

    def cosine_similarity(self, text1: str, text2: str) -> float:
        if self._embed_disabled:
            return 0.0
        e1 = np.array(self.get_embedding(text1))
        e2 = np.array(self.get_embedding(text2))
        if e1.shape == (1,):
            return 0.0
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))

    # ── Boundary Detectors ───────────────────────────────────────────────────

    def _is_page_break(self, s: str) -> bool:
        return (
            s.startswith("---")
            or (s.startswith("*Page") and s.endswith("*"))
            or bool(re.match(r"^\s*\[?Page\s+\d+", s, re.I))
        )

    def _is_policy_header(self, s: str) -> bool:
        sl = normalize_text(s).lower().strip()
        if len(sl.split()) > 8 or sl.endswith(".") or sl.endswith(":"):
            return False
        if self._is_markdown_table_row(sl):
            return False
        for h in self.policy_headers:
            if sl == h:
                return True
            if re.match(rf"^[a-z0-9\.\-\)\s]*{re.escape(h)}[a-z0-9\.\-\)\s]*$", sl, re.I):
                return True
        return False

    def _is_markdown_table_row(self, s: str) -> bool:
        s = s.strip()
        if not (s.startswith("|") and s.endswith("|")):
            return False
        cells = [c.strip() for c in s.split("|")[1:-1]]
        non_empty = [c for c in cells if c]
        if len(non_empty) < 2:
            return False
        if all(re.fullmatch(r"[-: ]+", c or "-") for c in cells):
            return True
        return True

    def _mentions_brand(self, s: str) -> bool:
        sl = normalize_text(s).lower()
        return any(re.search(rf"\b{re.escape(a)}\b", sl) for a in self._all_brand_aliases)

    def _is_noise(self, s: str) -> bool:
        sl = normalize_text(s).lower()
        return any(n in sl for n in self.noise_patterns)

    def _normalize_boundary_type(self, btype: str) -> str:
        if btype in ("table", "policy_header", "page_break"):
            return btype
        return "para"

    # ── Pass 2a: Split on Boundaries ─────────────────────────────────────────

    def split_by_boundaries(self, text: str) -> List[Tuple[str, str, int]]:
        text = normalize_text(text)
        segments: List[Tuple[str, str, int]] = []
        current: List[str] = []
        boundary = "start"
        pos = 0
        in_table = False

        def flush(new_boundary: str):
            nonlocal current, boundary, pos
            if current:
                content = normalize_text("\n".join(current))
                if content.strip():
                    segments.append((boundary, content, pos))
                    pos += len(content)
                current = []
            boundary = new_boundary

        for line in text.split("\n"):
            s = normalize_text(line.strip())
            if self._is_markdown_table_row(s):
                if not in_table:
                    flush("table")
                    in_table = True
                current.append(s)
                continue
            elif in_table and not self._is_markdown_table_row(s):
                flush("paragraph")
                in_table = False

            if self._is_page_break(s):
                flush("page_break")
                current.append(s)
            elif self._is_policy_header(s):
                flush("policy_header")
                current.append(s)
            elif self._mentions_brand(s) and (s.startswith("#") or s.endswith(":") or len(s.split()) <= 6):
                flush("paragraph")
                current.append(s)
            elif not s and current:
                current.append("")
                flush("paragraph")
            else:
                current.append(s)

        if current:
            content = normalize_text("\n".join(current))
            if content.strip():
                segments.append((boundary, content, pos))
        return segments

    # ── Pass 2b: Enforce Max Size ─────────────────────────────────────────────

    def _enforce_max_size(self, segs):
        out = []
        for btype, content, pos in segs:
            if self.count_tokens(content) <= self.max_chunk_size or btype == "table":
                out.append((btype, content, pos))
                continue
            sentences = re.split(r"(?<=[.!?])\s+", content)
            buf = []
            for sent in sentences:
                cand = (" ".join(buf) + " " + sent).strip()
                if self.count_tokens(cand) > self.max_chunk_size and buf:
                    out.append((btype, " ".join(buf).strip(), pos))
                    pos += len(" ".join(buf))
                    buf = [sent]
                else:
                    buf.append(sent)
            if buf:
                out.append((btype, " ".join(buf).strip(), pos))
                pos += len(" ".join(buf))
        return out

    # ── Pass 2c: Structural Merge ─────────────────────────────────────────────

    def _structural_merge(self, segs):
        if not segs:
            return segs
        priority = {"table": 4, "policy_header": 3, "paragraph": 2, "brand": 2, "start": 1}
        merged = [segs[0]]
        for btype, content, pos in segs[1:]:
            pbtype, pcontent, ppos = merged[-1]
            if btype == "table":
                combined = pcontent + "\n\n" + content
                if pbtype not in ("page_break",) and self.count_tokens(combined) <= self.max_chunk_size:
                    merged[-1] = ("table", combined, ppos)
                else:
                    merged.append((btype, content, pos))
                continue
            if pbtype in {"page_break", "table", "policy_header"} or btype in {"page_break", "table", "policy_header"}:
                merged.append((btype, content, pos))
                continue
            combined = pcontent + "\n" + content
            if self.count_tokens(combined) > self.max_chunk_size:
                merged.append((btype, content, pos))
                continue
            if self.count_tokens(pcontent) < self.min_chunk_size or self.count_tokens(content) < self.min_chunk_size:
                kept = pbtype if priority.get(pbtype, 0) >= priority.get(btype, 0) else btype
                merged[-1] = (kept, combined, ppos)
            else:
                merged.append((btype, content, pos))
        return merged

    # ── Pass 2d: Semantic Merge ───────────────────────────────────────────────

    def _semantic_merge(self, segs):
        if not segs or self._embed_disabled:
            return segs
        merged = [segs[0]]
        for btype, content, pos in segs[1:]:
            pbtype, pcontent, ppos = merged[-1]
            if btype in ("page_break", "policy_header", "table"):
                merged.append((btype, content, pos))
                continue
            small = (
                self.count_tokens(pcontent) < self.min_chunk_size
                and self.count_tokens(content) < self.min_chunk_size
            )
            combined = pcontent + "\n" + content
            fits = self.count_tokens(combined) <= self.max_chunk_size
            if small and fits and self.cosine_similarity(pcontent, content) >= self.merge_similarity_threshold:
                merged[-1] = (pbtype, combined, ppos)
            else:
                merged.append((btype, content, pos))
        return merged

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_section(self, text: str, section_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        section_metadata = section_metadata or {}
        text = normalize_text(text)
        segs = self.split_by_boundaries(text)
        segs = self._enforce_max_size(segs)
        segs = self._structural_merge(segs)
        segs = self._semantic_merge(segs)
        chunks = []
        for i, (btype, content, pos) in enumerate(segs):
            content_clean = normalize_text(content.strip())
            if not content_clean:
                continue
            if btype == "page_break" and self.count_tokens(content_clean) < 5:
                continue
            chunks.append({
                "chunk_id": i,
                "boundary_type": btype,
                "chunk_type": self._normalize_boundary_type(btype),
                "content": content_clean,
                "char_position": pos,
                "token_count": self.count_tokens(content_clean),
                "brands": detect_brands(content_clean),
                "policy_params": detect_policy_params(content_clean),
                "pso_indication": detect_pso_indication(content_clean),
                "is_noise": self._is_noise(content_clean),
                "header_path": section_metadata.get("header_path", ""),
                **section_metadata,
            })
        return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Top-level Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class PolicyChunkingPipeline:
    def __init__(
        self,
        min_chunk_size: int = 250,
        max_chunk_size: int = 800,
        merge_similarity_threshold: float = 0.80,
        overlap_tokens: int = 150,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.md_chunker = MarkdownChunker()
        self.overlap_tokens = overlap_tokens
        self.semantic = MedicalSemanticChunker(
            embedding_model_name=embedding_model_name,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            merge_similarity_threshold=merge_similarity_threshold,
        )

    def run(self, markdown_text: str, doc_id: str = "doc", pdf_name: str = "unknown.pdf") -> List[Dict[str, Any]]:
        markdown_text = normalize_text(markdown_text)
        md_chunks = self.md_chunker.split(markdown_text)
        all_chunks = []
        gid = 1
        for mc in md_chunks:
            section_meta = {
                "doc_id": doc_id,
                "pdf_name": pdf_name,
                "md_chunk_id": mc["md_chunk_id"],
                "header_path": mc["header_path"],
                "page_id": mc["md_chunk_id"],
            }
            section_chunks = self.semantic.chunk_section(mc["content"], section_meta)
            prev_content = ""
            for sc in section_chunks:
                content = sc["content"]
                if prev_content:
                    overlap_text = self.semantic.token_overlap_text(prev_content, self.overlap_tokens)
                    content = overlap_text + "\n\n" + content
                sc["content"] = normalize_text(content)
                prev_content = sc["content"]
                header_path = sc.get("header_path", "")
                all_chunks.append({
                    "chunk_id": make_chunk_id(pdf_name, gid),
                    "pdf_name": section_meta["pdf_name"],
                    "brand_name": sc["brands"],
                    "policy_param": sc["policy_params"],
                    "page_id": sc["page_id"],
                    "token_count": sc["token_count"],
                    "chunk_type": sc["chunk_type"],
                    "pso_indication": sc["pso_indication"],
                    "content": sc["content"],
                    "chunk_used": True,
                    "header_path": header_path,
                    "section_hierarchy": [h.strip() for h in header_path.split(">") if h.strip()],
                })
                gid += 1
        return all_chunks
