"""
convert_to_okf.py — PDF-to-OKF Bundle Converter
=================================================
Converts the data/ directory of Nepali legal PDFs into an OKF v0.2 knowledge
bundle (okf_bundle/) with YAML frontmatter, markdown body text, and
automated cross-links.

Features:
  - Three-tier text extraction: direct PyMuPDF → OCR fallback (EasyOCR) → stub
  - Incremental mode: skips PDFs whose .md is already up-to-date (SHA-256 hash)
  - --force flag: bypasses freshness checks and reprocesses everything
  - Four-tier summary: skipped / direct / OCR / failed

Usage:
    python convert_to_okf.py            # Incremental (default)
    python convert_to_okf.py --force    # Full reconversion
"""

import argparse
import hashlib
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OKF_BUNDLE_DIR = Path("okf_bundle")
MIN_CHARS_THRESHOLD = 50
OCR_DPI = 300
CONVERTER_VERSION = "process:okf-converter/1.0"

# ---------------------------------------------------------------------------
# Category → OKF type mapping (snake_case, path-based, longest match first)
# ---------------------------------------------------------------------------
CATEGORY_TO_TYPE: Dict[str, str] = {
    # Statutes
    "per_act_pdfs":                               "statute",
    "per_act_pdfs.ne":                            "statute",
    "acts_rules_and_judicial_policies":            "statute",
    "bafia_2063":                                 "statute",
    "companies_act_2063":                         "statute",
    "finance_acts":                               "statute",
    "foreign_exchange_act":                       "statute",
    "income_tax_act_2058":                        "statute",
    "nrb_act_2058":                               "statute",
    "public_procurement_and_infrastructure_rules": "statute",
    # Legislative Bills
    "parliament_bills_and_legislative_acts":       "legislative_bill",
    # Case Law
    "supreme_court_verdicts_and_precedents":       "case_law",
    "high_court_and_appellate_decisions":          "case_law",
    # Circulars (NRB bank supervision — literally titled "circular_*")
    "nrb_circulars/bank_supervision":             "circular",
    "nrb_circulars":                              "circular",
    # Regulatory Directives
    "nrb_monetary_and_banking_directives":         "regulatory_directive",
    "sebon_capital_market_regulations":            "regulatory_directive",
    "energy_and_electricity_regulations":          "regulatory_directive",
    # Fiscal Policy
    "fiscal_policy_and_budget_directives":         "fiscal_policy",
    # Annual Reports
    "court_bulletins_and_annual_reports":          "annual_report",
}

# Category → output subdirectory mapping
CATEGORY_TO_DIR: Dict[str, str] = {
    "per_act_pdfs":                               "statutes",
    "per_act_pdfs.ne":                            "statutes",
    "acts_rules_and_judicial_policies":            "statutes",
    "bafia_2063":                                 "statutes",
    "companies_act_2063":                         "statutes",
    "finance_acts":                               "statutes",
    "foreign_exchange_act":                       "statutes",
    "income_tax_act_2058":                        "statutes",
    "nrb_act_2058":                               "statutes",
    "public_procurement_and_infrastructure_rules": "statutes",
    "parliament_bills_and_legislative_acts":       "legislative_bills",
    "supreme_court_verdicts_and_precedents":       "case_law/supreme_court",
    "high_court_and_appellate_decisions":          "case_law/high_court",
    "nrb_circulars/bank_supervision":             "circulars/bank_supervision",
    "nrb_circulars":                              "circulars",
    "nrb_monetary_and_banking_directives":         "regulatory_directives/nrb_monetary_banking",
    "sebon_capital_market_regulations":            "regulatory_directives/sebon",
    "energy_and_electricity_regulations":          "regulatory_directives/energy",
    "fiscal_policy_and_budget_directives":         "fiscal_policy",
    "court_bulletins_and_annual_reports":          "annual_reports",
}

# ---------------------------------------------------------------------------
# OCR availability check (EasyOCR + PyTesseract fallback)
# ---------------------------------------------------------------------------
_tesseract_available: Optional[bool] = None
_easyocr_reader: Any = None


def check_tesseract() -> bool:
    """Check if Tesseract OCR is available on the system."""
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available
    try:
        import pytesseract
        try:
            import tesseract_bin
        except ImportError:
            pass

        # Check standard Windows installation paths and Python environment paths
        possible_paths = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")),
            Path(sys.prefix) / "Scripts" / "tesseract.exe",
            Path(sys.prefix) / "tesseract.exe",
            Path(r"C:\Tesseract-OCR\tesseract.exe"),
        ]
        for p in possible_paths:
            if p.exists():
                pytesseract.pytesseract.tesseract_cmd = str(p)
                break

        pytesseract.get_tesseract_version()
        _tesseract_available = True
        logger.info(f"Tesseract OCR detected ({pytesseract.pytesseract.tesseract_cmd}) — Tesseract fallback enabled (secondary to EasyOCR).")
    except Exception:
        _tesseract_available = False
        logger.warning("Tesseract OCR not found either. Both OCR engines missing.")
    return _tesseract_available


def check_ocr() -> bool:
    """Check if any OCR engine (EasyOCR or Tesseract) is available."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return True
    try:
        import easyocr
        logger.info("Initializing EasyOCR reader for Devanagari & English ('ne', 'en')...")
        _easyocr_reader = easyocr.Reader(['ne', 'en'], gpu=True)
        logger.info("EasyOCR initialized successfully — OCR primary enabled.")
        return True
    except Exception as e:
        logger.warning(f"EasyOCR initialization failed: {e}. Checking Tesseract...")
        return check_tesseract()


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean extracted PDF text (mirrors ingest.py logic)."""
    if not text:
        return ""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bPage \d+ of \d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?im)^\s*page\s+\d+\s*$', '', text)
    return text.strip()


def extract_page_text(page: Any, pdf_path: Path, page_num: int) -> Tuple[str, str]:
    """
    Extract text from a single PDF page with OCR fallback.
    Returns (text, method) where method is "direct", "ocr", or "failed".
    """
    # Tier 1: Direct text extraction (fast)
    text = clean_text(page.get_text("text"))
    if len(text) >= MIN_CHARS_THRESHOLD:
        return text, "direct"

    # Tier 2: OCR fallback (handles scanned pages)
    if check_ocr():
        try:
            pix = page.get_pixmap(dpi=OCR_DPI)
            if _easyocr_reader is not None:
                img_bytes = pix.tobytes("png")
                results = _easyocr_reader.readtext(img_bytes, detail=0)
                ocr_text = clean_text(" ".join(results))
            else:
                import pytesseract
                from PIL import Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = clean_text(pytesseract.image_to_string(img, lang="eng+nep"))

            if len(ocr_text) >= MIN_CHARS_THRESHOLD:
                return ocr_text, "ocr"
        except Exception as e:
            logger.warning(f"OCR failed for {pdf_path.name} page {page_num + 1}: {e}")

    # Tier 3: Empty/failed
    return text or "", "failed"


def extract_pdf_text(pdf_path: Path) -> Tuple[str, str]:
    """
    Extract full text from a PDF file with per-page OCR fallback.
    Returns (full_text_with_page_headers, extraction_method).
    Method is "direct" if all pages used direct, "ocr" if any page needed OCR,
    or "failed" if total extracted text is below threshold.
    """
    doc = fitz.open(str(pdf_path))
    page_texts: List[str] = []
    overall_method = "direct"

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text, method = extract_page_text(page, pdf_path, page_num)

        if method == "ocr" and overall_method == "direct":
            overall_method = "ocr"

        if text:
            page_texts.append(f"## Page {page_num + 1}\n\n{text}")

    doc.close()

    full_text = "\n\n".join(page_texts)
    if len(full_text.replace("#", "").strip()) < MIN_CHARS_THRESHOLD:
        return "", "failed"

    return full_text, overall_method


# ---------------------------------------------------------------------------
# Hashing & freshness
# ---------------------------------------------------------------------------
def compute_pdf_hash(pdf_path: Path) -> str:
    """SHA-256 hash of the source PDF file contents."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def is_concept_current(md_path: Path, pdf_path: Path) -> bool:
    """Check if an existing .md concept is up to date with its source PDF."""
    if not md_path.exists():
        return False
    try:
        text = md_path.read_text(encoding="utf-8")
        # Split on first two --- delimiters
        parts = text.split("---", 2)
        if len(parts) < 3:
            return False
        fm = yaml.safe_load(parts[1])
        if not fm:
            return False
        existing_hash = fm.get("source_pdf_hash", "")
        current_hash = compute_pdf_hash(pdf_path)
        return existing_hash == current_hash
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def resolve_category(pdf_path: Path) -> str:
    """
    Resolve the category key for a PDF by matching its relative path
    against CATEGORY_TO_TYPE keys (longest/most-specific match first).
    """
    rel = pdf_path.relative_to(DATA_DIR)
    # Build the relative directory path using forward slashes
    rel_dir = str(rel.parent).replace("\\", "/")

    # Sort keys by length descending so longer (more specific) paths match first
    for key in sorted(CATEGORY_TO_TYPE.keys(), key=len, reverse=True):
        if rel_dir == key or rel_dir.startswith(key + "/"):
            return key

    # Fallback: use the first directory component
    first_dir = rel.parts[0] if len(rel.parts) > 1 else "unknown"
    logger.warning(f"No category mapping for {rel_dir}, using first dir: {first_dir}")
    return first_dir


def slugify_filename(filename: str) -> str:
    """Convert a PDF filename to a clean snake_case slug for the .md file."""
    name = Path(filename).stem
    # Remove common prefixes/suffixes that add no value
    name = re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\.PDF$', '', name)
    # Replace non-alphanumeric (preserving Devanagari Unicode) with underscores
    name = re.sub(r'[^\w\u0900-\u097F]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name).strip('_').lower()
    # Ensure non-empty
    if not name:
        name = "unnamed"
    return name


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Try to extract a 4-digit year from a filename."""
    match = re.search(r'(\d{4})', filename)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            return year
    return None


def derive_title(filename: str) -> str:
    """Derive a human-readable title from a PDF filename."""
    name = Path(filename).stem
    # Remove .pdf extension artifacts
    name = re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE)
    # Replace underscores with spaces
    name = name.replace('_', ' ').strip()
    # Title case
    return name.title() if name else "Untitled"


def detect_language(pdf_path: Path, category: str) -> str:
    """Detect language from folder name or filename convention."""
    if category == "per_act_pdfs.ne" or pdf_path.stem.endswith("_ne"):
        return "ne"
    return "en"


def build_tags(okf_type: str, title: str, year: Optional[int], language: str) -> List[str]:
    """Build a tags list from type, title keywords, and year."""
    tags = [okf_type, "nepal-law"]
    if language == "ne":
        tags.append("nepali")
    if year:
        tags.append(str(year))
    # Add a few keywords from the title (skip short words)
    for word in title.lower().split():
        word_clean = re.sub(r'[^a-z0-9\u0900-\u097F]', '', word)
        if len(word_clean) > 3 and word_clean not in tags:
            tags.append(word_clean)
            if len(tags) >= 8:
                break
    return tags


# ---------------------------------------------------------------------------
# OKF concept writer
# ---------------------------------------------------------------------------
def write_concept(
    md_path: Path,
    okf_type: str,
    title: str,
    description: str,
    language: str,
    tags: List[str],
    resource: str,
    category: str,
    year: Optional[int],
    extraction_method: str,
    pdf_hash: str,
    body: str,
    translations: Optional[Dict[str, str]] = None,
) -> None:
    """Write a single OKF concept .md file with YAML frontmatter."""
    md_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    frontmatter: Dict[str, Any] = {
        "type": okf_type,
        "title": title,
        "description": description,
        "language": language,
        "tags": tags,
        "resource": resource,
        "jurisdiction": "Nepal",
        "source_category": category,
        "extraction_method": extraction_method,
        "source_pdf_hash": pdf_hash,
        "sources": [
            {
                "resource": resource,
                "title": "Source PDF",
                "author": CONVERTER_VERSION,
                "last_modified": today,
            }
        ],
        "generated": {
            "by": CONVERTER_VERSION,
            "at": now,
        },
    }

    if year:
        frontmatter["act_year"] = year
    if translations:
        frontmatter["translations"] = translations

    # Build the full file content
    fm_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False)
    content = f"---\n{fm_str}---\n\n{body}\n"

    md_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Index & log generators
# ---------------------------------------------------------------------------
def write_index_md(directory: Path, title: str, concepts: List[Path]) -> None:
    """Write an OKF index.md file for a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.md"

    fm = {
        "type": "index",
        "title": title,
        "generated": {
            "by": CONVERTER_VERSION,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    fm_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)

    lines = [f"---\n{fm_str}---\n"]
    lines.append(f"# {title}\n")
    lines.append(f"This directory contains {len(concepts)} concept(s).\n")

    # List concepts as links
    for concept in sorted(concepts, key=lambda p: p.stem):
        if concept.name in ("index.md", "log.md"):
            continue
        concept_id = concept.stem
        lines.append(f"- [{concept_id}](./{concept.name})")

    # List subdirectories
    subdirs = [d for d in sorted(directory.iterdir()) if d.is_dir()]
    if subdirs:
        lines.append(f"\n## Subdirectories\n")
        for sd in subdirs:
            lines.append(f"- [{sd.name}](./{sd.name}/index.md)")

    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_log_md(bundle_dir: Path, stats: Dict[str, int]) -> None:
    """Write the OKF log.md at the bundle root."""
    fm = {
        "type": "log",
        "title": "OKF Bundle Change Log",
        "generated": {
            "by": CONVERTER_VERSION,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    fm_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f"""---
{fm_str}---

# Change Log

## {now_str} — Initial conversion

Converted {stats['total']} source PDFs from `data/` into OKF v0.2 concepts.

| Metric | Count |
|---|---|
| Skipped (up to date) | {stats['skipped']} |
| Direct extraction | {stats['direct']} |
| OCR fallback | {stats['ocr']} |
| Failed (stub) | {stats['failed']} |
"""
    log_path = bundle_dir / "log.md"
    # Append if exists, write if new
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        # Find the body after the frontmatter
        parts = existing.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]
            new_entry = f"\n## {now_str} — Incremental update\n\n"
            new_entry += f"Processed {stats['total']} source PDFs.\n\n"
            new_entry += f"| Metric | Count |\n|---|---|\n"
            new_entry += f"| Skipped (up to date) | {stats['skipped']} |\n"
            new_entry += f"| Direct extraction | {stats['direct']} |\n"
            new_entry += f"| OCR fallback | {stats['ocr']} |\n"
            new_entry += f"| Failed (stub) | {stats['failed']} |\n"
            updated = f"---\n{parts[1]}---{body}\n{new_entry}"
            log_path.write_text(updated, encoding="utf-8")
            return

    log_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Cross-linking pass
# ---------------------------------------------------------------------------
def build_title_index(concepts_map: Dict[str, Path]) -> Dict[str, str]:
    """
    Build a mapping of normalized act titles → concept paths for cross-linking.
    concepts_map: {slug: md_path}
    """
    title_index: Dict[str, str] = {}
    for slug, md_path in concepts_map.items():
        try:
            text = md_path.read_text(encoding="utf-8")
            parts = text.split("---", 2)
            if len(parts) < 3:
                continue
            fm = yaml.safe_load(parts[1])
            if not fm or "title" not in fm:
                continue
            title = fm["title"]
            # Normalize: lowercase, strip non-alnum
            normalized = re.sub(r'[^a-z0-9\s]', '', title.lower()).strip()
            if normalized:
                # Store the bundle-relative path (without .md) for linking
                rel = md_path.relative_to(OKF_BUNDLE_DIR)
                concept_path = "/" + str(rel.with_suffix("")).replace("\\", "/")
                title_index[normalized] = concept_path
        except Exception:
            continue
    return title_index


def apply_cross_links(concepts_map: Dict[str, Path], title_index: Dict[str, str]) -> int:
    """
    Scan each concept body for mentions of other act titles and insert
    standard markdown links. Returns count of links inserted.
    """
    link_count = 0
    # Sort titles longest-first to avoid partial matches
    sorted_titles = sorted(title_index.keys(), key=len, reverse=True)

    for slug, md_path in concepts_map.items():
        try:
            text = md_path.read_text(encoding="utf-8")
            parts = text.split("---", 2)
            if len(parts) < 3:
                continue
            body = parts[2]
            modified = False

            for title_normalized in sorted_titles:
                concept_path = title_index[title_normalized]
                # Skip self-references
                rel = md_path.relative_to(OKF_BUNDLE_DIR)
                self_path = "/" + str(rel.with_suffix("")).replace("\\", "/")
                if concept_path == self_path:
                    continue

                # Look for the title text in the body (case-insensitive)
                # Only link if not already a markdown link
                pattern = re.compile(
                    r'(?<!\[)(' + re.escape(title_normalized) + r')(?!\])',
                    re.IGNORECASE
                )
                if pattern.search(body):
                    # Get the display title from the concept path
                    display = title_normalized.title()
                    body = pattern.sub(f'[{display}]({concept_path})', body, count=1)
                    modified = True
                    link_count += 1

            if modified:
                body = body.rstrip() + "\n\n<!-- TODO: review cross-links -->\n"
                updated = f"---{parts[1]}---{body}"
                md_path.write_text(updated, encoding="utf-8")

        except Exception as e:
            logger.warning(f"Cross-linking error for {md_path.name}: {e}")
            continue

    return link_count


# ---------------------------------------------------------------------------
# Bilingual twin linking
# ---------------------------------------------------------------------------
def link_bilingual_twins(concepts_map: Dict[str, Path]) -> int:
    """
    For concepts from per_act_pdfs and per_act_pdfs.ne, add translations
    cross-links between English and Nepali twins.
    """
    linked = 0
    # Build a map of base_slug → {en: path, ne: path}
    twins: Dict[str, Dict[str, Path]] = {}

    for slug, md_path in concepts_map.items():
        try:
            text = md_path.read_text(encoding="utf-8")
            parts = text.split("---", 2)
            if len(parts) < 3:
                continue
            fm = yaml.safe_load(parts[1])
            if not fm:
                continue

            lang = fm.get("language", "en")
            cat = fm.get("source_category", "")
            if cat not in ("per_act_pdfs", "per_act_pdfs.ne"):
                continue

            # Compute base slug (strip _ne suffix)
            base = slug.rstrip("_ne") if slug.endswith("_ne") else slug
            if base not in twins:
                twins[base] = {}
            twins[base][lang] = md_path
        except Exception:
            continue

    for base, langs in twins.items():
        if "en" not in langs or "ne" not in langs:
            continue
        try:
            # Add translation link to English concept
            en_path = langs["en"]
            ne_path = langs["ne"]

            ne_rel = "/" + str(ne_path.relative_to(OKF_BUNDLE_DIR).with_suffix("")).replace("\\", "/")
            en_rel = "/" + str(en_path.relative_to(OKF_BUNDLE_DIR).with_suffix("")).replace("\\", "/")

            for path, other_rel, other_lang in [(en_path, ne_rel, "ne"), (ne_path, en_rel, "en")]:
                text = path.read_text(encoding="utf-8")
                parts = text.split("---", 2)
                if len(parts) < 3:
                    continue
                fm = yaml.safe_load(parts[1])
                if not fm:
                    continue
                fm["translations"] = {other_lang: other_rel}
                fm_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
                updated = f"---\n{fm_str}---{parts[2]}"
                path.write_text(updated, encoding="utf-8")

            linked += 1
        except Exception as e:
            logger.warning(f"Twin linking error for {base}: {e}")

    return linked


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------
def collect_pdfs() -> List[Path]:
    """Collect all PDFs from data/ (excluding legacy_data, chroma_db, scrap artifacts)."""
    pdfs = []
    for pdf_path in DATA_DIR.rglob("*.pdf"):
        # Skip non-document directories and scrap artifact files (.md.pdf, _ne.pdf)
        rel = str(pdf_path.relative_to(DATA_DIR)).replace("\\", "/")
        if any(skip in rel for skip in ("chroma_db", "legacy_data", ".DS_Store")):
            continue
        if pdf_path.name.startswith(".") or pdf_path.name in ("_ne.pdf", ".md.pdf"):
            continue
        pdfs.append(pdf_path)
    # Also match .PDF extension
    for pdf_path in DATA_DIR.rglob("*.PDF"):
        rel = str(pdf_path.relative_to(DATA_DIR)).replace("\\", "/")
        if any(skip in rel for skip in ("chroma_db", "legacy_data", ".DS_Store")):
            continue
        if pdf_path.name.startswith(".") or pdf_path.name in ("_ne.pdf", ".md.pdf"):
            continue
        if pdf_path not in pdfs:
            pdfs.append(pdf_path)
    return sorted(pdfs)


def convert(force: bool = False) -> None:
    """Main conversion pipeline."""
    logger.info("=" * 60)
    logger.info("OKF Converter — Starting")
    logger.info("=" * 60)

    # Check OCR availability upfront
    check_ocr()

    # Collect all source PDFs
    pdfs = collect_pdfs()
    logger.info(f"Found {len(pdfs)} source PDFs in {DATA_DIR}/")

    if not pdfs:
        logger.warning("No PDFs found. Nothing to convert.")
        return

    # Create bundle root
    OKF_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    # Tracking
    stats = {"total": len(pdfs), "skipped": 0, "direct": 0, "ocr": 0, "failed": 0}
    concepts_map: Dict[str, Path] = {}  # slug → md_path

    for i, pdf_path in enumerate(pdfs, 1):
        try:
            # Resolve category and output path
            category = resolve_category(pdf_path)
            okf_type = CATEGORY_TO_TYPE.get(category, "statute")
            out_subdir = CATEGORY_TO_DIR.get(category, "uncategorized")
            slug = slugify_filename(pdf_path.name)
            md_path = OKF_BUNDLE_DIR / out_subdir / f"{slug}.md"

            # Incremental check
            if not force and is_concept_current(md_path, pdf_path):
                stats["skipped"] += 1
                concepts_map[slug] = md_path
                if i % 100 == 0:
                    logger.info(f"  [{i}/{len(pdfs)}] Skipped (up to date): {pdf_path.name}")
                continue

            # Extract text
            logger.info(f"  [{i}/{len(pdfs)}] Processing: {pdf_path.name}")
            body_text, method = extract_pdf_text(pdf_path)
            pdf_hash = compute_pdf_hash(pdf_path)

            # Derive metadata
            title = derive_title(pdf_path.name)
            language = detect_language(pdf_path, category)
            year = extract_year_from_filename(pdf_path.name)
            tags = build_tags(okf_type, title, year, language)
            resource = str(pdf_path.relative_to(Path("."))).replace("\\", "/")
            description = f"{language.upper()} text of {title} (Nepal)"

            if method == "failed" or not body_text:
                method = "failed"
                body_text = "<!-- extraction-failed: no text extracted from source PDF -->"

            # Write the concept
            write_concept(
                md_path=md_path,
                okf_type=okf_type,
                title=title,
                description=description,
                language=language,
                tags=tags,
                resource=resource,
                category=category,
                year=year,
                extraction_method=method,
                pdf_hash=pdf_hash,
                body=body_text,
            )

            stats[method] += 1
            concepts_map[slug] = md_path

        except Exception as e:
            logger.error(f"  [{i}/{len(pdfs)}] FAILED: {pdf_path.name} — {e}")
            stats["failed"] += 1
            # Write a stub so the bundle index is complete
            try:
                slug = slugify_filename(pdf_path.name)
                category = resolve_category(pdf_path)
                out_subdir = CATEGORY_TO_DIR.get(category, "uncategorized")
                stub_path = OKF_BUNDLE_DIR / out_subdir / f"{slug}.md"
                pdf_hash = compute_pdf_hash(pdf_path)
                write_concept(
                    md_path=stub_path,
                    okf_type=CATEGORY_TO_TYPE.get(category, "statute"),
                    title=derive_title(pdf_path.name),
                    description=f"Failed to extract: {e}",
                    language=detect_language(pdf_path, category),
                    tags=[CATEGORY_TO_TYPE.get(category, "statute"), "extraction-failed"],
                    resource=str(pdf_path.relative_to(Path("."))).replace("\\", "/"),
                    category=category,
                    year=extract_year_from_filename(pdf_path.name),
                    extraction_method="failed",
                    pdf_hash=pdf_hash,
                    body=f"<!-- extraction-error: {e} -->",
                )
                concepts_map[slug] = stub_path
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Post-processing passes
    # -----------------------------------------------------------------------
    logger.info("Running bilingual twin linking pass...")
    twin_count = link_bilingual_twins(concepts_map)
    logger.info(f"  Linked {twin_count} bilingual twin pairs.")

    logger.info("Running keyword cross-linking pass...")
    title_index = build_title_index(concepts_map)
    link_count = apply_cross_links(concepts_map, title_index)
    logger.info(f"  Inserted {link_count} cross-links.")

    # -----------------------------------------------------------------------
    # Generate index.md files for each directory
    # -----------------------------------------------------------------------
    logger.info("Generating index.md files...")
    for dirpath in sorted(set(p.parent for p in concepts_map.values())):
        md_files = list(dirpath.glob("*.md"))
        md_files = [f for f in md_files if f.name not in ("index.md", "log.md")]
        dir_name = dirpath.relative_to(OKF_BUNDLE_DIR)
        write_index_md(dirpath, str(dir_name).replace("\\", "/").replace("/", " / ").title(), md_files)

    # Root index
    all_concepts = [p for p in concepts_map.values() if p.name not in ("index.md", "log.md")]
    write_index_md(OKF_BUNDLE_DIR, "Nepal Legal Corpus — OKF Bundle", all_concepts)

    # Log
    write_log_md(OKF_BUNDLE_DIR, stats)

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("  OKF Conversion Summary")
    logger.info("=" * 50)
    logger.info(f"  Total source PDFs:        {stats['total']:>6}")
    logger.info(f"  ⏭️  Skipped (up to date):  {stats['skipped']:>6}  ({100*stats['skipped']/max(stats['total'],1):.1f}%)")
    logger.info(f"  ✅ Direct extraction:      {stats['direct']:>6}  ({100*stats['direct']/max(stats['total'],1):.1f}%)")
    logger.info(f"  🔍 OCR fallback used:      {stats['ocr']:>6}  ({100*stats['ocr']/max(stats['total'],1):.1f}%)")
    logger.info(f"  ❌ Failed (stub only):      {stats['failed']:>6}  ({100*stats['failed']/max(stats['total'],1):.1f}%)")
    logger.info(f"  Output: {OKF_BUNDLE_DIR.resolve()}")
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import psutil
    
    # Explicit sequential-execution safeguard
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and 'ingest.py' in ' '.join(cmdline):
                logger.error(
                    "CRITICAL: ingest.py is currently running! "
                    "Running OCR concurrently will cause severe VRAM contention and CUDA Out-Of-Memory errors. "
                    "Please wait for ingest.py to finish before starting convert_to_okf.py."
                )
                sys.exit(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    parser = argparse.ArgumentParser(
        description="Convert data/ PDFs to an OKF v0.2 knowledge bundle.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass incremental freshness checks and reprocess all PDFs.",
    )
    args = parser.parse_args()

    try:
        # Ensure stdout handles Unicode on Windows
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
    except Exception:
        pass

    convert(force=args.force)


if __name__ == "__main__":
    main()
