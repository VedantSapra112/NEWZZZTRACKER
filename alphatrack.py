"""
ALPHATRACK · v1.0 (beta)
A Streamlit app that ingests Indian-newspaper financial extracts (ET, Mint, BS,
HT, FE, BL), sends them to Claude for structured extraction, and renders a
sector-organized news grid.

Run:
    pip install streamlit anthropic pandas pypdf openpyxl
    streamlit run alphatrack.py
"""

from __future__ import annotations

import io
import os
import re
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from anthropic import Anthropic, APIError

try:
    from pypdf import PdfReader
except ImportError:  # pypdf renamed from PyPDF2; fall back gracefully
    PdfReader = None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG & CONSTANTS                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="AlphaTrack",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

TICKER_REFRESH_SEC = 5            # ticker bar auto-refresh interval
MAX_INPUT_CHARS = 120_000         # beta cost guard: cap text sent to Claude
MAX_OUTPUT_TOKENS = 4_000         # cap on Claude's response

# Sector taxonomy. `id` is the lowercase key Claude must return.
SECTORS = [
    {"id": "banking",  "name": "Banking/Finance", "color": "#38bdf8", "icon": "₹"},
    {"id": "it",       "name": "IT/Tech",         "color": "#a78bfa", "icon": "⌬"},
    {"id": "energy",   "name": "Energy",          "color": "#fbbf24", "icon": "⚡"},
    {"id": "pharma",   "name": "Pharma",          "color": "#34d399", "icon": "✚"},
    {"id": "auto",     "name": "Auto",            "color": "#fb7185", "icon": "⛭"},
    {"id": "consumer", "name": "Consumer/FMCG",   "color": "#f472b6", "icon": "◈"},
]

SECTOR_IDS = {s["id"] for s in SECTORS}
SECTOR_BY_ID = {s["id"]: s for s in SECTORS}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STYLES                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;700&display=swap');

    .stApp { background:#09090b; }
    .block-container { padding-top:1rem; max-width:100%; }

    .ticker-bar {
        background:#0c0c0e; border-top:1px solid #27272a; border-bottom:1px solid #27272a;
        padding:7px 12px; font-family:'JetBrains Mono',monospace; font-size:12px;
        white-space:nowrap; overflow:hidden; color:#a1a1aa; margin-bottom:14px;
    }
    .ticker-item { margin-right:26px; }
    .tick-up   { color:#34d399; }
    .tick-down { color:#fb7185; }

    .sub-header {
        font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:0.18em;
        color:#71717a; margin:4px 0 14px 0;
    }

    .col-header-row, .news-row {
        display:grid; grid-template-columns:1.1fr 2fr 1.5fr 1.8fr; gap:18px;
        padding:12px 8px;
    }
    .col-header-row {
        font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.16em;
        color:#52525b; text-transform:uppercase; border-bottom:1px solid #27272a;
    }
    .news-row {
        border-bottom:1px solid #18181b; align-items:start;
    }
    .news-row:hover { background:#0e0e11; }

    .company-block .ticker {
        font-family:'JetBrains Mono',monospace; font-weight:800; font-size:15px; color:#fafafa;
    }
    .company-block .name {
        font-family:'Inter',sans-serif; font-size:12px; color:#a1a1aa; margin:2px 0 6px 0;
    }
    .sector-row { display:flex; gap:6px; align-items:center; }

    .sector-pill {
        font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
        letter-spacing:0.08em; padding:2px 7px; border-radius:4px; border:1px solid;
    }
    .impact-pill {
        font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:800;
        padding:2px 7px; border-radius:4px;
    }
    .impact-HIGH { background:#fb71851a; color:#fb7185; border:1px solid #fb718566; }
    .impact-MED  { background:#fbbf241a; color:#fbbf24; border:1px solid #fbbf2466; }
    .impact-LOW  { background:#52525b1a; color:#a1a1aa; border:1px solid #52525b66; }

    .headline-block .headline {
        font-family:'Inter',sans-serif; font-size:14px; color:#e4e4e7; line-height:1.45;
    }
    .headline-block .sources { margin-top:8px; display:flex; gap:5px; align-items:center; flex-wrap:wrap; }
    .source-tag {
        font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
        padding:2px 6px; border-radius:3px; background:#18181b; color:#a1a1aa;
        border:1px solid #27272a;
    }
    .src-ET   { color:#fb923c; border-color:#fb923c66; }
    .src-Mint { color:#38bdf8; border-color:#38bdf866; }
    .src-BS   { color:#f472b6; border-color:#f472b666; }
    .src-HT   { color:#a78bfa; border-color:#a78bfa66; }
    .src-FE   { color:#34d399; border-color:#34d39966; }
    .src-BL   { color:#fbbf24; border-color:#fbbf2466; }

    .metric-line {
        font-family:'JetBrains Mono',monospace; font-size:12px; color:#d4d4d8;
        margin-bottom:4px; padding-left:2px;
    }
    .analysis-line {
        font-family:'Inter',sans-serif; font-size:12.5px; color:#c4c4c8; line-height:1.45;
    }

    .empty-state {
        text-align:center; padding:60px 20px; color:#52525b;
        font-family:'JetBrains Mono',monospace; font-size:13px; letter-spacing:0.16em;
    }
    .sidebar-label {
        font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.18em;
        color:#71717a; margin:16px 0 6px 0; text-transform:uppercase;
    }

    .stButton>button {
        font-family:'JetBrains Mono',monospace !important; font-weight:700 !important;
        letter-spacing:0.1em !important; font-size:12px !important;
        background:#fafafa !important; color:#09090b !important; border:none !important;
    }
    .stButton>button:disabled { background:#27272a !important; color:#52525b !important; }
    .stDownloadButton>button {
        font-family:'JetBrains Mono',monospace !important; font-size:11px !important;
        background:#18181b !important; color:#e4e4e7 !important; border:1px solid #27272a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TICKER BAR                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Static demo ticker. In a real beta you'd wire this to a quotes API; for now it
# rotates a deterministic pseudo-random walk so the bar feels alive without
# pretending to be real market data.
_TICKER_SEED = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "SBIN", "BHARTIARTL", "ITC", "LT", "SUNPHARMA"]


def render_ticker_bar() -> None:
    now = datetime.now()
    # Deterministic pseudo-movement keyed on the current second.
    items = []
    for i, sym in enumerate(_TICKER_SEED):
        pct = ((now.second * (i + 3)) % 47 - 23) / 10.0  # -2.3 .. +2.3 ish
        cls = "tick-up" if pct >= 0 else "tick-down"
        arrow = "▲" if pct >= 0 else "▼"
        items.append(
            f'<span class="ticker-item">{sym} '
            f'<span class="{cls}">{arrow} {abs(pct):.2f}%</span></span>'
        )
    clock = now.strftime("%H:%M:%S")
    html = (
        f'<div class="ticker-bar">'
        f'<span style="color:#fafafa;font-weight:800;margin-right:20px;">▲ ALPHATRACK</span>'
        f'<span style="color:#52525b;margin-right:24px;">{clock} IST</span>'
        + "".join(items) +
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEXT EXTRACTION                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def extract_text_from_upload(f) -> str:
    """Pull raw text out of an uploaded PDF/TXT/MD file."""
    name = (f.name or "").lower()
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed — run: pip install pypdf")
        reader = PdfReader(io.BytesIO(f.getvalue()))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(pages)
    # txt / md / anything text-like
    data = f.getvalue()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def clean_text(text: str) -> str:
    """Collapse whitespace and strip control noise from extracted text."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLAUDE EXTRACTION                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = """You are a sell-side equity research analyst's extraction \
engine. You read raw Indian financial-newspaper text and return ONLY structured \
JSON describing the noteworthy company / market items in it.

Return a JSON object with a single key "items", whose value is an array. Each \
item must have EXACTLY these keys:
  - "sector": one of ["banking","it","energy","pharma","auto","consumer"]
  - "company": company name (string), or "" if market-wide
  - "ticker": NSE ticker in caps if identifiable, else "—"
  - "headline": one concise sentence summarizing the item
  - "sources": array of source codes among ["ET","Mint","BS","HT","FE","BL"] \
that the item appears in (infer from the "SOURCE FILE" headers)
  - "metrics": array of short strings of concrete numbers \
(e.g. "Q3 PAT ₹4,200cr +18% YoY", "NIM 3.6%")
  - "analyst_assessment": array of 1-3 short analyst-style takeaways
  - "impact": one of "HIGH","MED","LOW"
  - "sentiment": one of "positive","negative","neutral"

Rules:
  - Only include items that map cleanly to one of the six sectors.
  - Do NOT invent numbers. If no metric is stated, leave "metrics" empty.
  - Keep each string tight; this feeds a dense terminal-style grid.
  - Output ONLY the JSON object. No prose, no markdown fences."""


def call_claude_extraction(client: Anthropic, model: str, combined_text: str) -> dict:
    """Send newspaper text to Claude and parse a structured JSON response."""
    # Beta cost guard: never ship more than MAX_INPUT_CHARS to the model.
    if len(combined_text) > MAX_INPUT_CHARS:
        combined_text = combined_text[:MAX_INPUT_CHARS] + "\n\n[TRUNCATED]"

    resp = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "Extract structured news items from the newspaper text below.\n\n"
                f"{combined_text}"
            ),
        }],
    )

    # Concatenate all text blocks from the response.
    text_out = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    ).strip()

    # Strip any stray markdown fences before parsing.
    text_out = re.sub(r"^```(?:json)?\s*", "", text_out)
    text_out = re.sub(r"\s*```$", "", text_out).strip()

    try:
        parsed = json.loads(text_out)
    except json.JSONDecodeError:
        # Try to recover the first {...} blob
        m = re.search(r"\{[\s\S]*\}", text_out)
        if not m:
            raise ValueError("Claude did not return parseable JSON.")
        parsed = json.loads(m.group(0))

    if not isinstance(parsed, dict) or "items" not in parsed:
        raise ValueError("Claude response missing 'items' key.")
    return parsed


def normalize_item(it: dict) -> dict | None:
    """Validate and normalize a single extracted item."""
    if not isinstance(it, dict):
        return None
    sector = (it.get("sector") or "").lower().strip()
    if sector not in SECTOR_IDS:
        return None
    return {
        "sector": sector,
        "company": str(it.get("company") or "").strip() or "—",
        "ticker": str(it.get("ticker") or "—").strip(),
        "headline": str(it.get("headline") or "").strip(),
        "sources": [s for s in (it.get("sources") or []) if isinstance(s, str)],
        "metrics": [m for m in (it.get("metrics") or []) if isinstance(m, str) and m.strip()],
        "analyst_assessment": [
            a for a in (it.get("analyst_assessment") or []) if isinstance(a, str) and a.strip()
        ],
        "impact": (it.get("impact") or "MED").upper().strip(),
        "sentiment": (it.get("sentiment") or "neutral").lower().strip(),
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RENDERING THE NEWS GRID                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def sector_pill_html(sector_id: str) -> str:
    s = SECTOR_BY_ID.get(sector_id)
    if not s:
        return ""
    color = s["color"]
    return (
        f'<span class="sector-pill" '
        f'style="color:{color};border-color:{color}66;background:{color}10;">'
        f'{s["icon"]} {s["name"].split("/")[0].strip()[:7].upper()}'
        f'</span>'
    )


def source_tag_html(src: str) -> str:
    cls = src if src in {"ET", "Mint", "BS", "HT", "FE", "BL"} else "Other"
    return f'<span class="source-tag src-{cls}">{src}</span>'


def render_item_row(item: dict, idx: int) -> str:
    s = SECTOR_BY_ID[item["sector"]]
    company_html = (
        f'<div class="company-block">'
        f'  <div class="ticker">{item["ticker"]}</div>'
        f'  <div class="name">{item["company"]}</div>'
        f'  <div class="sector-row">'
        f'    {sector_pill_html(item["sector"])}'
        f'    <span class="impact-pill impact-{item["impact"]}">{item["impact"]}</span>'
        f'  </div>'
        f'</div>'
    )

    sources_html = "".join(source_tag_html(src) for src in item["sources"]) or \
                   '<span style="color:#52525b;font-family:JetBrains Mono;font-size:10px;">—</span>'
    n_sources = len(item["sources"])
    coverage_str = f'{n_sources}/6 papers' if n_sources else 'unsourced'

    headline_html = (
        f'<div class="headline-block">'
        f'  <div class="headline">{item["headline"]}</div>'
        f'  <div class="sources">'
        f'    <span style="font-family:JetBrains Mono;font-size:9px;letter-spacing:0.18em;color:#52525b;">COVERAGE:</span>'
        f'    {sources_html}'
        f'    <span style="font-family:JetBrains Mono;font-size:10px;color:#52525b;margin-left:4px;">{coverage_str}</span>'
        f'  </div>'
        f'</div>'
    )

    if item["metrics"]:
        metrics_html = "".join(f'<div class="metric-line">{m}</div>' for m in item["metrics"])
    else:
        metrics_html = '<div style="color:#52525b;font-family:JetBrains Mono;font-size:11px;">No granular metrics extracted</div>'

    if item["analyst_assessment"]:
        analysis_html = "".join(
            f'<div class="analysis-line" style="border-left:2px solid {s["color"]};padding-left:10px;margin-bottom:6px;">{a}</div>'
            for a in item["analyst_assessment"]
        )
    else:
        analysis_html = '<div style="color:#52525b;font-family:JetBrains Mono;font-size:11px;">—</div>'

    return (
        f'<div class="news-row">'
        f'  {company_html}'
        f'  {headline_html}'
        f'  <div>{metrics_html}</div>'
        f'  <div>{analysis_html}</div>'
        f'</div>'
    )


def render_sector_tab(items: list[dict]) -> None:
    if not items:
        st.markdown(
            '<div class="empty-state">NO ITEMS IN THIS SECTOR<br/>'
            '<span style="color:#3f3f46;font-size:10px;">Upload files and run extraction to populate</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    header = (
        '<div class="col-header-row">'
        '  <div>Company · Ticker</div>'
        '  <div>Headline &amp; Sources</div>'
        '  <div>Financial Metrics</div>'
        '  <div>Analyst Assessment</div>'
        '</div>'
    )
    rows = "".join(render_item_row(it, i) for i, it in enumerate(items))
    st.markdown(header + rows, unsafe_allow_html=True)


def items_to_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        rows.append({
            "Sector":   SECTOR_BY_ID[it["sector"]]["name"],
            "Company":  it["company"],
            "Ticker":   it["ticker"],
            "Headline": it["headline"],
            "Sources":  ", ".join(it["sources"]),
            "Metrics":  " | ".join(it["metrics"]),
            "Analyst Assessment": "\n• " + "\n• ".join(it["analyst_assessment"]),
            "Impact":   it["impact"],
            "Sentiment": it["sentiment"],
        })
    return pd.DataFrame(rows)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if "news_items" not in st.session_state:
    st.session_state.news_items = []
if "raw_text_preview" not in st.session_state:
    st.session_state.raw_text_preview = ""
if "extraction_run_at" not in st.session_state:
    st.session_state.extraction_run_at = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ANTHROPIC API KEY RESOLUTION                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def resolve_api_key(user_supplied: str | None) -> str | None:
    """Priority: explicit input → st.secrets → env var → None."""
    if user_supplied and user_supplied.strip():
        return user_supplied.strip()
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  UI: TICKER BAR (auto-refreshing fragment)                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@st.fragment(run_every=TICKER_REFRESH_SEC)
def ticker_fragment():
    render_ticker_bar()

ticker_fragment()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  UI: SIDEBAR                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown(
        '<div style="font-family:JetBrains Mono;font-size:13px;font-weight:800;'
        'letter-spacing:0.2em;color:#fafafa;border-bottom:1px solid #27272a;'
        'padding-bottom:10px;margin-bottom:12px;">CONTROL PANEL</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">API KEY</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-…  (or set in env / secrets)",
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">MODEL</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Claude model",
        options=["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">UPLOAD NEWSPAPER EXTRACTS</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Drop raw text extracts from any of: ET, Mint, BS, HT, FE, BL",
    )

    extract_clicked = st.button(
        "▸  EXTRACT NEWS",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if st.session_state.news_items:
        st.markdown('<div class="sidebar-label">EXPORT</div>', unsafe_allow_html=True)
        df = items_to_dataframe(st.session_state.news_items)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "▾  DOWNLOAD .CSV",
            data=csv,
            file_name=f"alphatrack_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="AlphaTrack")
        st.download_button(
            "▾  DOWNLOAD .XLSX",
            data=excel_buffer.getvalue(),
            file_name=f"alphatrack_{datetime.now():%Y%m%d_%H%M}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Stats block
    if st.session_state.news_items:
        st.markdown('<div class="sidebar-label">SESSION STATS</div>', unsafe_allow_html=True)
        total = len(st.session_state.news_items)
        high = sum(1 for it in st.session_state.news_items if it["impact"] == "HIGH")
        by_sector = {s["id"]: 0 for s in SECTORS}
        for it in st.session_state.news_items:
            by_sector[it["sector"]] += 1
        st.markdown(
            f'<div style="font-family:JetBrains Mono;font-size:11px;color:#a1a1aa;line-height:1.7;">'
            f'<span style="color:#71717a;">TOTAL ITEMS</span> &nbsp; <b style="color:#fafafa;">{total}</b><br/>'
            f'<span style="color:#71717a;">HIGH IMPACT</span> &nbsp; <b style="color:#fb7185;">{high}</b><br/>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for s in SECTORS:
            cnt = by_sector[s["id"]]
            st.markdown(
                f'<div style="font-family:JetBrains Mono;font-size:10px;color:{s["color"]};'
                f'display:flex;justify-content:space-between;padding:2px 0;">'
                f'<span>{s["icon"]} {s["name"].upper()}</span><span><b>{str(cnt).zfill(2)}</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Footer
    st.markdown(
        '<div style="position:fixed;bottom:8px;left:14px;font-family:JetBrains Mono;'
        'font-size:9px;letter-spacing:0.2em;color:#3f3f46;">'
        'ALPHATRACK · v1.0 · LOCAL'
        '</div>',
        unsafe_allow_html=True,
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EXTRACTION HANDLER                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if extract_clicked and uploaded_files:
    api_key = resolve_api_key(api_key_input)
    if not api_key:
        st.error(
            "❌ No Anthropic API key found.\n\n"
            "Provide it via the sidebar, OR set `ANTHROPIC_API_KEY` in your environment, "
            "OR add it to `.streamlit/secrets.toml`."
        )
    else:
        client = Anthropic(api_key=api_key)
        # 1. Extract & clean text from every file
        all_text_blocks = []
        with st.spinner("Reading uploaded files…"):
            for f in uploaded_files:
                try:
                    txt = clean_text(extract_text_from_upload(f))
                    if txt:
                        all_text_blocks.append(f"### SOURCE FILE: {f.name}\n\n{txt}")
                except Exception as e:
                    st.warning(f"Could not read {f.name}: {e}")
        combined = "\n\n────────────────────────────────────────\n\n".join(all_text_blocks)
        st.session_state.raw_text_preview = combined[:1500]

        if not combined.strip():
            st.error("Files contained no extractable text.")
        else:
            with st.spinner(f"Extracting structured news with {selected_model}…"):
                try:
                    parsed = call_claude_extraction(client, selected_model, combined)
                    raw_items = parsed.get("items", [])
                    normalized = [normalize_item(it) for it in raw_items]
                    normalized = [it for it in normalized if it is not None]
                    # Stable sort: HIGH → MED → LOW
                    rank = {"HIGH": 0, "MED": 1, "LOW": 2}
                    normalized.sort(key=lambda x: rank.get(x["impact"], 3))
                    st.session_state.news_items = normalized
                    st.session_state.extraction_run_at = datetime.now()
                    st.session_state.last_error = None
                    st.success(
                        f"✓ Extracted {len(normalized)} item(s) from {len(uploaded_files)} file(s)."
                    )
                    st.rerun()
                except APIError as e:
                    st.session_state.last_error = f"Anthropic API error: {e}"
                    st.error(st.session_state.last_error)
                except Exception as e:
                    st.session_state.last_error = f"Extraction failed: {e}"
                    st.error(st.session_state.last_error)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN VIEW: SECTOR TABS                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if st.session_state.extraction_run_at:
    n = len(st.session_state.news_items)
    t_str = st.session_state.extraction_run_at.strftime("%H:%M:%S")
    sub = f"{n} ITEMS · EXTRACTED AT {t_str} IST · MODEL: {selected_model.upper()}"
else:
    sub = "AWAITING UPLOAD · USE THE LEFT PANEL TO LOAD NEWSPAPER FILES"
st.markdown(f'<div class="sub-header">{sub}</div>', unsafe_allow_html=True)

# Build tab list: "ALL" first, then each sector
tab_labels = [f"ALL · {len(st.session_state.news_items):02d}"] + [
    f"{s['icon']} {s['name'].split('/')[0].strip().upper()} · "
    f"{sum(1 for it in st.session_state.news_items if it['sector']==s['id']):02d}"
    for s in SECTORS
]
tabs = st.tabs(tab_labels)

with tabs[0]:
    render_sector_tab(st.session_state.news_items)

for i, sector in enumerate(SECTORS, start=1):
    with tabs[i]:
        items = [it for it in st.session_state.news_items if it["sector"] == sector["id"]]
        render_sector_tab(items)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RAW TEXT PREVIEW (collapsible, debugging aid)                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if st.session_state.raw_text_preview:
    with st.expander("⌗  Raw text preview (first 1,500 chars)", expanded=False):
        st.text(st.session_state.raw_text_preview)
