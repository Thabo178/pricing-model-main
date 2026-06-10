"""
Structured Note Pricer — Streamlit Dashboard

Launch: double-click start.command (Mac) or start.bat (Windows)
        or run: streamlit run dashboard.py
"""

import calendar
import json
import numpy as np
from datetime import date
from pathlib import Path
from pricer import price_worst_of, price_note_dict
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config + custom CSS
# ---------------------------------------------------------------------------

import base64
from io import BytesIO

def _hc_favicon_b64() -> str:
    from PIL import Image, ImageDraw, ImageFont
    img  = Image.new("RGBA", (64, 64), (30, 58, 95, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), "HC", font=font)
    x = (64 - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (64 - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), "HC", fill=(255, 255, 255, 255), font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

_FAVICON = _hc_favicon_b64()

st.set_page_config(
    page_title="Structured Note Pricer | Ryan Hysmith",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f'<link rel="shortcut icon" type="image/png" href="data:image/png;base64,{_FAVICON}">',
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #ffffff;
    color: #1a1a2e;
    font-size: 15px;
}

/* ── Shrink page margins ── */
.main .block-container {
    padding: 0.75rem 2rem 1rem;
    max-width: 100%;
}

/* ── Streamlit top toolbar — remove bottom padding that hides the title ── */
.st-emotion-cache-12fmjuu {
    padding-bottom: 0 !important;
}

/* ── Tighten vertical spacing between widgets ── */
.stVerticalBlock > div {
    gap: 0.35rem;
}
div[data-testid="column"] > div[data-testid="stVerticalBlock"] > div {
    gap: 0.35rem;
}

/* ── Tab bar ── */
[data-baseweb="tab-list"] {
    border-bottom: 2px solid #e2e8f0;
    gap: 0;
    margin-bottom: 0.75rem;
}
[data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 500;
    color: #64748b;
    padding: 0.5rem 1.1rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #1e3a5f;
    border-bottom-color: #1e3a5f;
    font-weight: 700;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.7rem 1rem;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
}
[data-testid="stMetricValue"] {
    font-size: 1.45rem;
    font-weight: 700;
    color: #1e3a5f;
}

/* ── Primary buttons ── */
[kind="primary"] button {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    font-size: 0.9rem !important;
}
[kind="primary"] button:hover {
    background-color: #2d5282 !important;
}

/* ── Dividers ── */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 0.6rem 0;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* ── Headings ── */
h1 { font-size: 1.5rem !important; margin-bottom: 0 !important; }
h2 { font-size: 1.15rem !important; color: #1e3a5f; font-weight: 700; margin: 0.4rem 0 0.2rem; }
h3 { font-size: 1rem !important; color: #1e3a5f; font-weight: 700; margin: 0.3rem 0 0.1rem; }
h4, h5 { font-size: 0.9rem !important; color: #1e3a5f; font-weight: 700; margin: 0.3rem 0 0.1rem; }

/* ── Widget labels ── */
[data-testid="stWidgetLabel"] p {
    font-size: 0.85rem;
    font-weight: 500;
    color: #374151;
}

/* ── Smaller select/input boxes ── */
[data-baseweb="input"] input,
[data-baseweb="select"] div {
    font-size: 0.875rem !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"] p {
    color: #64748b;
    font-size: 0.82rem;
}

/* ── Info / alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
}

/* ── Expanders ── */
[data-testid="stExpander"] summary {
    font-weight: 600;
    font-size: 0.88rem;
    color: #1e3a5f;
}

/* ── Vertical divider between input cols and results ── */
.results-panel {
    border-left: 1px solid #e2e8f0;
    padding-left: 1.5rem;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _safe_note_filename(prefix: str, tickers: list[str]) -> str:
    joined = "_".join(t.upper() for t in tickers)
    return f"{prefix}_{joined}.json"


def _save_note_json(note_dict: dict, filename: str) -> Path:
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(note_dict, f, indent=2)
    return path

def _list_saved_worstof_files() -> list[str]:
    if not DATA_DIR.exists():
        return []
    files = []
    for p in sorted(DATA_DIR.glob("worstof_note_*.json")):
        files.append(p.name)
    return files


def _load_note_json(filename: str) -> dict | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _quote_summary_text(
    note_dict: dict,
    wo_result: dict,
    single_res: dict | None = None,
    single_name: str | None = None,
) -> str:
    underliers = note_dict.get("underliers", [])
    basket = "/".join(underliers) if underliers else "N/A"
    fv = float(note_dict.get("face_value", 1000.0))
    npv_pct = float(wo_result.get("npv_pct", 0.0))
    npv_dollar = float(wo_result.get("npv_dollar", 0.0))
    se_bps = float(wo_result.get("se_bps", 0.0))
    discount_to_par = 100.0 - npv_pct

    lines = [
        f"Worst-of Phoenix quote: {basket}",
        f"Fair value: {npv_pct:.2f}% of face (${npv_dollar:.2f} per ${fv:,.0f} face)",
        f"Discount to par: {discount_to_par:.2f} points",
        f"Coupon: {100.0 * float(note_dict.get('coupon_rate', 0.0)):.2f}%",
        f"Autocall / coupon / knock-in barriers: "
        f"{100.0 * float(note_dict.get('autocall_barrier', 0.0)):.0f}% / "
        f"{100.0 * float(note_dict.get('coupon_barrier', 0.0)):.0f}% / "
        f"{100.0 * float(note_dict.get('knockin_barrier', 0.0)):.0f}%",
        f"Issue: {note_dict.get('issue_date', '—')} | Maturity: {note_dict.get('maturity_date', '—')}",
        f"MC standard error: ±{se_bps:.1f} bps",
    ]

    if single_res is not None and single_name:
        single_pct = float(single_res.get("npv_pct", 0.0))
        diff = single_pct - npv_pct
        lines.append(
            f"Vs single-name {single_name}: worst-of is {diff:.2f} points cheaper "
            f"({single_pct:.2f}% vs {npv_pct:.2f}%)."
        )

    return "\n".join(lines)

UNDERLIERS = [
    "NVDA",
    "TSLA",
    "AMD",
    "META",
    "GOOGL",
    "AMZN",
    "HOOD",
    "LULU",
    "NOW",
    "PLTR",
    "WFC",
    "SPY",
]

DEFAULT_SPOTS = {
    "NVDA": 219.16,
    "TSLA": 180.0,
    "AMD": 160.0,
    "META": 510.0,
    "GOOGL": 175.0,
    "AMZN": 190.0,
    "HOOD": 22.0,
    "LULU": 85.0,
    "NOW": 820.0,
    "PLTR": 25.0,
    "WFC": 57.0,
    "SPY": 525.0,
}

CALIBRATED_DIR = Path("data") / "calibrated"


def _calibrated_file_path(ticker: str) -> Path:
    return CALIBRATED_DIR / f"{ticker.upper()}.json"


def _load_calibrated_params(ticker: str) -> dict | None:
    path = _calibrated_file_path(ticker)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fmt_num(x, decimals=4):
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"


def _fmt_pct(x, decimals=2):
    try:
        return f"{100.0 * float(x):.{decimals}f}%"
    except Exception:
        return "—"

def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _generate_obs_dates(issue: date, maturity: date, freq: str) -> list:
    step = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[freq]
    dates, cur = [], _add_months(issue, step)
    while cur <= maturity:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur = _add_months(cur, step)
    return dates


def _build_note_dict(
    underlier,
    spot,
    face_value,
    issue_date,
    maturity_date,
    obs_freq,
    autocall_pct,
    coupon_pct,
    knockin_pct,
    coupon_rate,
    rfr,
    credit_spread_bps=0,
) -> dict:
    note = {
        "underlier": underlier,
        "spot": spot,
        "face_value": face_value,
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "maturity_date": maturity_date.strftime("%Y-%m-%d"),
        "observation_dates": _generate_obs_dates(issue_date, maturity_date, obs_freq),
        "autocall_barrier": autocall_pct / 100,
        "coupon_barrier": coupon_pct / 100,
        "knockin_barrier": knockin_pct / 100,
        "coupon_rate": coupon_rate / 100,
        "risk_free_rate": rfr / 100,
    }
    if credit_spread_bps:
        note["credit_spread"] = credit_spread_bps / 10000
    return note


def _build_wo_note_dict(
    tickers,
    spots,
    corr_matrix,
    face_value,
    issue_date,
    maturity_date,
    obs_freq,
    autocall_pct,
    coupon_pct,
    knockin_pct,
    coupon_rate,
    rfr,
    credit_spread_bps=0,
) -> dict:
    note = {
        "underliers": tickers,
        "spots": spots,
        "correlation_matrix": corr_matrix,
        "face_value": face_value,
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "maturity_date": maturity_date.strftime("%Y-%m-%d"),
        "observation_dates": _generate_obs_dates(issue_date, maturity_date, obs_freq),
        "autocall_barrier": autocall_pct / 100,
        "coupon_barrier": coupon_pct / 100,
        "knockin_barrier": knockin_pct / 100,
        "coupon_rate": coupon_rate / 100,
        "risk_free_rate": rfr / 100,
    }
    if credit_spread_bps:
        note["credit_spread"] = credit_spread_bps / 10000
    return note

def _build_corr_matrix(pairs: dict, n: int) -> np.ndarray:
    """
    Build a symmetric n×n correlation matrix from a dict of pairwise entries.
    pairs keys: (i, j) with i < j, values: float in [-1, 1].
    """
    mat = np.eye(n, dtype=float)
    for (i, j), rho in pairs.items():
        mat[i, j] = rho
        mat[j, i] = rho
    return mat

def _recommendation_badge(rec: str) -> str:
    colour = {"Buy": "#16a34a", "Skip": "#dc2626", "Gray Zone": "#d97706"}.get(
        rec, "#64748b"
    )
    return (
        f'<span style="background:{colour};color:#fff;padding:4px 14px;'
        f'border-radius:4px;font-weight:700;font-size:1rem;">{rec}</span>'
    )


def _label(text):
    st.markdown(f'<p class="section-label">{text}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='color:#1e3a5f;font-weight:800;letter-spacing:-0.02em;margin-top:0.5rem;'>"
    "Structured Note Pricer</h1>"
    "<p style='color:#94a3b8;font-size:0.82rem;margin-top:0.1rem;margin-bottom:0.5rem;'>"
    "Phoenix Autocallable &nbsp;·&nbsp; Single-Asset &amp; Worst-Of &nbsp;·&nbsp; "
    "Heston Stochastic Volatility &nbsp;·&nbsp; ORATS Live Data</p>",
    unsafe_allow_html=True,
)

tab_price, tab_vol, tab_cal, tab_wo, tab_port, tab_offer = st.tabs(
    [
        "Note Pricer",
        "Vol Surface",
        "Calibration",
        "Worst-Of Pricer",
        "Portfolio",
        "Offering Evaluator",
    ]
)

# ===========================================================================
# TAB 1 — NOTE PRICER
# ===========================================================================

with tab_price:
    col_a, col_b, col_out = st.columns([1, 1, 1.6])

    # ── Column A: Underlier & dates ──────────────────────────────────────
    with col_a:
        _label("Underlier")
        underlier = st.selectbox(
            "Ticker", UNDERLIERS, key="p_underlier", label_visibility="collapsed"
        )

        sp_c1, sp_c2 = st.columns(2)
        with sp_c1:
            spot = st.number_input(
                "Spot ($)",
                min_value=1.0,
                value=DEFAULT_SPOTS.get(underlier, 100.0),
                step=0.01,
                format="%.2f",
                key=f"p_spot_{underlier}",
            )
        with sp_c2:
            face_value = st.number_input(
                "Face Value ($)",
                min_value=100.0,
                value=1000.0,
                step=100.0,
                key="p_face",
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        _label("Schedule")

        d1, d2 = st.columns(2)
        with d1:
            issue_date = st.date_input(
                "Issue Date", value=date(2026, 6, 3), key="p_issue"
            )
        with d2:
            maturity_date = st.date_input(
                "Maturity Date", value=date(2027, 12, 3), key="p_mat"
            )

        obs_freq = st.selectbox(
            "Observation Frequency",
            ["Quarterly", "Monthly", "Semi-Annual"],
            key="p_freq",
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        _label("Model Settings")

        m1, m2 = st.columns(2)
        with m1:
            memory_on = st.checkbox(
                "Memory Coupon",
                value=False,
                key="p_memory",
                help="Unpaid coupons accrue and are paid at the next qualifying observation.",
            )
        with m2:
            n_paths = st.select_slider(
                "MC Paths",
                options=[10_000, 50_000, 100_000],
                value=50_000,
                format_func=lambda x: f"{x:,}",
                key="p_npaths",
            )

    # ── Column B: Barriers & rates ───────────────────────────────────────
    with col_b:
        _label("Barrier Structure")
        autocall_pct = st.slider(
            "Autocall Barrier", 80, 115, 100, step=5, format="%d%%", key="p_autocall"
        )
        coupon_pct = st.slider(
            "Coupon Barrier", 50, 95, 75, step=5, format="%d%%", key="p_coupon"
        )
        knockin_pct = st.slider(
            "Knock-In Barrier", 40, 80, 65, step=5, format="%d%%", key="p_knockin"
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        _label("Rates")

        r1, r2 = st.columns(2)
        with r1:
            coupon_rate = st.number_input(
                "Annual Coupon (%)",
                min_value=0.0,
                max_value=50.0,
                value=12.0,
                step=0.5,
                key="p_cpn",
            )
        with r2:
            rfr = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=3.75,
                step=0.25,
                key="p_rfr",
            )

        credit_spread_bps = st.number_input(
            "Issuer Credit Spread (bps)",
            min_value=0,
            max_value=500,
            value=0,
            step=5,
            help="§6.1 — Treasury + CDS spread. Added to discount rate. Typical A-rated issuer: 50–150 bps.",
            key="p_cs",
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        price_btn = st.button(
            "Run Pricing", type="primary", use_container_width=True, key="p_btn"
        )

    # ── Column C: Results ────────────────────────────────────────────────
    with col_out:
        st.markdown('<div class="results-panel">', unsafe_allow_html=True)
        _label("Results")

        if price_btn:
            note_dict = _build_note_dict(
                underlier,
                spot,
                face_value,
                issue_date,
                maturity_date,
                obs_freq,
                autocall_pct,
                coupon_pct,
                knockin_pct,
                coupon_rate,
                rfr,
                credit_spread_bps=credit_spread_bps,
            )
            note_dict["memory"] = memory_on
            if not note_dict["observation_dates"]:
                st.error(
                    "No observation dates — check that Maturity Date is after Issue Date."
                )
            else:
                with st.spinner(f"Pricing {underlier} · {n_paths:,} paths …"):
                    try:
                        from pricer.pricer import price_note_dict

                        result = price_note_dict(
                            note_dict, n_paths=n_paths, memory=memory_on
                        )
                        st.session_state["last_result"] = result
                        st.session_state["last_note"] = note_dict
                    except Exception as e:
                        st.error(f"Pricing failed: {e}")
                        st.session_state.pop("last_result", None)

        if "last_result" in st.session_state:
            r = st.session_state["last_result"]
            n = st.session_state["last_note"]

            rm1, rm2, rm3 = st.columns(3)
            rm1.metric("Fair Value", f"{r['npv_pct']:.2f}%")
            rm2.metric("Dollar FV", f"${r['npv_dollar']:,.2f}")
            rm3.metric("MC Std Err", f"±{r['se_bps']:.1f} bps")

            if credit_spread_bps:
                st.info(
                    f"Credit spread of {credit_spread_bps} bps applied to discount curve (§6.1).",
                    icon="ℹ️",
                )

            with st.expander("Greeks  (§6.3)", expanded=False):
                g_paths = st.select_slider(
                    "Paths for Greeks",
                    options=[10_000, 20_000, 30_000],
                    value=20_000,
                    format_func=lambda x: f"{x:,}",
                    key="p_g_paths",
                )
                if st.button("Compute Greeks", key="p_greeks_btn"):
                    with st.spinner("Computing Greeks (5 reprice calls) …"):
                        try:
                            from pricer.greeks import compute_greeks

                            st.session_state["last_greeks"] = compute_greeks(
                                n, n_paths=g_paths
                            )
                        except Exception as e:
                            st.error(f"Greeks failed: {e}")

                if "last_greeks" in st.session_state:
                    g = st.session_state["last_greeks"]
                    gc1, gc2, gc3, gc4 = st.columns(4)
                    gc1.metric("Δ (% / 1% spot)", f"{g['delta_pct']:+.3f}%")
                    gc2.metric("Δ ($ / $1 spot)", f"${g['delta_dollar']:+.4f}")
                    gc3.metric("ν (% / 1 vol pt)", f"{g['vega_pct']:+.3f}%")
                    gc4.metric("Θ ($ / day)", f"${g['theta_dollar']:+.4f}")
                    st.caption(
                        f"Gamma: ${g['gamma_dollar']:+.6f} per $1² · ±1% central-difference"
                    )

            st.markdown("<hr>", unsafe_allow_html=True)
            _label("Term Sheet Summary")

            obs = n["observation_dates"]
            cs = n.get("credit_spread", 0)
            summary = {
                "Underlier": n["underlier"],
                "Spot / Face": f"${n['spot']:,.2f}  /  ${n['face_value']:,.0f}",
                "Dates": f"{n['issue_date']} → {n['maturity_date']}",
                "Observations": f"{len(obs)} ({obs[0]} → {obs[-1]})",
                "Autocall / Coupon / KI": (
                    f"{n['autocall_barrier']*100:.0f}%  /  "
                    f"{n['coupon_barrier']*100:.0f}%  /  "
                    f"{n['knockin_barrier']*100:.0f}%"
                ),
                "Coupon / RFR": f"{n['coupon_rate']*100:.2f}%  /  {n['risk_free_rate']*100:.3f}%",
                "Credit Spread": f"{cs*10000:.0f} bps" if cs else "None",
                "Memory Coupon": "Yes" if n.get("memory") else "No",
                "MC Paths": f"{r['n_paths']:,}",
            }
            st.dataframe(
                pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]),
                use_container_width=True,
            )
        else:
            st.info(
                "Configure the term sheet in the columns on the left, then click **Run Pricing**."
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# TAB 2 — VOL SURFACE
# ===========================================================================

with tab_vol:
    col_v1, col_v2 = st.columns([0.22, 0.78])

    with col_v1:
        _label("Ticker")
        vol_ticker = st.selectbox(
            "Ticker", UNDERLIERS, key="v_ticker", label_visibility="collapsed"
        )
        fetch_btn = st.button(
            "Fetch from ORATS", type="primary", key="v_fetch", use_container_width=True
        )

    with col_v2:
        if fetch_btn:
            with st.spinner(f"Fetching {vol_ticker} from ORATS …"):
                try:
                    from pricer.orats import get_monies_implied, get_smv_summary

                    st.session_state["vol_mono"] = get_monies_implied(vol_ticker)
                    st.session_state["vol_smv"] = get_smv_summary(vol_ticker)
                    st.session_state["vol_ticker"] = vol_ticker
                except Exception as e:
                    st.error(f"ORATS fetch failed: {e}")

        if (
            st.session_state.get("vol_ticker") == vol_ticker
            and "vol_mono" in st.session_state
        ):
            df = st.session_state["vol_mono"]
            smv = st.session_state["vol_smv"]

            if not smv.empty:
                row = smv.iloc[0]
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Spot", f"${float(row.get('stockPrice', 0)):,.2f}")
                s2.metric("30d ATM IV", f"{float(row.get('iv30d', 0))*100:.1f}%")
                s3.metric("Skew (30d)", f"{float(row.get('rSlp30', 0)):.3f}")
                s4.metric(
                    "Implied Move", f"{float(row.get('impliedMove', 0))*100:.1f}%"
                )

            st.markdown("<hr>", unsafe_allow_html=True)

            want = ["expirDate", "atmiv", "slope", "vol25", "vol50", "vol75", "vol5"]
            show = [c for c in want if c in df.columns]
            disp = df[show].copy()
            disp.rename(
                columns={
                    "expirDate": "Expiry",
                    "atmiv": "ATM IV",
                    "slope": "Skew",
                    "vol25": "25Δ IV",
                    "vol50": "50Δ IV",
                    "vol75": "75Δ IV",
                    "vol5": "5Δ IV",
                },
                inplace=True,
            )
            for col in ["ATM IV", "25Δ IV", "50Δ IV", "75Δ IV", "5Δ IV"]:
                if col in disp.columns:
                    disp[col] = (disp[col].astype(float) * 100).round(2).astype(
                        str
                    ) + "%"
            st.dataframe(disp, use_container_width=True, hide_index=True)

            if "expirDate" in df.columns and "atmiv" in df.columns:
                chart_df = df[["expirDate", "atmiv"]].copy()
                chart_df["ATM IV (%)"] = chart_df["atmiv"].astype(float) * 100
                st.line_chart(
                    chart_df.set_index("expirDate")[["ATM IV (%)"]],
                    use_container_width=True,
                )
                st.caption(
                    "ATM Implied Volatility (%) by Expiry — source: ORATS /monies/implied"
                )
        else:
            st.info(
                "Select a ticker and click **Fetch from ORATS** to load the live vol surface."
            )

# ===========================================================================
# TAB 3 — CALIBRATION
# ===========================================================================

with tab_cal:
    col_c1, col_c2 = st.columns([0.32, 0.68])

    with col_c1:
        _label("Underlier")
        cal_ticker = st.selectbox(
            "Ticker", UNDERLIERS, key="c_ticker", label_visibility="collapsed"
        )

        latest_saved = _load_calibrated_params(cal_ticker)

        cc1, cc2 = st.columns(2)
        with cc1:
            cal_spot = st.number_input(
                "Spot ($)",
                value=float(DEFAULT_SPOTS.get(cal_ticker, 100.0)),
                min_value=1.0,
                step=1.0,
                key=f"c_spot_{cal_ticker}",
            )
        with cc2:
            cal_rfr = st.number_input("RFR (%)", value=3.75, step=0.25, key="c_rfr")

        use_orats = st.checkbox("Use live ORATS surface", value=True, key="c_orats")
        if not use_orats:
            st.caption("Mock mode — calibrates to a synthetic surface (RMSE ≈ 0).")

        cal_btn = st.button(
            "Run Calibration", type="primary", use_container_width=True, key="c_btn"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        if latest_saved is None:
            st.caption(f"No saved calibration file found yet for {cal_ticker}.")
        else:
            saved_mode = latest_saved.get("calibration_mode", "legacy")
            saved_rmse = latest_saved.get("rmse")
            saved_points = latest_saved.get("n_points", "—")
            st.caption(
                f"Latest saved: **{saved_mode}** · RMSE **{_fmt_num(saved_rmse, 5)}** · "
                f"{saved_points} points"
            )

        st.caption(
            "DE global search → L-BFGS-B polish (§8.1).  \n"
            "Bounds: κ [0.5,8] · θ [0.01,0.5] · σ [0.1,2.5] · ρ [−0.95,−0.1] · v₀ [0.01,0.6]"
        )

    with col_c2:
        if cal_btn:
            with st.spinner(f"Calibrating {cal_ticker} …"):
                try:
                    import QuantLib as ql

                    today = ql.Date.todaysDate()
                    rfr = cal_rfr / 100.0

                    if use_orats:
                        from pricer.orats import build_calibration_set, live_spot
                        from pricer.calibration import (
                            calibrate_heston_orats,
                            save_calibrated,
                        )

                        used_fallback_spot = False
                        try:
                            spot = live_spot(cal_ticker)
                        except Exception:
                            spot = cal_spot
                            used_fallback_spot = True

                        cal_set = build_calibration_set(cal_ticker, today, spot, r=rfr)
                        if not cal_set:
                            st.error(f"No ORATS data for {cal_ticker}.")
                        else:
                            result = calibrate_heston_orats(
                                cal_ticker, today, spot, rfr, cal_set
                            )
                            result["spot_used"] = float(spot)
                            result["used_fallback_spot"] = used_fallback_spot
                            result["calibration_mode"] = "live_orats"
                            save_calibrated(result)
                            st.session_state["cal_result"] = result
                    else:
                        from pricer.calibration import (
                            generate_mock_surface,
                            calibrate_heston,
                            save_calibrated,
                        )

                        surface = generate_mock_surface(
                            cal_ticker, cal_spot, rfr, today=today
                        )
                        result = calibrate_heston(
                            cal_ticker, cal_spot, rfr, surface, today=today
                        )
                        result["spot_used"] = float(cal_spot)
                        result["used_fallback_spot"] = True
                        result["calibration_mode"] = "mock_surface"
                        save_calibrated(result)
                        st.session_state["cal_result"] = result

                except Exception as e:
                    st.error(f"Calibration failed: {e}")

        display_result = st.session_state.get("cal_result")
        if display_result is None or display_result.get("underlier") != cal_ticker:
            display_result = _load_calibrated_params(cal_ticker)

        if display_result:
            r = display_result

            mode_label = r.get("calibration_mode", "legacy")
            live_or_fallback = (
                "Fallback Spot" if r.get("used_fallback_spot") else "Live Spot"
            )
            feller_ok = r.get("feller_satisfied", True)

            hdr1, hdr2, hdr3, hdr4 = st.columns(4)
            hdr1.metric("Mode", mode_label)
            hdr2.metric("RMSE", _fmt_num(r.get("rmse"), 6))
            hdr3.metric("IV Points", str(r.get("n_points", "—")))
            hdr4.metric("Spot Source", live_or_fallback)

            if "spot_used" in r:
                st.caption(f"Spot used in calibration: **${float(r['spot_used']):,.2f}**")

            if not feller_ok:
                st.warning(
                    f"Feller condition violated: 2κθ = {2*r['kappa']*r['theta']:.4f} < σ² = {r['sigma']**2:.4f}. "
                    "Full-truncation MC handles this numerically — result is still valid. (§8.2)",
                    icon="⚠️",
                )
            else:
                st.success("Feller condition satisfied.")

            st.markdown("<hr>", unsafe_allow_html=True)
            _label("Calibrated Parameters")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("v₀", _fmt_num(r.get("v0"), 6))
            m2.metric("κ", _fmt_num(r.get("kappa"), 6))
            m3.metric("θ", _fmt_num(r.get("theta"), 6))
            m4.metric("σ", _fmt_num(r.get("sigma"), 6))
            m5.metric("ρ", _fmt_num(r.get("rho"), 6))

            params = {
                "v₀  (initial variance)": _fmt_num(r.get("v0"), 6),
                "κ   (mean-reversion speed)": _fmt_num(r.get("kappa"), 6),
                "θ   (long-run variance)": _fmt_num(r.get("theta"), 6),
                "σ   (vol of vol)": _fmt_num(r.get("sigma"), 6),
                "ρ   (asset / vol corr.)": _fmt_num(r.get("rho"), 6),
            }

            st.dataframe(
                pd.DataFrame.from_dict(params, orient="index", columns=["Value"]),
                use_container_width=True,
            )
            st.caption(f"Saved → data/calibrated/{cal_ticker}.json")

            with st.expander("Raw calibrated JSON"):
                st.json(r)

        else:
            st.info("Select a ticker and click **Run Calibration**.")

# ===========================================================================
# TAB 4 — WORST-OF PRICER
# ===========================================================================

with tab_wo:
    st.subheader("Worst-Of Phoenix Pricer")

    wo_col_left, wo_col_right = st.columns([0.45, 0.55])

    # -----------------------------
    # Left: basket, dates, barriers
    # -----------------------------
    with wo_col_left:
        _label("Load saved worst-of note")

        saved_note_files = _list_saved_worstof_files()
        load_cols = st.columns([0.75, 0.25])

        with load_cols[0]:
            selected_saved_note = st.selectbox(
                "Saved worst-of JSON",
                options=["—"] + saved_note_files,
                key="wo_saved_note_select",
            )

        with load_cols[1]:
            load_saved_btn = st.button(
                "Load",
                use_container_width=True,
                key="wo_load_saved_btn",
            )

        if load_saved_btn and selected_saved_note != "—":
            loaded_note = _load_note_json(selected_saved_note)
            if loaded_note is None:
                st.error(f"Could not load {selected_saved_note}.")
            else:
                st.session_state["wo_loaded_note"] = loaded_note
                st.success(f"Loaded {selected_saved_note}")
        _label("Underliers (2–3 names)")
        loaded_note = st.session_state.get("wo_loaded_note", {})

        default_selected = loaded_note.get("underliers", ["NVDA", "TSLA"])
        selected = st.multiselect(
            "Select underliers",
            UNDERLIERS,
            default=default_selected,
            max_selections=3,
        )

        if len(selected) < 2:
            st.warning("Select at least two underliers to define a worst-of basket.")
        elif len(selected) > 3:
            st.error("Worst-of basket currently supports at most 3 underliers.")
        else:
            n_assets = len(selected)
            spots = []
            st.caption("Spots")
            spot_cols = st.columns(n_assets)
            loaded_spots = loaded_note.get("spots", [])
            spots = []
            st.caption("Spots")
            spot_cols = st.columns(n_assets)
            for i, ticker in enumerate(selected):
                default_spot = (
                    float(loaded_spots[i])
                    if i < len(loaded_spots)
                    else float(DEFAULT_SPOTS.get(ticker, 100.0))
                )
                with spot_cols[i]:
                    spot_val = st.number_input(
                        ticker,
                        min_value=1.0,
                        value=default_spot,
                        step=1.0,
                        key=f"wo_spot_{ticker}",
                    )
                    spots.append(spot_val)

            _label("Dates")
            today = date.today()
            default_issue = today
            default_maturity = date(today.year + 1, today.month, today.day)

            if loaded_note:
                try:
                    default_issue = date.fromisoformat(loaded_note.get("issue_date"))
                    default_maturity = date.fromisoformat(loaded_note.get("maturity_date"))
                except Exception:
                    pass

            issue_date = st.date_input("Issue date", value=default_issue, key="wo_issue")
            maturity_date = st.date_input(
                "Maturity date",
                value=default_maturity,
                key="wo_maturity",
            )

            obs_lookup = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}
            obs_dates = loaded_note.get("observation_dates", [])
            default_freq = "Quarterly"
            if loaded_note and len(obs_dates) >= 2:
                default_freq = loaded_note.get("_ui_obs_freq", "Quarterly")

            obs_freq = st.selectbox(
                "Observation frequency",
                ["Monthly", "Quarterly", "Semi-Annual"],
                index=["Monthly", "Quarterly", "Semi-Annual"].index(default_freq),
                key="wo_freq",
            )

            _label("Barriers & coupon")
            default_autocall = 100.0 * float(loaded_note.get("autocall_barrier", 1.00)) if loaded_note else 100.0
            default_coupon_barrier = 100.0 * float(loaded_note.get("coupon_barrier", 0.75)) if loaded_note else 75.0
            default_knockin = 100.0 * float(loaded_note.get("knockin_barrier", 0.65)) if loaded_note else 65.0
            default_coupon = 100.0 * float(loaded_note.get("coupon_rate", 0.12)) if loaded_note else 12.0
            default_rfr = 100.0 * float(loaded_note.get("risk_free_rate", 0.0375)) if loaded_note else 3.75
            default_face = float(loaded_note.get("face_value", 1000.0)) if loaded_note else 1000.0


            value=default_autocall
            value=default_coupon_barrier
            value=default_knockin
            value=default_coupon
            value=default_rfr
            value=default_face

    # -----------------------------
    # Right: correlation + pricing
    # -----------------------------
    with wo_col_right:
        if len(selected) < 2 or len(selected) > 3:
            st.info("Configure a 2–3 name basket on the left to enable pricing.")
        else:
            n_assets = len(selected)
            _label("Correlation matrix (pairwise entries)")

            pair_keys = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    pair_keys.append((i, j))

            pair_rhos = {}
            corr_cols = st.columns(len(pair_keys))
            for idx, (i, j) in enumerate(pair_keys):
                label = f"ρ({selected[i]},{selected[j]})"
                with corr_cols[idx]:
                    loaded_corr = loaded_note.get("correlation_matrix", [])
                    if (
                        loaded_corr
                        and i < len(loaded_corr)
                        and j < len(loaded_corr[i])
                    ):
                        default_rho = float(loaded_corr[i][j])
                    else:
                        default_rho = 0.55 if {"NVDA", "TSLA"} <= set(selected) else 0.50
                    rho_val = st.number_input(
                        label,
                        min_value=-0.99,
                        max_value=0.99,
                        value=default_rho,
                        step=0.05,
                        key=f"wo_rho_{i}_{j}",
                    )
                    pair_rhos[(i, j)] = rho_val

            corr_matrix = _build_corr_matrix(pair_rhos, n_assets)

            st.markdown("<hr>", unsafe_allow_html=True)
            _label("Correlation Scenario Sweep")

            sweep_enabled = st.checkbox(
                "Show fair value at low / base / high correlation",
                value=True,
                key="wo_sweep_enabled",
            )

            sweep_matrices = []
            if sweep_enabled:
                if n_assets == 2:
                    base_rho = pair_rhos[(0, 1)]
                    sweep_values = [
                        max(-0.99, round(base_rho - 0.25, 2)),
                        round(base_rho, 2),
                        min(0.99, round(base_rho + 0.25, 2)),
                    ]
                    for rho in sweep_values:
                        sweep_matrices.append(
                            (f"ρ={rho:.2f}", np.array([[1.0, rho], [rho, 1.0]]))
                        )
                else:
                    avg_rho = float(np.mean(list(pair_rhos.values())))
                    sweep_values = [
                        max(-0.99, round(avg_rho - 0.20, 2)),
                        round(avg_rho, 2),
                        min(0.99, round(avg_rho + 0.20, 2)),
                    ]
                    for rho in sweep_values:
                        mat = np.full((n_assets, n_assets), rho, dtype=float)
                        np.fill_diagonal(mat, 1.0)
                        sweep_matrices.append((f"avg ρ={rho:.2f}", mat))

                st.caption(
                    "This sweep is a quick scenario tool. For 3-asset baskets it applies the same "
                    "average off-diagonal correlation to all pairs."
                )

            st.caption(
                "The engine will project this matrix to the nearest PSD correlation "
                "matrix before simulation (§10)."
            )

            errors = []
            if maturity_date <= issue_date:
                errors.append("Maturity date must be after issue date.")
            if knockin_pct > coupon_pct or coupon_pct > autocall_pct:
                errors.append(
                    "Barriers must satisfy: knock-in ≤ coupon ≤ autocall (in % of initial)."
                )

            if errors:
                for msg in errors:
                    st.error(msg)
                st.stop()

            note_dict = _build_wo_note_dict(
                tickers=selected,
                spots=spots,
                corr_matrix=corr_matrix.tolist(),
                face_value=face_value,
                issue_date=issue_date,
                maturity_date=maturity_date,
                obs_freq=obs_freq,
                autocall_pct=autocall_pct,
                coupon_pct=coupon_pct,
                knockin_pct=knockin_pct,
                coupon_rate=coupon_rate,
                rfr=rfr,
            )

            action_col1, action_col2 = st.columns(2)
            with action_col1:
                price_btn = st.button(
                    "Price worst-of note",
                    type="primary",
                    use_container_width=True,
                    key="wo_price_btn",
                )
            with action_col2:
                save_btn = st.button(
                    "Save note JSON",
                    use_container_width=True,
                    key="wo_save_btn",
                )

            if save_btn:
                try:
                    filename = _safe_note_filename("worstof_note", selected)
                    saved_path = _save_note_json(note_dict, filename)
                    st.success(f"Saved note JSON → {saved_path.as_posix()}")
                except Exception as e:
                    st.error(f"Failed to save note JSON: {e}")

            if price_btn:
                from pricer import price_worst_of

                with st.spinner("Running multi-asset Monte Carlo …"):
                    try:
                        result = price_worst_of(note_dict, n_paths=50_000)
                        st.session_state["wo_result"] = result
                        st.session_state["wo_note_dict"] = note_dict
                    except Exception as e:
                        st.error(f"Pricing failed: {e}")

                if sweep_enabled:
                    sweep_rows = []
                    for label, sweep_mat in sweep_matrices:
                        try:
                            sweep_note = _build_wo_note_dict(
                                tickers=selected,
                                spots=spots,
                                corr_matrix=sweep_mat.tolist(),
                                face_value=face_value,
                                issue_date=issue_date,
                                maturity_date=maturity_date,
                                obs_freq=obs_freq,
                                autocall_pct=autocall_pct,
                                coupon_pct=coupon_pct,
                                knockin_pct=knockin_pct,
                                coupon_rate=coupon_rate,
                                rfr=rfr,
                            )
                            sweep_res = price_worst_of(sweep_note, n_paths=20_000)
                            sweep_rows.append(
                                {
                                    "Scenario": label,
                                    "Fair value (% face)": round(float(sweep_res["npv_pct"]), 2),
                                    "NPV ($)": round(float(sweep_res["npv_dollar"]), 2),
                                    "MC SE (bps)": round(float(sweep_res["se_bps"]), 1),
                                }
                            )
                        except Exception as e:
                            sweep_rows.append(
                                {
                                    "Scenario": label,
                                    "Fair value (% face)": f"Error: {e}",
                                    "NPV ($)": "—",
                                    "MC SE (bps)": "—",
                                }
                            )
                    st.session_state["wo_sweep_df"] = pd.DataFrame(sweep_rows)

            if "wo_result" in st.session_state:
                res = st.session_state["wo_result"]

                st.subheader("Worst-Of Pricing Result")
                m1, m2, m3 = st.columns(3)
                m1.metric("Fair value (% of face)", f"{res['npv_pct']:.2f}%")
                m2.metric("NPV ($ per face)", f"${res['npv_dollar']:.2f}")
                m3.metric("MC standard error (bps)", f"±{res['se_bps']:.1f}")

                st.caption(
                    f"Paths: {res['n_paths']:,} · Observation dates: "
                    f"{len(note_dict['observation_dates'])} · "
                    f"Correlation PSD-adjusted internally."
                )

                single_res = None
                single_name = None

                if selected:
                    try:
                        single_name = selected[0]
                        single_note = _build_note_dict(
                            underlier=single_name,
                            spot=spots[0],
                            face_value=face_value,
                            issue_date=issue_date,
                            maturity_date=maturity_date,
                            obs_freq=obs_freq,
                            autocall_pct=autocall_pct,
                            coupon_pct=coupon_pct,
                            knockin_pct=knockin_pct,
                            coupon_rate=coupon_rate,
                            rfr=rfr,
                        )
                        single_res = price_note_dict(single_note, n_paths=50_000)
                        wo_disc = single_res["npv_pct"] - res["npv_pct"]
                        st.markdown(
                            f"Compared to a single-name {single_name} note on identical terms, "
                            f"the worst-of basket is **{wo_disc:.2f} percentage points cheaper** "
                            f"({single_name} {single_res['npv_pct']:.2f}% vs worst-of {res['npv_pct']:.2f}%)."
                        )
                    except Exception:
                        single_res = None
                        single_name = None

                quote_text = _quote_summary_text(
                    note_dict=note_dict,
                    wo_result=res,
                    single_res=single_res,
                    single_name=single_name,
                )

                st.markdown("<hr>", unsafe_allow_html=True)
                _label("Quote Summary")
                st.text_area(
                    "Copy for client notes / email",
                    value=quote_text,
                    height=220,
                    key="wo_quote_summary",
                )
                if "wo_sweep_df" in st.session_state:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    _label("Correlation Scenario Results")
                    st.dataframe(
                        st.session_state["wo_sweep_df"],
                        use_container_width=True,
                        hide_index=True,
                    )

                if "wo_note_dict" in st.session_state:
                    with st.expander("Worst-of note JSON"):
                        st.json(st.session_state["wo_note_dict"])

            else:
                st.info("Configure the basket and click **Price worst-of note**.")
# ===========================================================================
# TAB 5 — PORTFOLIO
# ===========================================================================

with tab_port:
    PORTFOLIO_PATH = Path(__file__).parent / "data" / "portfolio.json"

    col_po1, col_po2 = st.columns([0.22, 0.78])

    with col_po1:
        _label("Settings")
        port_paths = st.select_slider(
            "MC Paths per Note",
            options=[5_000, 10_000, 30_000, 50_000],
            value=10_000,
            format_func=lambda x: f"{x:,}",
            key="po_npaths",
        )
        port_btn = st.button(
            "Run Portfolio Pricing",
            type="primary",
            use_container_width=True,
            key="po_btn",
        )
        if PORTFOLIO_PATH.exists():
            try:
                import json as _json

                _pdata = _json.loads(PORTFOLIO_PATH.read_text())
                _n = len(
                    [_nt for _nt in _pdata.get("notes", []) if not _nt.get("_comment")]
                )
                st.caption(f"**{_n} notes** in portfolio.json")
            except Exception:
                pass
        else:
            st.warning("data/portfolio.json not found.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption(
            "Flags (§10):  \n"
            "**OK** — |dev| ≤ 100 bps  \n"
            "**Review** — ≤ 300 bps  \n"
            "**Flag ⚠** — > 300 bps"
        )

    with col_po2:
        if port_btn:
            if not PORTFOLIO_PATH.exists():
                st.error("data/portfolio.json not found.")
            else:
                with st.spinner(f"Pricing portfolio at {port_paths:,} paths/note …"):
                    try:
                        from pricer.portfolio import price_portfolio

                        port_results = price_portfolio(
                            str(PORTFOLIO_PATH), n_paths=port_paths
                        )
                        st.session_state["port_results"] = port_results
                    except Exception as e:
                        st.error(f"Portfolio pricing failed: {e}")
                        st.session_state.pop("port_results", None)

        if "port_results" in st.session_state:
            rows = st.session_state["port_results"]
            ok_rows = [r for r in rows if "error" not in r]

            if ok_rows:
                total_face = sum(r["face_value"] for r in ok_rows)
                avg_model = (
                    sum(r["model_fv"] * r["face_value"] for r in ok_rows) / total_face
                )
                flagged = sum(1 for r in ok_rows if r["flag"] not in ("OK", "N/A"))
                total_pnl = sum(r["pnl_vs_purchase"] or 0 for r in ok_rows)

                pm1, pm2, pm3, pm4 = st.columns(4)
                pm1.metric("Notes Priced", str(len(ok_rows)))
                pm2.metric("Wtd Avg Model FV", f"{avg_model:.2f}%")
                pm3.metric("Review / Flag", str(flagged))
                pm4.metric("P&L vs Purchase", f"${total_pnl:+,.0f}")
                st.markdown("<hr>", unsafe_allow_html=True)

            table_rows = []
            for r in rows:
                if "error" in r:
                    table_rows.append(
                        {
                            "CUSIP": r["cusip"],
                            "Issuer": r["issuer"],
                            "Structure": "ERROR",
                            "Underlier(s)": r["underliers"],
                            "Issuer Mark": "—",
                            "Model FV": "—",
                            "Dev (bps)": "—",
                            "SE (bps)": "—",
                            "Flag": "ERROR",
                        }
                    )
                else:
                    table_rows.append(
                        {
                            "CUSIP": r["cusip"],
                            "Issuer": r["issuer"],
                            "Structure": r["structure"],
                            "Underlier(s)": r["underliers"],
                            "Issuer Mark": (
                                f"{r['issuer_mark']:.2f}%" if r["issuer_mark"] else "—"
                            ),
                            "Model FV": f"{r['model_fv']:.2f}%",
                            "Dev (bps)": (
                                f"{r['deviation_bps']:+.0f}"
                                if r["deviation_bps"] is not None
                                else "—"
                            ),
                            "SE (bps)": f"±{r.get('se_bps', 0):.1f}",
                            "Flag": r["flag"],
                        }
                    )
            st.dataframe(
                pd.DataFrame(table_rows), use_container_width=True, hide_index=True
            )
            st.caption(
                "Deviation = (Model FV − Issuer Mark) × 100.  Positive = model prices richer than issuer."
            )
        else:
            st.info(
                "Click **Run Portfolio Pricing** to price all notes and compare against issuer marks."
            )

# ===========================================================================
# TAB 6 — OFFERING EVALUATOR
# ===========================================================================

with tab_offer:
    col_oa, col_ob, col_oout = st.columns([1, 1, 1.6])

    # ── Column A: Structure & underlier(s) ──────────────────────────────
    with col_oa:
        _label("Structure")
        oe_type = st.radio(
            "Type",
            ["Single Underlier", "Worst-Of Basket"],
            horizontal=True,
            key="oe_type",
        )

        if oe_type == "Single Underlier":
            _label("Underlier")
            oe_ul = st.selectbox(
                "Ticker", UNDERLIERS, key="oe_ul", label_visibility="collapsed"
            )
            oe_sp = st.number_input(
                "Spot ($)",
                value=float(DEFAULT_SPOTS[oe_ul]),
                min_value=1.0,
                step=0.01,
                format="%.2f",
                key=f"oe_sp_{oe_ul}",
            )
        else:
            oe_n_assets = st.radio(
                "Basket Size",
                [2, 3],
                horizontal=True,
                key="oe_n",
                format_func=lambda x: f"{x} Underliers",
            )
            oa1, oa2 = st.columns(2)
            with oa1:
                oe_t1 = st.selectbox("Asset 1", UNDERLIERS, index=0, key="oe_t1")
            with oa2:
                oe_s1 = st.number_input(
                    "Spot 1",
                    value=float(DEFAULT_SPOTS[oe_t1]),
                    min_value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"oe_s1_{oe_t1}",
                )
            ob1, ob2 = st.columns(2)
            with ob1:
                oe_t2 = st.selectbox("Asset 2", UNDERLIERS, index=1, key="oe_t2")
            with ob2:
                oe_s2 = st.number_input(
                    "Spot 2",
                    value=float(DEFAULT_SPOTS[oe_t2]),
                    min_value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"oe_s2_{oe_t2}",
                )
            oe_t3 = oe_s3 = None
            if oe_n_assets == 3:
                oc1, oc2 = st.columns(2)
                with oc1:
                    oe_t3 = st.selectbox("Asset 3", UNDERLIERS, index=2, key="oe_t3")
                with oc2:
                    oe_s3 = st.number_input(
                        "Spot 3",
                        value=float(DEFAULT_SPOTS[oe_t3]),
                        min_value=1.0,
                        step=0.01,
                        format="%.2f",
                        key=f"oe_s3_{oe_t3}",
                    )
            oe_rho12 = st.slider("ρ (1/2)", -0.99, 0.99, 0.55, step=0.01, key="oe_r12")
            oe_rho13 = oe_rho23 = 0.0
            if oe_n_assets == 3:
                oe_rho13 = st.slider(
                    "ρ (1/3)", -0.99, 0.99, 0.50, step=0.01, key="oe_r13"
                )
                oe_rho23 = st.slider(
                    "ρ (2/3)", -0.99, 0.99, 0.50, step=0.01, key="oe_r23"
                )

        st.markdown("<hr>", unsafe_allow_html=True)
        _label("Schedule")
        od1, od2 = st.columns(2)
        with od1:
            oe_issue = st.date_input(
                "Issue Date", value=date(2026, 6, 3), key="oe_issue"
            )
        with od2:
            oe_mat = st.date_input(
                "Maturity Date", value=date(2027, 12, 3), key="oe_mat"
            )

        oe_freq = st.selectbox(
            "Observation Frequency",
            ["Quarterly", "Monthly", "Semi-Annual"],
            key="oe_freq",
        )

    # ── Column B: Barriers, rates & offer price ──────────────────────────
    with col_ob:
        oe_face = st.number_input(
            "Face Value ($)", value=1000.0, step=100.0, key="oe_face"
        )

        _label("Barrier Structure")
        oe_autocall = st.slider(
            "Autocall Barrier", 80, 115, 100, step=5, format="%d%%", key="oe_autocall"
        )
        oe_coupon = st.slider(
            "Coupon Barrier", 50, 95, 75, step=5, format="%d%%", key="oe_coupon"
        )
        oe_knockin = st.slider(
            "Knock-In Barrier", 40, 80, 65, step=5, format="%d%%", key="oe_knockin"
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        _label("Rates")

        oe1, oe2 = st.columns(2)
        with oe1:
            oe_cpn = st.number_input("Coupon (%)", value=12.0, step=0.5, key="oe_cpn")
        with oe2:
            oe_rfr = st.number_input("RFR (%)", value=3.75, step=0.25, key="oe_rfr")

        oe3, oe4 = st.columns(2)
        with oe3:
            oe_cs = st.number_input(
                "Credit Spread (bps)",
                value=100,
                step=5,
                min_value=0,
                max_value=500,
                key="oe_cs",
            )
        with oe4:
            oe_paths = st.select_slider(
                "MC Paths",
                options=[10_000, 50_000, 100_000],
                value=50_000,
                format_func=lambda x: f"{x:,}",
                key="oe_npaths",
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        _label("Offer Price")
        oe_offer = st.number_input(
            "Issuer Offer (% of Face)",
            min_value=50.0,
            max_value=110.0,
            value=100.0,
            step=0.1,
            key="oe_offer",
            help="New issuances are almost always at par (100%). Use secondary price for seasoned notes.",
        )

        oe_btn = st.button(
            "Evaluate Offering", type="primary", use_container_width=True, key="oe_btn"
        )

    # ── Column C: Evaluation result ──────────────────────────────────────
    with col_oout:
        st.markdown('<div class="results-panel">', unsafe_allow_html=True)
        _label("Evaluation")

        if oe_btn:
            if oe_type == "Single Underlier":
                oe_note = _build_note_dict(
                    oe_ul,
                    oe_sp,
                    oe_face,
                    oe_issue,
                    oe_mat,
                    oe_freq,
                    oe_autocall,
                    oe_coupon,
                    oe_knockin,
                    oe_cpn,
                    oe_rfr,
                    credit_spread_bps=oe_cs,
                )
            else:
                if oe_n_assets == 2:
                    oe_tickers = [oe_t1, oe_t2]
                    oe_spots = [oe_s1, oe_s2]
                    oe_corr = [[1.0, oe_rho12], [oe_rho12, 1.0]]
                else:
                    oe_tickers = [oe_t1, oe_t2, oe_t3]
                    oe_spots = [oe_s1, oe_s2, oe_s3]
                    oe_corr = [
                        [1.0, oe_rho12, oe_rho13],
                        [oe_rho12, 1.0, oe_rho23],
                        [oe_rho13, oe_rho23, 1.0],
                    ]
                oe_note = _build_wo_note_dict(
                    oe_tickers,
                    oe_spots,
                    oe_corr,
                    oe_face,
                    oe_issue,
                    oe_mat,
                    oe_freq,
                    oe_autocall,
                    oe_coupon,
                    oe_knockin,
                    oe_cpn,
                    oe_rfr,
                    credit_spread_bps=oe_cs,
                )

            if not oe_note["observation_dates"]:
                st.error(
                    "No observation dates — check Maturity Date is after Issue Date."
                )
            else:
                with st.spinner(f"Evaluating · {oe_paths:,} paths …"):
                    try:
                        from pricer.offering import evaluate_offering

                        oe_result = evaluate_offering(
                            oe_note, offer_pct=oe_offer, n_paths=oe_paths
                        )
                        st.session_state["oe_result"] = oe_result
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
                        st.session_state.pop("oe_result", None)

        if "oe_result" in st.session_state:
            oe_r = st.session_state["oe_result"]
            rec = oe_r["recommendation"]
            dev = oe_r["deviation_bps"]
            se = oe_r["se_bps"]

            st.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
                f"padding:1rem 1.25rem;margin-bottom:0.75rem;'>"
                f"<p class='section-label'>Model Recommendation</p>"
                f"{_recommendation_badge(rec)}"
                f"<p style='margin:0.6rem 0 0;font-size:0.85rem;color:#374151;'>"
                f"Confidence: <strong>{oe_r['confidence']}</strong> — deviation ({dev:+.0f} bps) "
                f"{'exceeds' if oe_r['confidence'] == 'High' else 'is within'} "
                f"the 2σ MC noise band (±{se*2:.0f} bps).</p></div>",
                unsafe_allow_html=True,
            )

            om1, om2, om3 = st.columns(3)
            om1.metric(
                "Model Fair Value",
                f"{oe_r['model_fv']:.2f}%",
                delta=f"{oe_r['deviation_pct']:+.2f}% vs offer",
            )
            om2.metric("Issuer Offer", f"{oe_r['offer_pct']:.2f}%")
            om3.metric("Deviation", f"{dev:+.0f} bps")

            om4, om5 = st.columns(2)
            om4.metric("Model FV ($)", f"${oe_r['model_dollar']:,.2f}")
            om5.metric("MC Std Error", f"±{se:.1f} bps")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.caption(
                f"**Buy** if deviation > +150 bps  ·  **Skip** if < −150 bps  ·  "
                f"**Gray Zone** within ±150 bps.  Current: {dev:+.0f} bps → **{rec}**."
            )
        else:
            st.info(
                "Enter the term sheet in the two columns on the left, "
                "set the issuer's offer price, and click **Evaluate Offering**."
            )

        st.markdown("</div>", unsafe_allow_html=True)
