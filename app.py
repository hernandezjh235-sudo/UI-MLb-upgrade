# -*- coding: utf-8 -*-
# ============================================================
# MLB STRIKEOUT PROP ENGINE — ONE FILE — v10.8 BAYESIAN MARKOV + XGB ASSIST
# Refresh first, then save official before-game snapshot
# Real lines only. No fake prop lines.
# Google Drive persistent logs + grading + learning.
# ============================================================

import os
import json
import math
import difflib
import io
import unicodedata
import requests
import numpy as np
import pandas as pd
import streamlit as st
from math import exp, factorial
from datetime import datetime, timedelta

APP_VERSION = "v10.8.3 FULL PRO UI + RAW DEBUG + HIT RATE"

try:
    import pytz
except Exception:
    pytz = None

# =========================
# STORAGE
# =========================
DRIVE_DIR = "/content/drive/MyDrive/mlb_engine"
LOCAL_DIR = "mlb_engine"

try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive", force_remount=False)
    os.makedirs(DRIVE_DIR, exist_ok=True)
    STORAGE_DIR = DRIVE_DIR
except Exception:
    os.makedirs(LOCAL_DIR, exist_ok=True)
    STORAGE_DIR = LOCAL_DIR

PICK_LOG = os.path.join(STORAGE_DIR, "auto_pick_log.json")
RESULT_LOG = os.path.join(STORAGE_DIR, "auto_result_log.json")
LEARN_FILE = os.path.join(STORAGE_DIR, "pitcher_learning.json")
CLV_FILE = os.path.join(STORAGE_DIR, "clv_tracker.json")
REQUEST_LOG_FILE = os.path.join(STORAGE_DIR, "request_log.json")
SIGNAL_TRACKING_FILE = os.path.join(STORAGE_DIR, "signal_tracking.json")
LONG_BACKTEST_FILE = os.path.join(STORAGE_DIR, "long_backtest_rows.json")
LINEUP_CACHE_FILE = os.path.join(STORAGE_DIR, "locked_lineup_cache.json")
LINE_HISTORY_FILE = os.path.join(STORAGE_DIR, "line_history.json")
RAW_PROP_DEBUG_FILE = os.path.join(STORAGE_DIR, "raw_prop_debug_rows.json")

MLB_BASE = "https://statsapi.mlb.com/api/v1"
MLB_LIVE = "https://statsapi.mlb.com/api/v1.1"
ODDS_BASE = "https://api.the-odds-api.com/v4"
PRIZEPICKS_URL = "https://api.prizepicks.com/projections"
UNDERDOG_URLS = [
    # v10.8.1: stable v1 first. Beta endpoints are backups only because they can expose alternate ladders.
    "https://api.underdogfantasy.com/v1/over_under_lines",
    "https://api.underdogfantasy.com/beta/v6/over_under_lines",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "https://api.underdogfantasy.com/beta/v4/over_under_lines",
    "https://api.underdogfantasy.com/beta/v3/over_under_lines",
    "https://api.underdogfantasy.com/beta/v2/over_under_lines",
]
SPORTSGAMEODDS_BASE = "https://api.sportsgameodds.com/v2"
OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

SPORTSBOOK_PITCHER_K_MARKETS = [
    "pitcher_strikeouts",
    "player_pitcher_strikeouts",
    "pitcher_strikeouts_alternate",
    "player_pitcher_strikeouts_alternate",
    "pitcher_strikeouts_over_under",
]

LEAGUE_AVG_K = 0.225
DEFAULT_BF = 22.0

# =========================
# v10.8 WEATHER + UMPIRE CAPS
# =========================
# These are deliberately small nudges. They cannot override lines or no-bet gates.
WEATHER_FACTOR_MIN = 0.975
WEATHER_FACTOR_MAX = 1.025
UMPIRE_FACTOR_MIN = 0.975
UMPIRE_FACTOR_MAX = 1.025
# =========================
# v10.3 UNDERDOG DEBUG + PRIMARY BOARD LINE SETTINGS
# =========================
# Goal: fewer plays, fewer coin-flips, higher true hit quality.
# These settings intentionally PASS on borderline props.
MIN_BETTABLE_GAP_KS = 1.00
MIN_ELITE_DATA_SCORE = 92
MIN_ELITE_NO_VIG_EDGE = 8.0
MIN_MATCH_SCORE_STRICT = 0.88

MIN_OFFICIAL_SAVE_SCORE = 82
MIN_BETTABLE_SCORE = 88
MIN_BETTABLE_PROB = 0.64
MIN_BETTABLE_EV = 0.06
MIN_CONFIRMED_LINEUP_SCORE = 90
MAX_RECOMMENDED_KELLY = 0.02
LEARNING_MIN_PRIOR_STARTS = 5
LEARNING_RATE = 0.04
LEARNING_SCALE_MIN = 0.92
LEARNING_SCALE_MAX = 1.08

# =========================
# v10.7 ADVANCED SIM / AI ASSIST SETTINGS
# =========================
# Bayesian + Markov is safe and ON by default.
# XGBoost is experimental and OFF by default until enough graded history exists.
BAYESIAN_MARKOV_SIMS = 14000
BAYESIAN_PROJECTION_STD_MIN = 0.45
BAYESIAN_PROJECTION_STD_MAX = 1.85
XGB_MIN_GRADED_SAMPLES = 100
XGB_MAX_RESIDUAL_ADJ_KS = 0.35
XGB_MAX_PERCENT_ADJ = 0.05
XGB_RECENT_TRAIN_LIMIT = 700

# =========================
# v10.8.2 MERGED FROM TRUE 10.0 EDGE ENGINE
# Conservative market-move signal only. This does NOT choose or rewrite lines.
# It only gives the model a tiny projection nudge after the real active line is found.
# =========================
MARKET_MOVE_FACTOR_MIN = 0.985
MARKET_MOVE_FACTOR_MAX = 1.015
MARKET_MOVE_K_SHIFT_CAP = 0.25


LEAGUE_AVG_WHIFF_BY_PITCH_TYPE = {
    "FF": 0.22, "SI": 0.17, "FC": 0.20, "SL": 0.34, "CU": 0.31,
    "KC": 0.31, "CH": 0.31, "FS": 0.34, "ST": 0.36, "SV": 0.30,
    "KN": 0.25, "EP": 0.15, "UNK": 0.25
}

def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

ODDS_API_KEY = get_secret("ODDS_API_KEY", "c9f5eadbe263f64c3fd17df20a4f1f3b")
SPORTSGAMEODDS_API_KEY = get_secret("SPORTSGAMEODDS_API_KEY", "")
OPTICODDS_API_KEY = get_secret("OPTICODDS_API_KEY", "")


# ============================================================
# v10.8.1 ORIGINAL BASE LINE FIX
# - Built from the original v10.8 file.
# - No live pitch-by-pitch layer.
# - No stacked v11 parser wrappers.
# - Underdog v1 is tried first.
# - Underdog line chooser no longer selects max(line).
# ============================================================

# =========================
# PAGE CONFIG + UI
# =========================
st.set_page_config(
    page_title="MLB K Prop Engine — Refresh Then Save",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at top,#260000 0%,#090909 42%,#020202 100%); color:#fff;}
.block-container {padding-top:1.1rem; max-width:1550px;}
h1,h2,h3 {color:#fff;}
[data-testid="stMetric"] {
    background:linear-gradient(145deg,#111,#1b0000);
    border:1px solid rgba(255,45,45,.36);
    border-radius:18px;
    padding:16px;
    box-shadow:0 0 18px rgba(255,0,0,.18);
}
.hero-panel {
    background:linear-gradient(135deg,rgba(50,0,0,.92),rgba(8,8,8,.96));
    border:1px solid rgba(255,70,70,.42);
    border-radius:26px;
    padding:22px;
    box-shadow:0 0 34px rgba(255,0,0,.18);
    margin-bottom:18px;
}
.pick-card {
    background:linear-gradient(145deg,#101010,#180000);
    border:1px solid rgba(255,45,45,.36);
    border-radius:22px;
    padding:20px;
    box-shadow:0 0 26px rgba(255,0,0,.17);
    margin-bottom:16px;
}
.green-card {
    background:linear-gradient(145deg,#001b0e,#07110b);
    border:1px solid rgba(0,255,135,.48);
    border-radius:22px;
    padding:22px;
    box-shadow:0 0 28px rgba(0,255,135,.22);
    margin-bottom:16px;
}
.warn-card {
    background:linear-gradient(145deg,#1c1200,#0f0a00);
    border:1px solid rgba(255,190,60,.45);
    border-radius:22px;
    padding:20px;
    box-shadow:0 0 24px rgba(255,190,60,.13);
    margin-bottom:16px;
}
.small-muted {color:#bdbdbd; font-size:13px;}
.big-title {font-size:42px; font-weight:950; color:#fff; letter-spacing:-1px;}
.sub-title {color:#d3d3d3; font-size:15px; margin-top:-6px;}
.player-name {font-size:23px; font-weight:900; color:#fff;}
.big-number {font-size:42px; font-weight:950; line-height:1.05;}
.green {color:#31e84f;}
.orange {color:#ffbe3c;}
.red {color:#ff5f5f;}
.badge {
    display:inline-block;
    padding:6px 12px;
    border-radius:999px;
    background:#2c0000;
    border:1px solid rgba(255,95,95,.48);
    color:#ffc4c4;
    font-weight:800;
    margin:3px 4px 3px 0;
}
.good-badge {background:#002916;border-color:rgba(0,255,135,.55);color:#b5ffd9;}
.yellow-badge {background:#2b1d00;border-color:rgba(255,210,70,.55);color:#ffe2a1;}
.red-badge {background:#2b0000;border-color:rgba(255,75,75,.55);color:#ffc0c0;}
.kpi-strip {display:grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap:12px; margin:12px 0 18px 0;}
.kpi-box {background:linear-gradient(145deg,#101010,#190000);border:1px solid rgba(255,70,70,.30);border-radius:18px;padding:14px;min-height:92px;}
.kpi-label {font-size:12px;color:#aaa;font-weight:800;letter-spacing:.04em;text-transform:uppercase;}
.kpi-value {font-size:26px;font-weight:900;color:#fff;margin-top:6px;}
.kpi-sub {font-size:12px;color:#cfcfcf;margin-top:5px;}
.progress-wrap {width:100%;height:14px;border-radius:99px;background:#050505;overflow:hidden;border:1px solid rgba(255,255,255,.08);}
.progress-green {height:100%;border-radius:99px;background:linear-gradient(90deg,#00d66b,#46ff9a);}
.progress-orange {height:100%;border-radius:99px;background:linear-gradient(90deg,#ff8c00,#ffbf30);}
.progress-red {height:100%;border-radius:99px;background:linear-gradient(90deg,#ff2d2d,#ff7272);}
.mini-k-bars {display:flex;align-items:flex-end;gap:10px;min-height:76px;margin-top:4px;overflow-x:auto;}
.mini-k-bar-wrap {display:inline-flex;flex-direction:column;align-items:center;justify-content:flex-end;min-width:18px;}
.mini-k-bar {display:block;width:17px;background:#31e84f;border-radius:3px;box-shadow:0 0 10px rgba(49,232,79,.18);}
.mini-k-label {font-size:12px;color:#bdbdbd;margin-top:3px;}
.hr-soft {border-top:1px solid rgba(255,255,255,.12); margin:14px 0;}
.section-title-pro {margin-top:22px;margin-bottom:10px;font-size:24px;font-weight:950;color:#fff;border-left:5px solid #ff3b3b;padding-left:12px;}
.stTabs [data-baseweb="tab"] {color:#b8c3cf;font-weight:850;}
.stTabs [aria-selected="true"] {color:#31e84f!important;border-bottom:3px solid #31e84f;}
@media (max-width: 1100px) {.kpi-strip {grid-template-columns: repeat(2, minmax(0, 1fr));}}

/* =========================
   v10.8.2 UI REFRESH CSS
   Visual-only; backend unchanged.
========================= */
.ui-hero{display:flex;justify-content:space-between;align-items:center;background:linear-gradient(135deg,#101923,#0b1118 55%,#071019);border:1px solid rgba(80,120,150,.24);border-radius:18px;padding:24px 28px;margin-bottom:18px;box-shadow:0 12px 36px rgba(0,0,0,.35);}
.ui-title{font-size:36px;font-weight:950;letter-spacing:-.8px;color:#f5f7fb;}
.ui-subtitle{color:#b8c3cc;margin-top:8px;font-size:16px;}
.ui-bankroll{min-width:190px;background:linear-gradient(145deg,#111c27,#0c141c);border:1px solid rgba(90,130,160,.35);border-radius:14px;padding:16px 18px;}
.ui-bankroll-label{color:#cbd3dc;font-size:15px;font-weight:700;}
.ui-bankroll-value{color:#31e84f;font-size:28px;font-weight:950;margin-top:4px;}
.ui-bankroll-sub{color:#93a0aa;font-size:12px;margin-top:6px;}
.ui-kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));background:#0c1219;border:1px solid rgba(90,130,160,.18);border-radius:0;margin-bottom:18px;overflow:hidden;}
.ui-kpi{padding:18px 20px;text-align:center;border-right:1px solid rgba(120,150,175,.16);}
.ui-kpi:last-child{border-right:0;}
.ui-kpi-label{font-size:14px;color:#b6c0ca;font-weight:800;}
.ui-kpi-value{font-size:24px;font-weight:950;margin-top:8px;color:#f4f7fa;}
.ui-card{background:linear-gradient(180deg,#101922,#0b141c);border-top:1px solid rgba(92,130,155,.22);border-bottom:1px solid rgba(92,130,155,.18);border-left:1px solid rgba(92,130,155,.10);border-right:1px solid rgba(92,130,155,.10);border-radius:0;margin:0 0 18px 0;padding:18px 20px;box-shadow:0 10px 30px rgba(0,0,0,.26);}
.ui-player{display:flex;align-items:center;gap:16px;margin-bottom:14px;}
.ui-avatar{width:72px;height:72px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,#253241,#0e161f);border:1px solid rgba(120,160,190,.30);color:#eaf0f5;font-weight:950;font-size:22px;}
.ui-name{font-size:20px;font-weight:900;color:#f7fafc;}
.ui-match{font-size:13px;color:#aeb9c4;margin-top:3px;}
.ui-main-cols{display:grid;grid-template-columns:1.15fr .7fr 1.1fr 1.2fr;gap:18px;border-top:1px solid rgba(150,170,190,.10);padding-top:14px;}
.ui-stat{border-right:1px solid rgba(150,170,190,.14);padding-right:16px;min-height:96px;}
.ui-stat:last-child{border-right:0;}
.ui-label{font-size:13px;color:#9eabb8;font-weight:800;margin-bottom:6px;}
.ui-big{font-size:30px;font-weight:950;line-height:1.05;}
.ui-mid{font-size:26px;font-weight:950;margin-top:4px;}
.green-text{color:#35ef4f!important;}
.orange-text{color:#ff9f1a!important;}
.white-text{color:#eef3f7!important;}
.ui-prog{height:15px;background:#071017;border-radius:999px;margin-top:12px;overflow:hidden;}
.ui-prog-fill{height:100%;background:#35ef4f;border-radius:999px;}
.ui-signal{font-size:18px;font-weight:950;margin-bottom:14px;}
.signal-green{color:#35ef4f;}
.signal-orange{color:#ff9f1a;}
.signal-muted{color:#c6d0d8;}
.ui-mini-row{display:flex;justify-content:space-between;color:#b9c4cf;margin:6px 0;font-size:14px;}
.ui-mini-row b{color:#eef3f7;}
.ui-footer{display:grid;grid-template-columns:.7fr .7fr .9fr 2.2fr;gap:16px;border-top:1px solid rgba(150,170,190,.10);margin-top:14px;padding-top:14px;}
.ui-footer span{display:block;color:#9eabb8;font-size:12px;font-weight:800;}
.ui-footer b{display:block;color:#eaf0f6;font-size:18px;margin-top:4px;}
.ui-last{min-width:240px;}
.ui-bars{display:flex;gap:10px;align-items:flex-end;height:74px;margin-top:2px;}
.ui-bar-wrap{display:flex;flex-direction:column;align-items:center;font-size:12px;color:#d4dde5;}
.ui-bar{width:16px;background:#35ef4f;border-radius:2px;box-shadow:0 0 10px rgba(53,239,79,.18);}
@media (max-width: 1000px){.ui-hero{flex-direction:column;align-items:flex-start;gap:14px;}.ui-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.ui-main-cols{grid-template-columns:1fr;}.ui-stat{border-right:0;border-bottom:1px solid rgba(150,170,190,.12);padding-bottom:12px;}.ui-footer{grid-template-columns:repeat(2,minmax(0,1fr));}}


/* =========================
   v10.8.3 FULL PRO UI CSS
   Visual-only; backend unchanged.
========================= */
.pro-shell{margin-top:4px;}
.pro-header{display:flex;justify-content:space-between;align-items:center;background:linear-gradient(135deg,#101923,#0b1118 58%,#071019);border:1px solid rgba(78,118,145,.28);border-radius:18px;padding:25px 30px;margin-bottom:0;box-shadow:0 16px 44px rgba(0,0,0,.35);}
.pro-title{font-size:36px;font-weight:950;letter-spacing:-.9px;color:#f4f8fb;text-shadow:0 0 14px rgba(255,255,255,.08);}
.pro-sub{font-size:16px;color:#b9c5cf;margin-top:8px;}
.pro-bankroll{background:linear-gradient(145deg,#111d28,#0b141d);border:1px solid rgba(96,135,165,.38);border-radius:15px;min-width:190px;padding:16px 18px;}
.pro-bank-label{font-size:15px;color:#d0d8df;font-weight:800;}
.pro-bank-value{font-size:28px;font-weight:950;color:#35ef4f;margin-top:5px;}
.pro-bank-time{font-size:12px;color:#97a4ae;margin-top:5px;}
.pro-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));background:#0c1219;border:1px solid rgba(100,135,165,.18);border-top:0;overflow:hidden;margin-bottom:16px;}
.pro-kpi{text-align:center;padding:18px 20px;border-right:1px solid rgba(130,160,185,.16);}
.pro-kpi:last-child{border-right:0;}
.pro-kpi span{display:block;color:#b9c3cc;font-size:14px;font-weight:800;}
.pro-kpi b{display:block;color:#f1f5f8;font-size:25px;margin-top:8px;font-weight:950;}
.pro-card{background:linear-gradient(180deg,#101922,#0b141c);border-top:1px solid rgba(100,135,165,.26);border-bottom:1px solid rgba(100,135,165,.18);border-left:1px solid rgba(100,135,165,.12);border-right:1px solid rgba(100,135,165,.12);padding:18px 20px;margin:0 0 18px 0;box-shadow:0 12px 30px rgba(0,0,0,.28);}
.pro-player{display:flex;align-items:center;gap:16px;margin-bottom:14px;}
.pro-avatar{width:72px;height:72px;border-radius:50%;background:linear-gradient(145deg,#263444,#101822);border:1px solid rgba(120,160,190,.35);display:flex;align-items:center;justify-content:center;color:#e9f0f6;font-weight:950;font-size:22px;}
.pro-name{font-size:20px;font-weight:900;color:#f6f9fb;}
.pro-game{font-size:13px;color:#aab6c1;margin-top:3px;}
.pro-grid{display:grid;grid-template-columns:1.12fr .7fr 1.12fr 1.2fr;gap:18px;border-top:1px solid rgba(150,170,190,.10);padding-top:14px;}
.pro-cell{border-right:1px solid rgba(150,170,190,.14);min-height:98px;padding-right:16px;}
.pro-cell:last-child{border-right:0;}
.pro-cell span,.pro-cell em{display:block;color:#9fabb7;font-size:13px;font-weight:800;font-style:normal;margin-bottom:6px;}
.pro-cell b{display:block;color:#edf3f8;font-size:30px;line-height:1.05;font-weight:950;}
.pro-cell strong{display:block;font-size:27px;margin-top:4px;font-weight:950;}
.pro-progress{height:15px;background:#071017;border-radius:999px;margin-top:12px;overflow:hidden;}
.pro-progress div{height:100%;background:#35ef4f;border-radius:999px;box-shadow:0 0 16px rgba(53,239,79,.25);}
.pro-signal{font-size:18px!important;margin-bottom:14px;}
.pro-money{display:flex;justify-content:space-between;gap:14px;color:#b9c4cf;margin:6px 0;font-size:14px;}
.pro-money span{margin:0!important;font-size:14px!important;}
.pro-money b{font-size:14px!important;color:#eef4f8!important;}
.pro-footer{display:grid;grid-template-columns:.7fr .7fr .9fr 2.2fr;gap:16px;border-top:1px solid rgba(150,170,190,.10);margin-top:14px;padding-top:14px;}
.pro-footer span{display:block;color:#9fabb7;font-size:12px;font-weight:800;}
.pro-footer b{display:block;color:#edf3f8;font-size:18px;margin-top:4px;}
.pro-last{min-width:240px;}
.pro-bars{display:flex;gap:10px;align-items:flex-end;height:74px;margin-top:2px;}
.pro-bar-wrap{display:flex;flex-direction:column;align-items:center;font-size:12px;color:#d4dde5;}
.pro-bar{width:16px;background:#35ef4f;border-radius:2px;box-shadow:0 0 10px rgba(53,239,79,.18);}
.green-text{color:#35ef4f!important;}
.orange-text{color:#ff9f1a!important;}
.signal-green{color:#35ef4f!important;}
.signal-orange{color:#ff9f1a!important;}
.signal-muted{color:#c6d0d8!important;}
@media(max-width:1000px){.pro-header{flex-direction:column;align-items:flex-start;gap:14px}.pro-kpis{grid-template-columns:repeat(2,minmax(0,1fr));}.pro-grid{grid-template-columns:1fr}.pro-cell{border-right:0;border-bottom:1px solid rgba(150,170,190,.12);padding-bottom:12px}.pro-footer{grid-template-columns:repeat(2,minmax(0,1fr));}}

</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def california_now():
    if pytz:
        return datetime.now(pytz.timezone("America/Los_Angeles"))
    return datetime.utcnow() - timedelta(hours=7)

def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def log_source_request(source, status, message=""):
    rows = load_json(REQUEST_LOG_FILE, [])
    rows.append({
        "time": now_iso(),
        "source": str(source)[:180],
        "status": str(status)[:80],
        "message": str(message)[:350]
    })
    save_json(REQUEST_LOG_FILE, rows[-500:])

def strip_accents(text):
    """Normalize accents so Underdog names like Sánchez match MLB names like Sanchez."""
    try:
        return "".join(
            ch for ch in unicodedata.normalize("NFKD", str(text or ""))
            if not unicodedata.combining(ch)
        )
    except Exception:
        return str(text or "")

def normalize_name(name):
    s = strip_accents(name).lower().strip()
    for ch in [".", ",", "'", "-", "_", " jr", " sr", " ii", " iii", " iv"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())

def name_score(a, b):
    """Robust player-name match.

    Handles full names, abbreviations, and Underdog initial + last-name display:
    - Cristopher Sanchez vs C. Sánchez
    - Gavin Williams vs G. Williams
    - Jacob deGrom vs J. deGrom
    """
    a_norm, b_norm = normalize_name(a), normalize_name(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return 0.94

    a_parts, b_parts = a_norm.split(), b_norm.split()
    if a_parts and b_parts:
        a_first, b_first = a_parts[0], b_parts[0]
        a_last, b_last = a_parts[-1], b_parts[-1]

        # Exact last-name + first-initial match, e.g. "Cristopher Sanchez" vs "C Sanchez".
        if a_last == b_last and a_first[:1] == b_first[:1]:
            return 0.93

        # Multi-word last names / particles still get strong credit if the last token and initial match.
        if a_last == b_last:
            return max(0.82, difflib.SequenceMatcher(None, a_norm, b_norm).ratio())

    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

def is_pitcher_k_text(text):
    t = str(text or "").lower()
    return (
        "strikeout" in t
        or "strike out" in t
        or "pitcher k" in t
        or t in ["ks", "k", "pitcher strikeouts"]
    ) and not any(bad in t for bad in ["batter", "hitter"])

def is_bad_sport_text(text):
    """Hard block non-MLB/basketball contamination from prop feeds."""
    t = f" {str(text or '').lower()} "
    bad_terms = [
        " nba", " nba_", "basketball", "wnba", "nfl", "football", "nhl",
        "soccer", "tennis", "golf", "college basketball", "ncaab"
    ]
    return any(x in t for x in bad_terms)

def is_bad_k_market_text(text):
    """Reject non-pitcher-K or alternate/novelty markets that can carry misleading values."""
    t = str(text or "").lower()
    bad_terms = [
        "batter", "hitter", "team strikeouts", "fantasy points", "fantasy score",
        "runs+rbi", "hits+runs+rbi", "total bases", "stolen base", "walks allowed",
        "earned runs", "outs recorded", "pitching outs", "hits allowed", "runs allowed",
        "single", "double", "home run", "rbi", "runs scored", "combo", "rival",
        "special", "discount", "alternative", "alt "
    ]
    return any(x in t for x in bad_terms)

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_json(url, params=None, timeout=14, headers=None):
    try:
        h = {
            "User-Agent": "Mozilla/5.0 MLBKPropEngine/refresh-save-build",
            "Accept": "application/json,text/plain,*/*",
        }
        if headers:
            h.update(headers)
        r = requests.get(url, params=params, timeout=timeout, headers=h)
        if r.status_code != 200:
            log_source_request(url, f"HTTP {r.status_code}", r.text[:250])
            return None
        try:
            return r.json()
        except Exception as e:
            log_source_request(url, "BAD_JSON", str(e))
            return None
    except Exception as e:
        log_source_request(url, "REQUEST_ERROR", str(e))
        return None

def baseball_ip_to_float(ip):
    if ip is None:
        return None
    try:
        s = str(ip)
        if "." not in s:
            return float(s)
        whole, frac = s.split(".", 1)
        outs = int(frac[:1]) if frac else 0
        if outs not in [0, 1, 2]:
            return float(s)
        return int(whole) + outs / 3
    except Exception:
        return None

def get_first_stat_split(data):
    if not isinstance(data, dict):
        return None
    stats = data.get("stats") or []
    if not stats or not isinstance(stats[0], dict):
        return None
    splits = stats[0].get("splits") or []
    if not splits or not isinstance(splits[0], dict):
        return None
    return splits[0]

def flatten_json(obj):
    items = []
    if isinstance(obj, dict):
        items.append(obj)
        for v in obj.values():
            items.extend(flatten_json(v))
    elif isinstance(obj, list):
        for x in obj:
            items.extend(flatten_json(x))
    return items

def first_value(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in [None, ""]:
            return d[k]
    return None


# =========================
# v10.8.2 UI REFRESH HELPERS
# Visual-only. Does not change lines, projections, EV, or pick logic.
# =========================
def ui_pct(x, default="—"):
    try:
        if x is None:
            return default
        f = float(x)
        if f <= 1:
            f *= 100
        return f"{f:.0f}%"
    except Exception:
        return default

def ui_num(x, digits=2, default="—"):
    try:
        if x is None:
            return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default

def ui_signal_class(signal_text):
    s = str(signal_text or "").lower()
    if "strong" in s or "over" in s or "elite" in s:
        return "signal-green"
    if "pass" in s or "no" in s:
        return "signal-orange"
    return "signal-muted"

def ui_pick_signal(row):
    sig = row.get("Signal") or row.get("signal") or row.get("Grade") or row.get("grade") or row.get("Confidence") or row.get("confidence")
    if sig:
        return str(sig)
    prob = safe_float(row.get("Over Probability", row.get("over_prob", row.get("prob"))))
    edge = safe_float(row.get("Edge", row.get("edge")))
    if prob is not None and prob >= 0.66 and (edge is None or edge >= 0.75):
        return "🔥 STRONG OVER"
    if prob is not None and prob >= 0.58:
        return "✅ OVER"
    return "PASS"

def ui_last_ks(row):
    vals = row.get("last_10_ks") or row.get("Last 10 Ks") or row.get("Last 10 Games (K)") or row.get("recent_ks")
    if isinstance(vals, list):
        return vals[:10]
    if isinstance(vals, str):
        nums = re.findall(r"\d+", vals)
        return [int(x) for x in nums[:10]]
    return []

def render_ui_refresh_hero(board_rows=None, bankroll=None):
    board_rows = board_rows if isinstance(board_rows, list) else []
    games_today = len(set([r.get("game_pk") for r in board_rows if isinstance(r, dict) and r.get("game_pk")])) or "—"
    pitchers = len(board_rows) if board_rows else "—"
    edges = []
    best_signal = "—"
    for r in board_rows:
        if not isinstance(r, dict):
            continue
        e = safe_float(r.get("Edge", r.get("edge")))
        if e is not None:
            edges.append(e)
    avg_edge = np.mean(edges) if edges else None
    if board_rows:
        try:
            best = sorted(board_rows, key=lambda x: safe_float(x.get("Score", x.get("score")), 0) or 0, reverse=True)[0]
            best_signal = ui_pick_signal(best)
        except Exception:
            pass
    bankroll_txt = f"${float(bankroll or 1000):,.2f}" if safe_float(bankroll, None) is not None else "$1,000.00"
    st.markdown(f"""
    <div class="ui-hero">
      <div>
        <div class="ui-title">🔥 MLB STRIKEOUT PROP ENGINE</div>
        <div class="ui-subtitle">Real Projections | Edge | Kelly Criterion | Simulations</div>
      </div>
      <div class="ui-bankroll">
        <div class="ui-bankroll-label">Bankroll</div>
        <div class="ui-bankroll-value">{bankroll_txt}</div>
        <div class="ui-bankroll-sub">Updated: {california_now().strftime('%I:%M %p PT')}</div>
      </div>
    </div>
    <div class="ui-kpi-grid">
      <div class="ui-kpi"><div class="ui-kpi-label">Games Today</div><div class="ui-kpi-value">{games_today}</div></div>
      <div class="ui-kpi"><div class="ui-kpi-label">Pitchers Analyzed</div><div class="ui-kpi-value">{pitchers}</div></div>
      <div class="ui-kpi"><div class="ui-kpi-label">Avg Edge</div><div class="ui-kpi-value green-text">{ui_num(avg_edge,2)}</div></div>
      <div class="ui-kpi"><div class="ui-kpi-label">Best Play</div><div class="ui-kpi-value green-text">{best_signal}</div></div>
    </div>
    """, unsafe_allow_html=True)

def render_ui_pick_card(row, idx=0):
    if not isinstance(row, dict):
        return
    name = row.get("pitcher") or row.get("Pitcher") or row.get("player") or row.get("Player") or "Unknown Pitcher"
    matchup = row.get("matchup") or row.get("Matchup") or row.get("Team Matchup") or ""
    hand = row.get("hand") or row.get("Hand") or ""
    proj = safe_float(row.get("projection", row.get("Projection")))
    line = safe_float(row.get("line", row.get("Line")))
    edge = safe_float(row.get("edge", row.get("Edge")))
    prob = safe_float(row.get("over_prob", row.get("prob", row.get("Over Probability"))))
    signal = ui_pick_signal(row)
    sig_class = ui_signal_class(signal)
    k_rate = safe_float(row.get("pitcher_k", row.get("Pitcher K%", row.get("K%"))))
    opp_k = safe_float(row.get("lineup_k", row.get("Opp K%", row.get("Opponent K%"))))
    bf = safe_float(row.get("expected_bf", row.get("Expected BF")))
    kelly = safe_float(row.get("kelly", row.get("Kelly %")))
    bet_size = safe_float(row.get("bet_size", row.get("Bet Size")))
    ks = ui_last_ks(row)
    if not ks:
        ks = [0]*10
    max_k = max(max(ks), 1)
    bars = "".join([f'<div class="ui-bar-wrap"><div class="ui-bar" style="height:{max(18, int((k/max_k)*56))}px"></div><div>{k}</div></div>' for k in ks[:10]])
    initials = "".join([p[:1] for p in str(name).split()[:2]]).upper() or "P"
    st.markdown(f"""
    <div class="ui-card">
      <div class="ui-player">
        <div class="ui-avatar">{initials}</div>
        <div class="ui-namebox">
          <div class="ui-name">{name}</div>
          <div class="ui-match">{matchup}</div>
          <div class="ui-match">{hand}</div>
        </div>
      </div>
      <div class="ui-main-cols">
        <div class="ui-stat">
          <div class="ui-label">Projection</div>
          <div class="ui-big green-text">{ui_num(proj,2)}</div>
          <div class="ui-label">Edge</div>
          <div class="ui-mid {('green-text' if (edge or 0) >= 0.5 else 'orange-text')}">{ui_num(edge,2)}</div>
        </div>
        <div class="ui-stat">
          <div class="ui-label">Line</div>
          <div class="ui-big white-text">{ui_num(line,1)}</div>
        </div>
        <div class="ui-stat">
          <div class="ui-label">Over Probability</div>
          <div class="ui-big green-text">{ui_pct(prob)}</div>
          <div class="ui-prog"><div class="ui-prog-fill" style="width:{min(max((prob or 0)*100,0),100):.0f}%"></div></div>
        </div>
        <div class="ui-stat">
          <div class="ui-label">Signal</div>
          <div class="ui-signal {sig_class}">{signal}</div>
          <div class="ui-mini-row"><span>Bet Size</span><b>${bet_size or 0:,.2f}</b></div>
          <div class="ui-mini-row"><span>Kelly %</span><b>{(kelly*100 if kelly and kelly <= 1 else (kelly or 0)):.2f}%</b></div>
        </div>
      </div>
      <div class="ui-footer">
        <div><span>K%</span><b>{ui_num(k_rate,2)}</b></div>
        <div><span>Opp K%</span><b>{ui_num(opp_k,2)}</b></div>
        <div><span>Expected BF</span><b>{ui_num(bf,1)}</b></div>
        <div class="ui-last"><span>Last 10 Games (K)</span><div class="ui-bars">{bars}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_ui_refresh_board(board_rows, max_cards=8):
    if not isinstance(board_rows, list) or not board_rows:
        st.info("No board rows available yet. Refresh/sync props first.")
        return
    def _rank(r):
        return (
            safe_float(r.get("Score", r.get("score")), 0) or 0,
            safe_float(r.get("Edge", r.get("edge")), 0) or 0,
            safe_float(r.get("over_prob", r.get("prob")), 0) or 0,
        )
    rows = sorted([r for r in board_rows if isinstance(r, dict)], key=_rank, reverse=True)[:max_cards]
    for i, row in enumerate(rows):
        render_ui_pick_card(row, i)



# =========================
# v10.8.1 READ-ONLY RAW DEBUG TABLE
# =========================
def log_raw_prop_debug_row(row, source_hint=""):
    """Read-only debug logger.

    This does not modify lines, projections, picks, probabilities, or EV.
    It only saves what the parser saw so wrong/missing lines can be inspected.
    """
    try:
        if not isinstance(row, dict):
            return
        rows = load_json(RAW_PROP_DEBUG_FILE, [])
        slim = {
            "time": now_iso(),
            "source_hint": source_hint,
            "Source": row.get("Source") or row.get("Provider") or source_hint,
            "Feed Name": row.get("Feed Name") or row.get("Matched Name") or row.get("Player") or row.get("Pitcher"),
            "Matched Name": row.get("Matched Name"),
            "Line": row.get("Line") or row.get("line"),
            "Market": row.get("Market") or row.get("Stat") or row.get("stat_type"),
            "Side": row.get("Side"),
            "Parser Mode": row.get("Parser Mode"),
            "Match Score": row.get("Match Score"),
            "Line Evidence": row.get("Line Evidence"),
            "Underdog Path": row.get("Underdog Path"),
            "Board Match": row.get("Board Match"),
        }
        rows.append(slim)
        save_json(RAW_PROP_DEBUG_FILE, rows[-1500:])
    except Exception:
        pass

def log_raw_prop_debug_rows(rows, source_hint=""):
    try:
        if isinstance(rows, list):
            for r in rows:
                log_raw_prop_debug_row(r, source_hint=source_hint)
    except Exception:
        pass

def render_raw_prop_debug_table():
    rows = load_json(RAW_PROP_DEBUG_FILE, [])
    if not rows:
        st.info("No raw prop debug rows logged yet. Refresh the board first.")
        return
    df = pd.DataFrame(rows)
    wanted_cols = [
        "time", "source_hint", "Source", "Feed Name", "Matched Name", "Line",
        "Market", "Side", "Parser Mode", "Match Score", "Line Evidence",
        "Underdog Path", "Board Match"
    ]
    cols = [c for c in wanted_cols if c in df.columns]
    st.dataframe(df[cols].tail(500).iloc[::-1], use_container_width=True, hide_index=True)




# =========================
# v10.8.1 READ-ONLY HIT-RATE DASHBOARD
# =========================
def build_hit_rate_dashboard_rows():
    """Read-only dashboard from RESULT_LOG.

    Does not grade, change, or save anything. It only summarizes already-logged results.
    """
    results = load_json(RESULT_LOG, [])
    rows = []
    for r in results:
        if not isinstance(r, dict):
            continue
        actual = safe_float(r.get("actual"))
        win = r.get("win")
        if actual is None or win is None:
            continue
        score = safe_float(r.get("score"))
        if score is None:
            tier = "Unknown"
        elif score >= 92:
            tier = "Elite 92+"
        elif score >= 88:
            tier = "Strong 88-91"
        elif score >= 82:
            tier = "Watch 82-87"
        else:
            tier = "Low <82"

        projection = safe_float(r.get("projection"))
        rows.append({
            "Date": r.get("date", ""),
            "Pitcher": r.get("pitcher", r.get("player", "")),
            "Side": str(r.get("side", r.get("Side", "Over"))).title(),
            "Line": safe_float(r.get("line", r.get("Line"))),
            "Projection": projection,
            "Actual Ks": actual,
            "Win": bool(win),
            "Score": score,
            "Tier": tier,
            "EV": safe_float(r.get("ev", r.get("EV"))),
            "Probability": safe_float(r.get("prob", r.get("over_prob", r.get("Probability")))),
            "Error": None if projection is None else actual - projection,
            "Abs Error": None if projection is None else abs(actual - projection),
        })
    return rows

def summarize_hit_rate_group(df, group_col):
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    out = df.groupby(group_col).agg(
        Picks=("Win", "count"),
        Hit_Rate=("Win", "mean"),
        Avg_Error=("Error", "mean"),
        MAE=("Abs Error", "mean"),
    ).reset_index()
    out["Hit_Rate"] = (out["Hit_Rate"] * 100).round(1)
    out["Avg_Error"] = out["Avg_Error"].round(2)
    out["MAE"] = out["MAE"].round(2)
    return out.sort_values(["Picks", "Hit_Rate"], ascending=[False, False])

def render_hit_rate_dashboard():
    rows = build_hit_rate_dashboard_rows()
    if not rows:
        st.info("No graded result rows found yet. Once picks are graded into RESULT_LOG, this dashboard will populate.")
        return

    df = pd.DataFrame(rows)
    total = len(df)
    hit_rate = df["Win"].mean() * 100 if total else 0
    mae = df["Abs Error"].dropna().mean()
    bias = df["Error"].dropna().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graded Picks", total)
    c2.metric("Hit Rate", f"{hit_rate:.1f}%")
    c3.metric("MAE", "—" if pd.isna(mae) else f"{mae:.2f} Ks")
    c4.metric("Bias", "—" if pd.isna(bias) else f"{bias:+.2f} Ks")

    st.markdown("#### By Confidence Tier")
    tier_df = summarize_hit_rate_group(df, "Tier")
    if not tier_df.empty:
        st.dataframe(tier_df, use_container_width=True, hide_index=True)

    st.markdown("#### By Side")
    side_df = summarize_hit_rate_group(df, "Side")
    if not side_df.empty:
        st.dataframe(side_df, use_container_width=True, hide_index=True)

    st.markdown("#### By Pitcher")
    pitcher_df = summarize_hit_rate_group(df, "Pitcher")
    if not pitcher_df.empty:
        st.dataframe(pitcher_df.head(75), use_container_width=True, hide_index=True)

    st.markdown("#### Recent Graded Picks")
    st.dataframe(df.tail(150).iloc[::-1], use_container_width=True, hide_index=True)



# =========================
# v10.8.3 FULL PRO UI HELPERS
# Visual/reporting only. Does not change lines, projections, EV, or pick logic.
# =========================
def get_current_board_rows_for_ui():
    """Find the active board/pick list without modifying it."""
    candidates = [
        globals().get("board"),
        globals().get("picks"),
        globals().get("final_board"),
        globals().get("all_picks"),
        globals().get("top_picks"),
        st.session_state.get("board") if hasattr(st, "session_state") else None,
        st.session_state.get("picks") if hasattr(st, "session_state") else None,
    ]
    for c in candidates:
        if isinstance(c, list) and len(c) > 0 and any(isinstance(x, dict) for x in c):
            return c
    return []

def ui_get(row, *keys, default=None):
    for k in keys:
        if isinstance(row, dict) and k in row and row.get(k) not in [None, ""]:
            return row.get(k)
    return default

def ui_probability(row):
    return safe_float(ui_get(row, "over_prob", "prob", "Probability", "Over Probability", "over_probability"))

def ui_projection(row):
    return safe_float(ui_get(row, "projection", "Projection", "Projected Ks", "Projected K"))

def ui_line(row):
    return safe_float(ui_get(row, "line", "Line", "Prop Line"))

def ui_edge(row):
    e = safe_float(ui_get(row, "edge", "Edge"))
    if e is not None:
        return e
    p = ui_projection(row)
    l = ui_line(row)
    if p is not None and l is not None:
        return p - l
    return None

def ui_score(row):
    return safe_float(ui_get(row, "score", "Score", "data_score", "Data Score"), 0) or 0

def ui_pick_sort_key(row):
    return (ui_score(row), ui_edge(row) or 0, ui_probability(row) or 0)

def ui_team_text(row):
    return ui_get(row, "matchup", "Matchup", "Team Matchup", "game", "Game", default="")

def ui_name(row):
    return ui_get(row, "pitcher", "Pitcher", "player", "Player", "Feed Name", default="Unknown Pitcher")

def ui_hand(row):
    return ui_get(row, "hand", "Hand", "throws", "Pitch Hand", default="")

def ui_bet_size(row):
    return safe_float(ui_get(row, "bet_size", "Bet Size", "recommended_bet", "Recommended Bet"), 0) or 0

def ui_kelly(row):
    k = safe_float(ui_get(row, "kelly", "Kelly", "Kelly %", "kelly_fraction"), 0) or 0
    return k * 100 if k <= 1 else k

def ui_signal(row):
    sig = ui_get(row, "Signal", "signal", "Grade", "grade", "Confidence", "confidence")
    if sig:
        return str(sig)
    prob = ui_probability(row)
    edge = ui_edge(row)
    if prob is not None and prob >= 0.66 and (edge is None or edge >= 0.75):
        return "🔥 STRONG OVER"
    if prob is not None and prob >= 0.58:
        return "✅ OVER"
    return "PASS"

def ui_sig_class(sig):
    s = str(sig or "").lower()
    if "strong" in s or "elite" in s:
        return "signal-green"
    if "over" in s or "play" in s:
        return "signal-green"
    if "pass" in s or "no" in s:
        return "signal-orange"
    return "signal-muted"

def ui_find_recent_ks(row):
    for key in ["last_10_ks", "Last 10 Ks", "recent_ks", "Recent Ks", "ks_history"]:
        vals = row.get(key) if isinstance(row, dict) else None
        if isinstance(vals, list):
            return [safe_int(v, 0) or 0 for v in vals[:10]]
        if isinstance(vals, str):
            nums = re.findall(r"\d+", vals)
            if nums:
                return [int(x) for x in nums[:10]]
    # Try nested recent log rows
    logs = row.get("recent_rows") or row.get("recent_logs") or row.get("Recent Logs") if isinstance(row, dict) else None
    if isinstance(logs, list):
        out = []
        for g in logs[:10]:
            if isinstance(g, dict):
                out.append(safe_int(g.get("Ks"), 0) or 0)
        if out:
            return out
    return []

def ui_short_num(x, digits=2, default="—"):
    try:
        if x is None:
            return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default

def ui_short_pct(x, default="—"):
    try:
        if x is None:
            return default
        f = float(x)
        if f <= 1:
            f *= 100
        return f"{f:.0f}%"
    except Exception:
        return default

def render_full_pro_header(rows, bankroll=None):
    rows = rows if isinstance(rows, list) else []
    games = len(set([r.get("game_pk") for r in rows if isinstance(r, dict) and r.get("game_pk")])) or "—"
    pitchers = len(rows) if rows else "—"
    edges = [ui_edge(r) for r in rows if isinstance(r, dict) and ui_edge(r) is not None]
    avg_edge = float(np.mean(edges)) if edges else None
    best = None
    if rows:
        try:
            best = sorted([r for r in rows if isinstance(r, dict)], key=ui_pick_sort_key, reverse=True)[0]
        except Exception:
            best = None
    best_signal = ui_signal(best) if best else "—"
    bankroll_value = safe_float(bankroll, None)
    if bankroll_value is None:
        bankroll_value = 1000
    st.markdown(f"""
    <div class="pro-shell">
      <div class="pro-header">
        <div>
          <div class="pro-title">🔥 MLB STRIKEOUT PROP ENGINE</div>
          <div class="pro-sub">Real Projections | Edge | Kelly Criterion | Simulations</div>
        </div>
        <div class="pro-bankroll">
          <div class="pro-bank-label">Bankroll</div>
          <div class="pro-bank-value">${bankroll_value:,.2f}</div>
          <div class="pro-bank-time">Updated: {california_now().strftime('%I:%M %p PT')}</div>
        </div>
      </div>
      <div class="pro-kpis">
        <div class="pro-kpi"><span>Games Today</span><b>{games}</b></div>
        <div class="pro-kpi"><span>Pitchers Analyzed</span><b>{pitchers}</b></div>
        <div class="pro-kpi"><span>Avg Edge</span><b class="green-text">{ui_short_num(avg_edge,2)}</b></div>
        <div class="pro-kpi"><span>Best Play</span><b class="green-text">{best_signal}</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_full_pro_card(row, rank=1):
    name = ui_name(row)
    matchup = ui_team_text(row)
    hand = ui_hand(row)
    proj = ui_projection(row)
    line = ui_line(row)
    edge = ui_edge(row)
    prob = ui_probability(row)
    sig = ui_signal(row)
    sig_class = ui_sig_class(sig)
    bet_size = ui_bet_size(row)
    kelly = ui_kelly(row)
    k_rate = safe_float(ui_get(row, "pitcher_k", "Pitcher K%", "K%", "k_rate"))
    opp_k = safe_float(ui_get(row, "lineup_k", "Opp K%", "Opponent K%", "opp_k"))
    bf = safe_float(ui_get(row, "expected_bf", "Expected BF", "BF"))
    initials = "".join([p[:1] for p in str(name).split()[:2]]).upper() or "P"
    ks = ui_find_recent_ks(row)
    if not ks:
        ks = [0,0,0,0,0,0,0,0,0,0]
    max_k = max(max(ks), 1)
    bars = "".join([f'<div class="pro-bar-wrap"><div class="pro-bar" style="height:{max(14, int((k/max_k)*56))}px"></div><small>{k}</small></div>' for k in ks[:10]])
    prob_width = min(max((prob or 0) * 100, 0), 100)
    st.markdown(f"""
    <div class="pro-card">
      <div class="pro-player">
        <div class="pro-avatar">{initials}</div>
        <div>
          <div class="pro-name">{name}</div>
          <div class="pro-game">{matchup}</div>
          <div class="pro-game">{hand}</div>
        </div>
      </div>
      <div class="pro-grid">
        <div class="pro-cell">
          <span>Projection</span>
          <b class="green-text">{ui_short_num(proj,2)}</b>
          <em>Edge</em>
          <strong class="{('green-text' if (edge or 0) >= 0.5 else 'orange-text')}">{ui_short_num(edge,2)}</strong>
        </div>
        <div class="pro-cell">
          <span>Line</span>
          <b>{ui_short_num(line,1)}</b>
        </div>
        <div class="pro-cell">
          <span>Over Probability</span>
          <b class="{('green-text' if (prob or 0) >= 0.58 else 'orange-text')}">{ui_short_pct(prob)}</b>
          <div class="pro-progress"><div style="width:{prob_width:.0f}%"></div></div>
        </div>
        <div class="pro-cell">
          <span>Signal</span>
          <b class="{sig_class} pro-signal">{sig}</b>
          <div class="pro-money"><span>Bet Size</span><b>${bet_size:,.2f}</b></div>
          <div class="pro-money"><span>Kelly %</span><b>{kelly:.2f}%</b></div>
        </div>
      </div>
      <div class="pro-footer">
        <div><span>K%</span><b>{ui_short_num(k_rate,2)}</b></div>
        <div><span>Opp K%</span><b>{ui_short_num(opp_k,2)}</b></div>
        <div><span>Expected BF</span><b>{ui_short_num(bf,1)}</b></div>
        <div class="pro-last"><span>Last 10 Games (K)</span><div class="pro-bars">{bars}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_full_pro_board(rows, max_cards=10):
    rows = rows if isinstance(rows, list) else []
    valid = [r for r in rows if isinstance(r, dict)]
    if not valid:
        st.info("No current board data available yet. Refresh/sync props first.")
        return
    ranked = sorted(valid, key=ui_pick_sort_key, reverse=True)
    for i, row in enumerate(ranked[:max_cards], start=1):
        render_full_pro_card(row, i)


# =========================
# BETTING MATH
# =========================
def poisson_over_probability(lam, line):
    lam = safe_float(lam, 0)
    line = safe_float(line)
    if line is None or lam <= 0:
        return None
    k = int(math.floor(line))
    prob_under_or_equal = sum((lam ** i) * exp(-lam) / factorial(i) for i in range(k + 1))
    return float(clamp(1 - prob_under_or_equal, 0.001, 0.999))

def american_to_implied(price):
    price = safe_float(price)
    if price is None:
        return None
    if price > 0:
        return 100 / (price + 100)
    return abs(price) / (abs(price) + 100)

def decimal_odds(odds):
    odds = safe_float(odds)
    if odds is None:
        return None
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)

def expected_value(prob, odds):
    dec = decimal_odds(odds)
    if prob is None or dec is None:
        return None
    return (prob * (dec - 1)) - (1 - prob)

def kelly_fraction(prob, odds):
    dec = decimal_odds(odds)
    if prob is None or dec is None:
        return 0.0
    b = dec - 1
    q = 1 - prob
    if b <= 0:
        return 0.0
    return float(clamp(((b * prob) - q) / b, 0, 0.25))

def paired_no_vig_probability(rows, target_row):
    price = safe_float(target_row.get("Price"))
    listed = american_to_implied(price)
    if listed is None:
        return None
    provider = str(target_row.get("Provider", target_row.get("Source", ""))).lower()
    line = safe_float(target_row.get("Line"))
    side = str(target_row.get("Side", "")).lower()
    if line is None or not side:
        return listed
    want = "under" if "over" in side else "over" if "under" in side else None
    if not want:
        return listed
    opposite = None
    for r in rows or []:
        if safe_float(r.get("Line")) != line:
            continue
        if str(r.get("Provider", r.get("Source", ""))).lower() != provider:
            continue
        if want in str(r.get("Side", "")).lower():
            opposite = american_to_implied(r.get("Price"))
            break
    if opposite is None:
        return listed
    denom = listed + opposite
    return listed / denom if denom > 0 else listed

# =========================
# LEARNING / CLV / LOGGING
# =========================
def load_learning():
    return load_json(LEARN_FILE, {})

def apply_learning(pid, lam):
    data = load_learning()
    scale = safe_float(data.get(str(pid)), 1.0) or 1.0
    return lam * scale, scale

def pitcher_learning_sample_count(pid):
    """Count previous graded official snapshots for this pitcher before changing learning scale."""
    results = load_json(RESULT_LOG, [])
    return sum(
        1 for r in results
        if str(r.get("pitcher_id")) == str(pid)
        and r.get("actual") is not None
        and r.get("projection") is not None
    )

def update_learning(pid, projected, actual):
    """
    Safer learning:
    - does NOT move from one random outcome
    - waits for prior samples
    - uses a smaller learning rate
    - caps pitcher scale tighter
    """
    data = load_learning()
    projected = safe_float(projected, 0) or 0
    actual = safe_float(actual)
    current = safe_float(data.get(str(pid)), 1.0) or 1.0

    if actual is None or projected <= 0:
        return current

    prior_samples = pitcher_learning_sample_count(pid)
    if prior_samples < LEARNING_MIN_PRIOR_STARTS:
        data[str(pid)] = current
        save_json(LEARN_FILE, data)
        return current

    err = clamp((actual - projected) / max(1.0, projected), -0.35, 0.35)
    new_scale = clamp(current * (1 + LEARNING_RATE * err), LEARNING_SCALE_MIN, LEARNING_SCALE_MAX)
    data[str(pid)] = new_scale
    save_json(LEARN_FILE, data)
    return new_scale

def update_clv_snapshot(player_name, source, line):
    if line is None:
        return None
    data = load_json(CLV_FILE, {})
    today = california_now().strftime("%Y-%m-%d")
    key = f"{today}_{normalize_name(player_name)}_{source}"
    old = data.get(key)
    line = float(line)
    if not old:
        data[key] = {
            "player": player_name,
            "source": source,
            "open_line": line,
            "latest_line": line,
            "last_updated": now_iso()
        }
        save_json(CLV_FILE, data)
        return 0.0
    open_line = safe_float(old.get("open_line"))
    old["latest_line"] = line
    old["last_updated"] = now_iso()
    data[key] = old
    save_json(CLV_FILE, data)
    if open_line is None:
        return 0.0
    return round(line - open_line, 2)

def track_line_delta(player_name, source, line):
    if line is None:
        return None
    hist = load_json(LINE_HISTORY_FILE, {})
    key = f"{normalize_name(player_name)}_{source}"
    rows = hist.get(key, [])
    rows.append({"t": now_iso(), "line": safe_float(line)})
    hist[key] = rows[-30:]
    save_json(LINE_HISTORY_FILE, hist)
    if len(hist[key]) < 2:
        return 0.0
    first = safe_float(hist[key][0].get("line"))
    last = safe_float(hist[key][-1].get("line"))
    if first is None or last is None:
        return None
    return round(last - first, 2)

def log_long_backtest_row(pick):
    rows = load_json(LONG_BACKTEST_FILE, [])
    pid = pick.get("pick_id")
    ids = set(r.get("pick_id") for r in rows)
    if pid not in ids:
        slim = {k: v for k, v in pick.items() if k not in ["prop_rows", "lineup_rows", "pitch_type_rows"]}
        rows.append(slim)
        save_json(LONG_BACKTEST_FILE, rows[-20000:])

def build_model_calibration_profile(results):
    finished = [r for r in results if r.get("actual") is not None and r.get("projection") is not None]
    if not finished:
        return {"samples": 0, "mae": None, "bias": None, "hit_rate": None, "quality_score": 50}
    errs = [safe_float(r.get("actual"), 0) - safe_float(r.get("projection"), 0) for r in finished]
    mae = float(np.mean([abs(e) for e in errs]))
    bias = float(np.mean(errs))
    wins = [1 if r.get("win") else 0 for r in finished if r.get("win") is not None]
    hit_rate = float(np.mean(wins)) if wins else None
    quality = 50
    quality += min(len(finished), 50) * 0.6
    quality -= min(mae, 3) * 8
    quality -= abs(bias) * 4
    quality = int(clamp(quality, 0, 100))
    return {
        "samples": len(finished),
        "mae": round(mae, 2),
        "bias": round(bias, 2),
        "hit_rate": hit_rate,
        "quality_score": quality
    }

def apply_calibration_adjustment(k_rate, calibration_profile, enabled=True):
    if not enabled:
        return k_rate, "Calibration adjustment disabled"
    if not calibration_profile or calibration_profile.get("samples", 0) < 10:
        return k_rate, "Calibration sample too small; no adjustment"
    bias = safe_float(calibration_profile.get("bias"), 0) or 0
    factor = clamp(1 + (bias * 0.01), 0.96, 1.04)
    return clamp(k_rate * factor, 0.08, 0.50), f"Historical calibration adjustment x{factor:.3f}"

# =========================
# MLB DATA
# =========================
def target_dates(day_mode):
    now = california_now()
    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if day_mode == "Today":
        return [today]
    if day_mode == "Tomorrow":
        return [tomorrow]
    return [today, tomorrow]

@st.cache_data(ttl=300, show_spinner=False)
def get_schedule(date_str):
    return safe_get_json(
        f"{MLB_BASE}/schedule",
        params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,venue,team"}
    ) or {"dates": []}

def extract_probable_pitchers(date_str):
    sched = get_schedule(date_str)
    rows = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            game_pk = g.get("gamePk")
            teams = g.get("teams", {})
            away = teams.get("away", {}).get("team", {})
            home = teams.get("home", {}).get("team", {})
            away_pp = teams.get("away", {}).get("probablePitcher")
            home_pp = teams.get("home", {}).get("probablePitcher")
            status = g.get("status", {}).get("abstractGameState", "Preview")
            game_time = g.get("gameDate", "")
            venue = g.get("venue", {}).get("name", "")

            if away_pp:
                rows.append({
                    "date": date_str,
                    "game_pk": game_pk,
                    "game_time": game_time,
                    "status": status,
                    "venue": venue,
                    "pitcher_id": away_pp.get("id"),
                    "pitcher": away_pp.get("fullName"),
                    "hand": away_pp.get("pitchHand", {}).get("code", "R"),
                    "team": away.get("abbreviation", away.get("name")),
                    "team_id": away.get("id"),
                    "opponent": home.get("abbreviation", home.get("name")),
                    "opp_team_id": home.get("id"),
                    "home_team": home.get("name"),
                    "away_team": away.get("name"),
                    "opp_side": "home",
                    "matchup": f"{away.get('abbreviation', away.get('name'))} @ {home.get('abbreviation', home.get('name'))}",
                    "pitcher_confirmed": True
                })
            if home_pp:
                rows.append({
                    "date": date_str,
                    "game_pk": game_pk,
                    "game_time": game_time,
                    "status": status,
                    "venue": venue,
                    "pitcher_id": home_pp.get("id"),
                    "pitcher": home_pp.get("fullName"),
                    "hand": home_pp.get("pitchHand", {}).get("code", "R"),
                    "team": home.get("abbreviation", home.get("name")),
                    "team_id": home.get("id"),
                    "opponent": away.get("abbreviation", away.get("name")),
                    "opp_team_id": away.get("id"),
                    "home_team": home.get("name"),
                    "away_team": away.get("name"),
                    "opp_side": "away",
                    "matchup": f"{away.get('abbreviation', away.get('name'))} @ {home.get('abbreviation', home.get('name'))}",
                    "pitcher_confirmed": True
                })
    for _p in locals().get("board", locals().get("rows", locals().get("out", []))):
        if isinstance(_p, dict) and "prop_rows" in _p:
            _p["prop_rows"] = clean_real_prop_debug_rows(_p.get("prop_rows", []))
    return rows

def get_pitcher_profile(pid):
    data = safe_get_json(
        f"{MLB_BASE}/people/{pid}/stats",
        params={"stats": "season", "group": "pitching"}
    )
    default = {"Pitcher K%": LEAGUE_AVG_K, "BF": 0, "SO": 0, "AVG IP": None, "K/9": None, "source": "Fallback league avg"}
    try:
        split = get_first_stat_split(data)
        if not split:
            return default
        stat = split.get("stat", {})
        ip = baseball_ip_to_float(stat.get("inningsPitched"))
        so = safe_float(stat.get("strikeOuts"), 0) or 0
        bf = safe_float(stat.get("battersFaced"), 0) or 0
        gs = safe_float(stat.get("gamesStarted"), None)
        gp = safe_float(stat.get("gamesPlayed"), 0) or 0
        starts = gs if gs and gs > 0 else gp
        k_pct = so / bf if bf > 0 else LEAGUE_AVG_K
        k9 = so / ip * 9 if ip and ip > 0 else None
        avg_ip = ip / starts if starts and starts > 0 and ip else None
        shrunk = ((k_pct * bf) + (LEAGUE_AVG_K * 150)) / max(bf + 150, 1)
        return {"Pitcher K%": float(clamp(shrunk, 0.08, 0.45)), "BF": bf, "SO": so, "AVG IP": avg_ip, "K/9": k9, "source": "Season K/BF with shrink"}
    except Exception:
        return default

def get_recent_logs(pid, n=12):
    data = safe_get_json(f"{MLB_BASE}/people/{pid}/stats", params={"stats": "gameLog", "group": "pitching"})
    rows = []
    try:
        splits = data["stats"][0]["splits"]
    except Exception:
        return rows
    for g in splits[:n]:
        stat = g.get("stat", {})
        ip_float = baseball_ip_to_float(stat.get("inningsPitched"))
        bf = safe_float(stat.get("battersFaced"))
        so = safe_float(stat.get("strikeOuts"))
        pitches = safe_float(stat.get("numberOfPitches"))
        rows.append({
            "Date": g.get("date"),
            "Opponent": g.get("opponent", {}).get("name"),
            "IP": stat.get("inningsPitched"),
            "IP_float": ip_float,
            "Ks": so,
            "BF": bf,
            "Pitches": pitches,
            "K%": None if not bf else round((so or 0) / bf * 100, 1)
        })
    return rows

def build_leash_model(recent_rows):
    """Projected batters faced with a safer pitcher-leash model."""
    if not recent_rows:
        return {
            "expected_bf": DEFAULT_BF,
            "ppb": 3.9,
            "recent_ip": 5.5,
            "last_10_ks": [],
            "leash_risk": "UNKNOWN",
            "source": "Default fallback"
        }

    df = pd.DataFrame(recent_rows)

    def mean_col(col, rows=None):
        try:
            x = df[col] if rows is None else df.head(rows)[col]
            x = pd.to_numeric(x, errors="coerce").dropna()
            return float(x.mean()) if len(x) else None
        except Exception:
            return None

    avg_bf_l10 = mean_col("BF")
    avg_bf_l5 = mean_col("BF", 5)
    avg_bf_l3 = mean_col("BF", 3)
    avg_ip_l3 = mean_col("IP_float", 3)
    avg_pitches_l3 = mean_col("Pitches", 3)
    avg_pitches_l5 = mean_col("Pitches", 5)

    if avg_bf_l3 and avg_bf_l5 and avg_bf_l10:
        expected_bf = avg_bf_l3 * 0.55 + avg_bf_l5 * 0.30 + avg_bf_l10 * 0.15
        source = "Weighted L3/L5/L10 BF"
    elif avg_bf_l3 and avg_bf_l10:
        expected_bf = avg_bf_l3 * 0.65 + avg_bf_l10 * 0.35
        source = "Weighted L3/L10 BF"
    elif avg_bf_l3:
        expected_bf = avg_bf_l3
        source = "Last 3 BF"
    elif avg_bf_l10:
        expected_bf = avg_bf_l10
        source = "Last 10 BF"
    else:
        expected_bf = DEFAULT_BF
        source = "Default fallback"

    ppb = 3.9
    if avg_pitches_l3 and avg_bf_l3 and avg_bf_l3 > 0:
        ppb = avg_pitches_l3 / avg_bf_l3

    leash_risk = "NORMAL"

    # v9.7 stricter leash: volume is the biggest source of false OVER confidence.
    if ppb >= 4.25:
        expected_bf -= 2.7
        leash_risk = "HIGH_PITCH_COUNT"
    elif ppb >= 4.05:
        expected_bf -= 1.4
        leash_risk = "MILD_PITCH_COUNT"

    # Recent short starts reduce leash confidence more aggressively.
    if avg_ip_l3 is not None and avg_ip_l3 < 5.0:
        expected_bf -= 2.1
        leash_risk = "SHORT_RECENT_STARTS"

    # Recent very high pitch workload: stronger fatigue haircut.
    if avg_pitches_l5 is not None and avg_pitches_l5 > 95:
        expected_bf -= 1.4
        leash_risk = "HIGH_RECENT_WORKLOAD"

    return {
        "expected_bf": float(clamp(expected_bf, 14, 31)),
        "ppb": float(ppb),
        "recent_ip": float(avg_ip_l3 or 5.5),
        "last_10_ks": [safe_int(r.get("Ks"), 0) or 0 for r in recent_rows[:10]],
        "leash_risk": leash_risk,
        "source": source
    }

def blend_pitcher_k_rate(profile_k, recent_rows, pitcher_id):
    profile_k = profile_k if profile_k is not None else LEAGUE_AVG_K
    recent_rates = []
    for r in recent_rows[:5]:
        bf = safe_float(r.get("BF"))
        ks = safe_float(r.get("Ks"))
        if bf and bf > 0 and ks is not None:
            recent_rates.append(ks / bf)
    if recent_rates:
        l5 = float(np.mean(recent_rates))
        base = profile_k * 0.70 + l5 * 0.30
        source = "Season K% + recent-start K% blend"
    else:
        base = profile_k
        source = "Season pitcher K%"
    learned, scale = apply_learning(pitcher_id, base)
    return clamp(learned, 0.08, 0.48), source, scale

def calculate_log5_k_rate(pitcher_k, lineup_k, league_avg_k=LEAGUE_AVG_K):
    pitcher_k = clamp(pitcher_k, 0.01, 0.60)
    lineup_k = clamp(lineup_k, 0.01, 0.60)
    num = (pitcher_k * lineup_k) / league_avg_k
    den = num + ((1 - pitcher_k) * (1 - lineup_k)) / (1 - league_avg_k)
    return float(num / den)

# =========================
# LINEUP / BATTER K
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def get_batter_season_k_rate(player_id):
    data = safe_get_json(f"{MLB_BASE}/people/{player_id}/stats", params={"stats": "season", "group": "hitting"})
    try:
        split = get_first_stat_split(data)
        if not split:
            return None, None, None
        stat = split.get("stat", {})
        so = safe_float(stat.get("strikeOuts"), 0) or 0
        pa = safe_float(stat.get("plateAppearances"), 0) or 0
        ab = safe_float(stat.get("atBats"), 0) or 0
        denom = pa if pa and pa > 0 else ab
        return (so / denom if denom and denom > 0 else None), so, denom
    except Exception:
        return None, None, None

@st.cache_data(ttl=600, show_spinner=False)
def get_batter_k_rate_vs_pitcher_hand(player_id, pitcher_hand):
    if not player_id or pitcher_hand not in ["R", "L"]:
        return None, None, None, "No pitcher hand"
    sit_code = "vrhp" if pitcher_hand == "R" else "vlhp"
    urls = [
        (f"{MLB_BASE}/people/{player_id}/stats", {"stats": "statSplits", "group": "hitting", "sitCodes": sit_code}),
        (f"{MLB_BASE}/people/{player_id}/stats", {"stats": "season", "group": "hitting", "sitCodes": sit_code}),
    ]
    for url, params in urls:
        data = safe_get_json(url, params=params)
        if not isinstance(data, dict):
            continue
        stats = data.get("stats") or []
        for block in stats:
            for split in (block.get("splits") or []):
                stat = split.get("stat") or {}
                so = safe_float(stat.get("strikeOuts"), 0) or 0
                pa = safe_float(stat.get("plateAppearances"), 0) or 0
                ab = safe_float(stat.get("atBats"), 0) or 0
                denom = pa if pa and pa > 0 else ab
                if denom and denom >= 10:
                    return float(so / denom), so, denom, f"Real split vs {'RHP' if pitcher_hand == 'R' else 'LHP'}"
    return None, None, None, "Split unavailable"


@st.cache_data(ttl=21600, show_spinner=False)
def get_batter_rolling_k_rates(player_id, days_list=(14, 30)):
    """Real rolling hitter K rates from MLB game logs.

    Returns only rates supported by real PA/SO game-log rows. Missing data gets no fake weight.
    """
    result = {int(d): None for d in days_list}
    if not player_id:
        return result
    data = safe_get_json(f"{MLB_BASE}/people/{player_id}/stats", params={"stats": "gameLog", "group": "hitting"})
    if not isinstance(data, dict):
        return result
    stats = data.get("stats") or []
    if not stats or not isinstance(stats[0], dict):
        return result
    splits = stats[0].get("splits") or []
    if not splits:
        return result
    today_dt = datetime.utcnow().date()
    for window in days_list:
        so_total, pa_total = 0.0, 0.0
        for g in splits:
            try:
                gdate = datetime.strptime(g.get("date", ""), "%Y-%m-%d").date()
            except Exception:
                continue
            age = (today_dt - gdate).days
            if age < 0 or age > int(window):
                continue
            stat = g.get("stat") or {}
            so = safe_float(stat.get("strikeOuts"), 0) or 0
            pa = safe_float(stat.get("plateAppearances"), 0) or 0
            if pa <= 0:
                pa = safe_float(stat.get("atBats"), 0) or 0
            so_total += so
            pa_total += pa
        if pa_total >= 8:
            result[int(window)] = float(so_total / pa_total)
    return result

def blend_batter_k_inputs(season_k, split_k=None, season_pa=None, split_pa=None, rolling14=None, rolling30=None):
    """Blend only real batter K inputs. Missing parts get zero weight."""
    parts = []
    if split_k is not None:
        # hand split is most matchup-specific, but still sample-sensitive
        split_weight = min(max((split_pa or 25) / 160, 0.20), 0.50)
        parts.append((float(split_k), split_weight, "hand split"))
    if rolling14 is not None:
        parts.append((float(rolling14), 0.25, "rolling 14d"))
    if rolling30 is not None:
        parts.append((float(rolling30), 0.15, "rolling 30d"))
    if season_k is not None:
        season_weight = min(max((season_pa or 50) / 300, 0.25), 0.45)
        parts.append((float(season_k), season_weight, "season"))
    if not parts:
        return None, "No batter K data"
    total_w = sum(w for _, w, _ in parts)
    blended = sum(v * w for v, w, _ in parts) / max(total_w, 1e-9)
    sources = ", ".join(src for _, _, src in parts)
    return clamp(blended, 0.04, 0.55), f"Blended real K inputs: {sources}"


def lineup_cache_key(game_pk, opp_side, pitcher_hand):
    return f"{game_pk}_{opp_side}_{pitcher_hand or 'NA'}"

def get_cached_lineup_rows(game_pk, opp_side, pitcher_hand):
    cache = load_json(LINEUP_CACHE_FILE, {})
    rec = cache.get(lineup_cache_key(game_pk, opp_side, pitcher_hand))
    return rec.get("rows", []) if rec else []

def set_cached_lineup_rows(game_pk, opp_side, pitcher_hand, rows):
    if not rows:
        return
    cache = load_json(LINEUP_CACHE_FILE, {})
    cache[lineup_cache_key(game_pk, opp_side, pitcher_hand)] = {"saved_at": now_iso(), "rows": rows[:9]}
    save_json(LINEUP_CACHE_FILE, cache)

@st.cache_data(ttl=300, show_spinner=False)
def calculate_lineup_k_rate(game_pk, opp_side, pitcher_hand=None):
    box = safe_get_json(f"{MLB_BASE}/game/{game_pk}/boxscore")
    if not box:
        cached_rows = get_cached_lineup_rows(game_pk, opp_side, pitcher_hand)
        valid_cached = [r.get("Raw_K_Rate") for r in cached_rows[:9] if r.get("Raw_K_Rate") is not None]
        if len(valid_cached) >= 5:
            return float(np.mean(valid_cached)), cached_rows[:9], "Using cached locked lineup", True
        return None, [], "Boxscore not available", False
    players = box.get("teams", {}).get(opp_side, {}).get("players", {})
    rows = []
    for _, pdata in players.items():
        order = pdata.get("battingOrder")
        if not order:
            continue
        person = pdata.get("person", {})
        player_id = person.get("id")
        name = person.get("fullName")
        season_k, season_so, season_pa = get_batter_season_k_rate(player_id)
        split_k, split_so, split_pa, split_source = get_batter_k_rate_vs_pitcher_hand(player_id, pitcher_hand) if pitcher_hand else (None, None, None, "No split")
        rolling = get_batter_rolling_k_rates(player_id, days_list=(14, 30))
        rolling14 = rolling.get(14)
        rolling30 = rolling.get(30)
        used_k, used_source = blend_batter_k_inputs(
            season_k,
            split_k=split_k,
            season_pa=season_pa,
            split_pa=split_pa,
            rolling14=rolling14,
            rolling30=rolling30,
        )
        if used_k is None:
            used_k = split_k if split_k is not None else season_k
            used_source = split_source if split_k is not None else "Season batter K%"
        rows.append({
            "Order": int(str(order)[:3]),
            "Batter": name,
            "Player ID": player_id,
            "Season K%": None if season_k is None else round(season_k * 100, 1),
            "Split K%": None if split_k is None else round(split_k * 100, 1),
            "Rolling 14d K%": None if rolling14 is None else round(rolling14 * 100, 1),
            "Rolling 30d K%": None if rolling30 is None else round(rolling30 * 100, 1),
            "Split PA/AB": split_pa,
            "Used K%": None if used_k is None else round(used_k * 100, 1),
            "K Source": used_source,
            "SO": season_so,
            "PA/AB": season_pa,
            "Raw_K_Rate": used_k
        })
    rows = sorted(rows, key=lambda x: x["Order"])
    valid = [r["Raw_K_Rate"] for r in rows[:9] if r["Raw_K_Rate"] is not None]
    if len(valid) >= 5:
        set_cached_lineup_rows(game_pk, opp_side, pitcher_hand, rows[:9])
        lineup_k = float(np.mean(valid))
        split_count = sum(1 for r in rows[:9] if r.get("Split K%") is not None)
        msg = f"Posted lineup K%; splits for {split_count}/9 hitters"
        return lineup_k, rows[:9], msg, len(rows[:9]) >= 8
    cached_rows = get_cached_lineup_rows(game_pk, opp_side, pitcher_hand)
    valid_cached = [r.get("Raw_K_Rate") for r in cached_rows[:9] if r.get("Raw_K_Rate") is not None]
    if len(valid_cached) >= 5:
        return float(np.mean(valid_cached)), cached_rows[:9], "Current lineup thin; using cached locked lineup", True
    return None, rows, "Lineup not posted or not enough hitter K data", False

def team_k_vs_hand(team_id, hand):
    data = safe_get_json(f"{MLB_BASE}/teams/{team_id}/stats", params={"stats": "season", "group": "hitting"})
    try:
        split = get_first_stat_split(data)
        if not split:
            return LEAGUE_AVG_K, "League average fallback"
        stat = split.get("stat", {})
        so = safe_float(stat.get("strikeOuts"), 0) or 0
        pa = safe_float(stat.get("plateAppearances"), 0) or 0
        if pa > 0:
            return float(so / pa), "Team season K/PA fallback"
    except Exception:
        pass
    return LEAGUE_AVG_K, "League average fallback"

# =========================
# STATCAST
# =========================
@st.cache_data(ttl=21600, show_spinner=False)
def get_statcast_pitch_profile(pitcher_id, days=365):
    empty = {"available": False, "message": "No pitcher id", "rows": 0, "csw": None, "whiff": None, "pitch_mix": [], "pitch_type_profile": [], "putaway": None}
    if not pitcher_id:
        return empty
    end = datetime.now()
    start = end - timedelta(days=int(days))
    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    params = {
        "all": "true",
        "player_type": "pitcher",
        "pitchers_lookup[]": str(pitcher_id),
        "game_date_gt": start.strftime("%Y-%m-%d"),
        "game_date_lt": end.strftime("%Y-%m-%d"),
        "type": "details",
    }
    try:
        r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text.strip():
            empty["message"] = f"Statcast HTTP {r.status_code}"
            return empty
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        if df.empty or "description" not in df.columns:
            empty["message"] = "Statcast returned no pitch rows"
            return empty
        desc = df["description"].astype(str).str.lower()
        pitch_count = int(len(df))
        called_mask = desc.eq("called_strike")
        whiff_mask = desc.isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])
        swing_mask = desc.isin(["swinging_strike", "swinging_strike_blocked", "foul_tip", "foul", "foul_bunt", "missed_bunt", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"])
        called = int(called_mask.sum())
        whiffs_n = int(whiff_mask.sum())
        swings = int(swing_mask.sum())
        csw = (called + whiffs_n) / pitch_count if pitch_count else None
        whiff = whiffs_n / swings if swings else None
        pitch_mix = []
        pitch_type_profile = []
        if "pitch_type" in df.columns:
            df2 = df.copy()
            df2["pitch_type"] = df2["pitch_type"].fillna("UNK").astype(str)
            df2["_called"] = called_mask.astype(int)
            df2["_whiff"] = whiff_mask.astype(int)
            df2["_swing"] = swing_mask.astype(int)
            total = max(len(df2), 1)
            grouped = df2.groupby("pitch_type").agg(Pitches=("pitch_type", "size"), Called=("_called", "sum"), Whiffs=("_whiff", "sum"), Swings=("_swing", "sum")).reset_index()
            grouped["Usage"] = grouped["Pitches"] / total
            grouped["CSW"] = (grouped["Called"] + grouped["Whiffs"]) / grouped["Pitches"].replace(0, np.nan)
            grouped["WhiffRate"] = grouped["Whiffs"] / grouped["Swings"].replace(0, np.nan)
            grouped = grouped.sort_values("Usage", ascending=False).head(8)
            for _, row in grouped.iterrows():
                pt = str(row["pitch_type"])
                usage = safe_float(row["Usage"], 0) or 0
                wr = safe_float(row["WhiffRate"])
                csw_rate = safe_float(row["CSW"])
                pitch_mix.append({"Pitch Type": pt, "Usage %": round(usage * 100, 1)})
                pitch_type_profile.append({
                    "Pitch Type": pt,
                    "Usage %": round(usage * 100, 1),
                    "Pitcher Whiff%": None if wr is None or pd.isna(wr) else round(wr * 100, 1),
                    "Pitcher CSW%": None if csw_rate is None or pd.isna(csw_rate) else round(csw_rate * 100, 1),
                    "Pitches": int(row["Pitches"]),
                    "Swings": int(row["Swings"]),
                })
        return {"available": True, "message": "Real Statcast pitch-level data loaded", "rows": pitch_count, "csw": None if csw is None else float(csw), "whiff": None if whiff is None else float(whiff), "pitch_mix": pitch_mix, "pitch_type_profile": pitch_type_profile}
    except Exception as e:
        empty["message"] = f"Statcast unavailable: {e}"
        return empty

def apply_statcast_csw_adjustment(pitcher_k, statcast_profile, enabled=True):
    if not enabled or not statcast_profile or not statcast_profile.get("available"):
        return pitcher_k, "No Statcast adjustment"
    csw = statcast_profile.get("csw")
    if csw is None:
        return pitcher_k, "No Statcast CSW available"
    factor = clamp(1 + ((float(csw) - 0.275) * 0.45), 0.93, 1.07)
    return clamp(pitcher_k * factor, 0.08, 0.50), f"Real Statcast CSW adjustment x{factor:.3f}"

def apply_pitch_type_matchup_adjustment(pitcher_k, pitcher_statcast, enabled=True):
    if not enabled or not pitcher_statcast or not pitcher_statcast.get("available"):
        return pitcher_k, "No pitch-type matchup adjustment", False, [], 1.0
    # Conservative simplified pitch-type factor from pitcher whiff vs league ref.
    rows = []
    weighted = 0
    total_w = 0
    for r in pitcher_statcast.get("pitch_type_profile", []):
        pt = r.get("Pitch Type")
        usage = (safe_float(r.get("Usage %"), 0) or 0) / 100
        wr = safe_float(r.get("Pitcher Whiff%"))
        ref = LEAGUE_AVG_WHIFF_BY_PITCH_TYPE.get(pt, 0.25)
        if usage >= 0.03 and wr is not None:
            idx = clamp((wr / 100) / max(ref, 0.01), 0.85, 1.18)
            weighted += usage * idx
            total_w += usage
            rows.append({"Pitch Type": pt, "Usage %": round(usage * 100, 1), "Pitcher Whiff%": wr, "League Ref Whiff%": round(ref * 100, 1), "Index": round(idx, 3)})
    if total_w <= 0:
        return pitcher_k, "Pitch-type rows unavailable", False, rows, 1.0
    combined = weighted / total_w
    factor = clamp(1 + ((combined - 1) * 0.08), 0.97, 1.03)
    return clamp(pitcher_k * factor, 0.08, 0.50), f"Pitch-type whiff mix adjustment x{factor:.3f}", True, rows, factor



@st.cache_data(ttl=21600, show_spinner=False)
def get_batter_statcast_pitch_type_profile(batter_id, days=365, pitcher_hand=None):
    """Real batter whiff profile by pitch type from Baseball Savant.

    This never estimates missing data. If Statcast is unavailable or too thin, no adjustment is applied.
    """
    empty = {"available": False, "message": "No batter id", "rows": 0, "pitch_type_profile": []}
    if not batter_id:
        return empty
    end = datetime.now()
    start = end - timedelta(days=int(days))
    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    params = {
        "all": "true",
        "player_type": "batter",
        "batters_lookup[]": str(batter_id),
        "game_date_gt": start.strftime("%Y-%m-%d"),
        "game_date_lt": end.strftime("%Y-%m-%d"),
        "type": "details",
    }
    try:
        r = requests.get(url, params=params, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text.strip():
            empty["message"] = f"Batter Statcast HTTP {r.status_code}"
            return empty
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        if df.empty or "description" not in df.columns or "pitch_type" not in df.columns:
            empty["message"] = "Batter Statcast returned no pitch-type rows"
            return empty
        # Use hand split only if the split sample is not tiny. Otherwise use all pitcher hands.
        if pitcher_hand in ["R", "L"] and "p_throws" in df.columns:
            hand_df = df[df["p_throws"].astype(str).str.upper() == pitcher_hand].copy()
            if len(hand_df) >= 25:
                df = hand_df
        desc = df["description"].astype(str).str.lower()
        whiff_mask = desc.isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])
        swing_mask = desc.isin([
            "swinging_strike", "swinging_strike_blocked", "foul_tip", "foul", "foul_bunt",
            "missed_bunt", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"
        ])
        df2 = df.copy()
        df2["pitch_type"] = df2["pitch_type"].fillna("UNK").astype(str)
        df2["_whiff"] = whiff_mask.astype(int)
        df2["_swing"] = swing_mask.astype(int)
        grouped = df2.groupby("pitch_type").agg(
            Pitches=("pitch_type", "size"),
            Whiffs=("_whiff", "sum"),
            Swings=("_swing", "sum"),
        ).reset_index()
        grouped = grouped[grouped["Swings"] >= 5]
        if grouped.empty:
            empty["message"] = "Batter Statcast has too few swings by pitch type"
            return empty
        grouped["WhiffRate"] = grouped["Whiffs"] / grouped["Swings"].replace(0, np.nan)
        profile = []
        for _, row in grouped.iterrows():
            wr = safe_float(row["WhiffRate"])
            if wr is None or pd.isna(wr):
                continue
            profile.append({
                "Pitch Type": str(row["pitch_type"]),
                "Batter Whiff%": round(wr * 100, 1),
                "Swings": int(row["Swings"]),
                "Pitches Seen": int(row["Pitches"]),
            })
        if not profile:
            empty["message"] = "No batter pitch-type whiff rows passed sample filter"
            return empty
        return {"available": True, "message": "Real batter Statcast pitch-type whiff loaded", "rows": int(len(df)), "pitch_type_profile": profile}
    except Exception as e:
        empty["message"] = f"Batter Statcast unavailable: {e}"
        return empty


def build_pitch_type_matchup_profile(pitcher_statcast, lineup_rows, enabled=True, min_batters=5, pitcher_hand=None):
    """Compare real pitcher pitch mix to real batter whiff by pitch type.

    Applies only when enough real batter Statcast profiles load. Missing pitch types are ignored, not guessed.
    """
    result = {"available": False, "factor": 1.0, "message": "Pitch-type matchup disabled or unavailable", "rows": [], "batters_loaded": 0}
    if not enabled:
        result["message"] = "Pitch-type matchup disabled"
        return result
    if not pitcher_statcast or not pitcher_statcast.get("available"):
        result["message"] = "Pitcher Statcast pitch mix unavailable"
        return result
    pitch_profile = pitcher_statcast.get("pitch_type_profile") or []
    if not pitch_profile:
        result["message"] = "Pitcher pitch-type profile unavailable"
        return result
    if not lineup_rows:
        result["message"] = "No posted lineup for batter pitch-type matching"
        return result

    pitcher_usage = {r.get("Pitch Type"): (safe_float(r.get("Usage %"), 0) or 0) / 100.0 for r in pitch_profile}
    pitcher_whiff = {
        r.get("Pitch Type"): (safe_float(r.get("Pitcher Whiff%")) / 100.0 if safe_float(r.get("Pitcher Whiff%")) is not None else None)
        for r in pitch_profile
    }
    pitch_types = [pt for pt, use in pitcher_usage.items() if pt and use >= 0.03]

    batter_profiles = []
    for r in lineup_rows[:9]:
        bid = r.get("Player ID")
        prof = get_batter_statcast_pitch_type_profile(bid, days=365, pitcher_hand=pitcher_hand)
        if prof.get("available"):
            by_pt = {x.get("Pitch Type"): x for x in prof.get("pitch_type_profile", [])}
            batter_profiles.append({"Batter": r.get("Batter"), "by_pt": by_pt})
    result["batters_loaded"] = len(batter_profiles)
    if len(batter_profiles) < min_batters:
        result["message"] = f"Only {len(batter_profiles)}/9 batter pitch-type profiles loaded; no adjustment applied"
        return result

    rows = []
    weighted_index = 0.0
    used_weight = 0.0
    for pt in pitch_types:
        use = pitcher_usage.get(pt, 0) or 0
        batter_rates = []
        batter_swings = 0
        for bp in batter_profiles:
            row = bp["by_pt"].get(pt)
            if not row:
                continue
            wr = safe_float(row.get("Batter Whiff%"))
            swings = safe_int(row.get("Swings"), 0) or 0
            if wr is not None and swings >= 5:
                batter_rates.append(wr / 100.0)
                batter_swings += swings
        if len(batter_rates) < 3:
            continue
        avg_batter_whiff = float(np.mean(batter_rates))
        league_ref = LEAGUE_AVG_WHIFF_BY_PITCH_TYPE.get(pt, 0.25)
        pitcher_wr = pitcher_whiff.get(pt)
        pitcher_bonus = 1.0
        if pitcher_wr is not None:
            pitcher_bonus = clamp(pitcher_wr / max(league_ref, 0.01), 0.85, 1.18)
        batter_index = avg_batter_whiff / max(league_ref, 0.01)
        combined_index = clamp((batter_index * 0.70) + (pitcher_bonus * 0.30), 0.82, 1.22)
        weighted_index += use * combined_index
        used_weight += use
        rows.append({
            "Pitch Type": pt,
            "Pitcher Usage %": round(use * 100, 1),
            "Avg Batter Whiff%": round(avg_batter_whiff * 100, 1),
            "League Ref Whiff%": round(league_ref * 100, 1),
            "Pitcher Whiff%": None if pitcher_wr is None else round(pitcher_wr * 100, 1),
            "Index": round(combined_index, 3),
            "Batter Profiles Used": len(batter_rates),
            "Batter Swings": batter_swings,
        })
    if used_weight <= 0 or not rows:
        result["message"] = "No overlapping pitcher/batter pitch-type rows passed sample filter"
        return result
    avg_index = weighted_index / used_weight
    factor = clamp(1 + ((avg_index - 1) * 0.10), 0.965, 1.035)
    result.update({
        "available": True,
        "factor": factor,
        "message": f"Real batter-vs-pitch-type matchup x{factor:.3f} ({len(batter_profiles)}/9 batters loaded)",
        "rows": rows,
    })
    return result


def apply_advanced_pitch_type_matchup_adjustment(pitcher_k, matchup_profile, enabled=True):
    if not enabled or not matchup_profile or not matchup_profile.get("available"):
        msg = matchup_profile.get("message", "No batter-vs-pitch-type matchup adjustment") if matchup_profile else "No batter-vs-pitch-type matchup adjustment"
        return pitcher_k, msg
    factor = safe_float(matchup_profile.get("factor"), 1.0) or 1.0
    return clamp(pitcher_k * factor, 0.08, 0.50), matchup_profile.get("message", f"Pitch-type matchup x{factor:.3f}")

# =========================
# SIMULATION
# =========================
def park_k_factor(venue_name):
    """Small, conservative park adjustment. Missing venue stays neutral."""
    v = normalize_name(venue_name)
    park_map = {
        "tropicana field": 1.025,
        "loan depot park": 1.015,
        "oracle park": 1.010,
        "petco park": 1.010,
        "t mobile park": 1.010,
        "citi field": 1.010,
        "pnc park": 1.008,
        "coors field": 0.965,
        "great american ball park": 0.985,
        "fenway park": 0.990,
        "citizens bank park": 0.990,
        "yankee stadium": 0.990,
        "globe life field": 1.005,
    }
    for name, factor in park_map.items():
        if name in v:
            return factor
    return 1.00

def market_movement_k_factor(open_to_current_delta, active_line=None, enabled=True):
    """Tiny market movement signal merged from file #2.

    Important: this never selects, overwrites, or locks a prop line. The first file's
    Underdog/player matching remains the source of truth. Positive line movement
    gives a small K boost; negative movement gives a small haircut.
    """
    if not enabled:
        return 1.0, "Market movement adjustment off"
    delta = safe_float(open_to_current_delta)
    if delta is None:
        return 1.0, "No market movement history yet; neutral"
    # Keep this conservative because line movement can reflect price, availability, or alternate ladders.
    if delta >= 0.5:
        factor = 1.010
    elif delta <= -0.5:
        factor = 0.990
    elif delta > 0:
        factor = 1.004
    elif delta < 0:
        factor = 0.996
    else:
        factor = 1.0
    factor = float(clamp(factor, MARKET_MOVE_FACTOR_MIN, MARKET_MOVE_FACTOR_MAX))
    return factor, f"Conservative market movement x{factor:.3f} from open-to-current line delta {delta:+.2f}"

# MLB venue coordinates for live weather. Indoor/retractable parks default neutral.
VENUE_WEATHER_META = {
    "angel stadium": (33.8003, -117.8827, False),
    "busch stadium": (38.6226, -90.1928, False),
    "camden yards": (39.2839, -76.6217, False),
    "citizens bank park": (39.9061, -75.1665, False),
    "coors field": (39.7559, -104.9942, False),
    "dodger stadium": (34.0739, -118.2400, False),
    "fenway park": (42.3467, -71.0972, False),
    "great american ball park": (39.0979, -84.5066, False),
    "guaranteed rate field": (41.8300, -87.6339, False),
    "kauffman stadium": (39.0517, -94.4803, False),
    "loan depot park": (25.7781, -80.2197, True),
    "minute maid park": (29.7572, -95.3555, True),
    "nationals park": (38.8730, -77.0074, False),
    "oracle park": (37.7786, -122.3893, False),
    "petco park": (32.7073, -117.1573, False),
    "pnc park": (40.4469, -80.0057, False),
    "progressive field": (41.4962, -81.6852, False),
    "rogers centre": (43.6414, -79.3894, True),
    "sutter health park": (38.5803, -121.5139, False),
    "target field": (44.9817, -93.2776, False),
    "t mobile park": (47.5914, -122.3325, True),
    "tropicana field": (27.7682, -82.6534, True),
    "truist park": (33.8908, -84.4678, False),
    "wrigley field": (41.9484, -87.6553, False),
    "yankee stadium": (40.8296, -73.9262, False),
    "american family field": (43.0280, -87.9712, True),
    "chase field": (33.4455, -112.0667, True),
    "citi field": (40.7571, -73.8458, False),
    "comerica park": (42.3390, -83.0485, False),
    "globe life field": (32.7473, -97.0842, True),
}

def venue_weather_meta(venue_name):
    v = normalize_name(venue_name)
    for name, meta in VENUE_WEATHER_META.items():
        if name in v:
            return meta
    return None

def parse_game_hour_pt(game_time):
    try:
        s = str(game_time or "").replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if pytz and dt.tzinfo is not None:
            dt = dt.astimezone(pytz.timezone("America/Los_Angeles"))
        return dt.strftime("%Y-%m-%dT%H:00")
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def get_open_meteo_hourly(lat, lon, date_str):
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/Los_Angeles",
            "start_date": date_str,
            "end_date": date_str,
        }
        return safe_get_json("https://api.open-meteo.com/v1/forecast", params=params, timeout=12) or {}
    except Exception as e:
        log_source_request("OpenMeteo", "ERROR", str(e))
        return {}

def weather_k_factor(venue_name, game_time, enabled=True):
    """Conservative live weather K factor.

    Weather only nudges K probability slightly and defaults neutral when unavailable.
    Indoor/retractable parks are neutral because roof status is often unknown.
    """
    if not enabled:
        return 1.0, "Weather adjustment off", {}
    meta = venue_weather_meta(venue_name)
    if not meta:
        return 1.0, "Weather unavailable for venue; neutral", {}
    lat, lon, indoor = meta
    if indoor:
        return 1.0, "Indoor/retractable venue; weather neutral", {"indoor": True}
    try:
        date_str = str(game_time or "")[:10]
        hour_key = parse_game_hour_pt(game_time)
        data = get_open_meteo_hourly(lat, lon, date_str)
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return 1.0, "Weather feed empty; neutral", {}
        idx = 0
        if hour_key in times:
            idx = times.index(hour_key)
        else:
            # nearest available hour by string distance fallback
            idx = min(range(len(times)), key=lambda i: abs(i - len(times)//2))
        temp = safe_float((hourly.get("temperature_2m") or [None])[idx])
        wind = safe_float((hourly.get("wind_speed_10m") or [None])[idx])
        humidity = safe_float((hourly.get("relative_humidity_2m") or [None])[idx])
        precip = safe_float((hourly.get("precipitation_probability") or [None])[idx])

        factor = 1.0
        # Cold air can help pitchers slightly; extreme heat can reduce stamina/command slightly.
        if temp is not None:
            if temp <= 55:
                factor += 0.006
            elif temp >= 88:
                factor -= 0.008
        # Strong wind can increase run environment/long innings; tiny K haircut.
        if wind is not None and wind >= 15:
            factor -= 0.006
        # Very high humidity/precip risk can affect grip/command; tiny K haircut.
        if humidity is not None and humidity >= 80:
            factor -= 0.004
        if precip is not None and precip >= 35:
            factor -= 0.006

        factor = float(clamp(factor, WEATHER_FACTOR_MIN, WEATHER_FACTOR_MAX))
        details = {"temp_f": temp, "wind_mph": wind, "humidity": humidity, "precip_prob": precip, "indoor": False}
        note = f"Weather x{factor:.3f}: {temp if temp is not None else 'NA'}F, wind {wind if wind is not None else 'NA'} mph, humidity {humidity if humidity is not None else 'NA'}%, precip {precip if precip is not None else 'NA'}%"
        return factor, note, details
    except Exception as e:
        return 1.0, f"Weather error; neutral: {e}", {}

# Conservative umpire K tendency table. Missing/unknown umps stay neutral.
UMPIRE_K_TENDENCY = {
    "Lance Barrett": 1.020,
    "Mark Wegner": 1.018,
    "Pat Hoberg": 1.015,
    "Adam Hamari": 1.012,
    "Ryan Blakney": 1.010,
    "Bill Miller": 0.982,
    "Chris Segal": 0.985,
    "Angel Hernandez": 0.990,
    "Laz Diaz": 0.990,
    "CB Bucknor": 0.992,
}

def umpire_factor(game_pk, enabled=True):
    if not enabled:
        return 1.00, "Umpire adjustment off", "Umpire adjustment off"
    data = safe_get_json(f"{MLB_LIVE}/game/{game_pk}/feed/live")
    try:
        officials = data["liveData"]["boxscore"].get("officials", [])
        name = officials[0]["official"]["fullName"] if officials else "Unknown"
        raw = safe_float(UMPIRE_K_TENDENCY.get(name), 1.0) or 1.0
        factor = float(clamp(raw, UMPIRE_FACTOR_MIN, UMPIRE_FACTOR_MAX))
        if name == "Unknown":
            return 1.00, name, "Umpire unknown; neutral"
        return factor, name, f"Umpire K tendency x{factor:.3f} ({name})"
    except Exception:
        return 1.00, "Unknown", "Umpire unavailable; neutral"

def build_pa_sequence(lineup_rows, bf, fallback_k):
    bf = int(round(bf))
    if lineup_rows:
        rates = [r.get("Raw_K_Rate") for r in lineup_rows[:9] if r.get("Raw_K_Rate") is not None]
        if len(rates) >= 5:
            return [rates[i % len(rates)] for i in range(max(1, bf))], "Batter-by-batter posted lineup"
    return [fallback_k for _ in range(max(1, bf))], "Team/fallback K sequence"

def simulate_matchup(pitcher_k, batter_rates, park=1.0, ump=1.0, sims=12000):
    rates = []
    for br in batter_rates:
        k = calculate_log5_k_rate(pitcher_k, br)
        k *= park * ump
        rates.append(clamp(k, 0.03, 0.60))
    out = np.random.binomial(1, np.array(rates), size=(sims, len(rates))).sum(axis=1)
    return out, rates


def bayesian_projection_std(data_score, lineup_locked, pitcher_confirmed, leash):
    """Dynamic uncertainty for K simulations.

    Higher data quality = tighter distribution. Missing lineup, unconfirmed pitcher,
    or leash risk = wider uncertainty. This does not create edge; it usually shrinks
    extreme confidence back toward reality.
    """
    score = safe_float(data_score, 50) or 50
    std = 1.25 - (score / 100.0) * 0.55
    if not lineup_locked:
        std += 0.28
    if not pitcher_confirmed:
        std += 0.32
    if leash and leash.get("leash_risk") in ["HIGH_PITCH_COUNT", "SHORT_RECENT_STARTS", "HIGH_RECENT_WORKLOAD"]:
        std += 0.25
    ppb = safe_float((leash or {}).get("ppb"), 4.0) or 4.0
    if ppb >= 4.15:
        std += 0.15
    return float(clamp(std, BAYESIAN_PROJECTION_STD_MIN, BAYESIAN_PROJECTION_STD_MAX))


def simulate_bayesian_markov_matchup(pitcher_k, batter_rates, expected_bf, park=1.0, ump=1.0, data_score=50, lineup_locked=False, pitcher_confirmed=True, leash=None, sims=BAYESIAN_MARKOV_SIMS):
    """MLB-specific Bayesian + Markov Monte Carlo.

    This keeps our current batter-by-batter K probabilities, but adds realistic uncertainty:
    - starter volume uncertainty around expected BF
    - pitcher K-rate uncertainty based on data quality/leash
    - PA-by-PA Markov flow instead of fixed 27 outs
    """
    base_rates = []
    for br in batter_rates:
        k = calculate_log5_k_rate(pitcher_k, br)
        base_rates.append(clamp(k * park * ump, 0.03, 0.60))

    if not base_rates:
        base_rates = [clamp(pitcher_k * park * ump, 0.03, 0.60)] * int(max(1, round(expected_bf or DEFAULT_BF)))

    data_score = safe_float(data_score, 50) or 50
    proj_std = bayesian_projection_std(data_score, lineup_locked, pitcher_confirmed, leash)
    expected_bf = safe_float(expected_bf, DEFAULT_BF) or DEFAULT_BF

    # Better score -> tighter BF range. Risky leash -> wider BF range.
    bf_sd = 1.25 + (1 - data_score / 100.0) * 2.0
    if leash and leash.get("leash_risk") in ["HIGH_PITCH_COUNT", "SHORT_RECENT_STARTS", "HIGH_RECENT_WORKLOAD"]:
        bf_sd += 1.2

    # Convert projection-level uncertainty into a conservative multiplier on PA K probabilities.
    baseline_projection = max(sum(base_rates[:int(round(expected_bf))]), 0.25)
    mult_sd = clamp(proj_std / max(baseline_projection, 1.0), 0.04, 0.22)

    results = np.zeros(int(sims), dtype=float)
    rates_arr = np.array(base_rates, dtype=float)
    n_rates = len(rates_arr)

    for i in range(int(sims)):
        sampled_bf = int(round(np.random.normal(expected_bf, bf_sd)))
        sampled_bf = int(clamp(sampled_bf, 12, 34))
        k_mult = float(np.random.normal(1.0, mult_sd))
        k_mult = clamp(k_mult, 0.72, 1.28)
        idx = np.arange(sampled_bf) % n_rates
        probs = np.clip(rates_arr[idx] * k_mult, 0.02, 0.68)
        results[i] = np.random.binomial(1, probs).sum()

    note = f"Bayesian Markov MC: sims={int(sims)}, BF μ={expected_bf:.1f}, BF σ={bf_sd:.2f}, K σ={proj_std:.2f}"
    return results, base_rates, note


XGB_FEATURES = [
    "projection", "pitcher_k", "opp_k", "expected_bf", "ppb", "recent_ip",
    "data_score", "lineup_locked", "pitcher_confirmed", "statcast_available",
    "statcast_csw", "statcast_whiff", "pitch_type_matchup_available", "pitch_type_factor",
    "consensus_count", "consensus_spread"
]


def xgb_feature_row_from_picklike(d):
    def b(v):
        return 1.0 if bool(v) else 0.0
    return {
        "projection": safe_float(d.get("projection"), 0) or 0,
        "pitcher_k": safe_float(d.get("pitcher_k"), LEAGUE_AVG_K) or LEAGUE_AVG_K,
        "opp_k": safe_float(d.get("opp_k"), LEAGUE_AVG_K) or LEAGUE_AVG_K,
        "expected_bf": safe_float(d.get("expected_bf"), DEFAULT_BF) or DEFAULT_BF,
        "ppb": safe_float(d.get("ppb"), 4.0) or 4.0,
        "recent_ip": safe_float(d.get("recent_ip"), 5.5) or 5.5,
        "data_score": safe_float(d.get("data_score"), 50) or 50,
        "lineup_locked": b(d.get("lineup_locked")),
        "pitcher_confirmed": b(d.get("pitcher_confirmed")),
        "statcast_available": b(d.get("statcast_available")),
        "statcast_csw": safe_float(d.get("statcast_csw"), 0) or 0,
        "statcast_whiff": safe_float(d.get("statcast_whiff"), 0) or 0,
        "pitch_type_matchup_available": b(d.get("pitch_type_matchup_available")),
        "pitch_type_factor": safe_float(d.get("pitch_type_factor"), 1.0) or 1.0,
        "consensus_count": safe_float(d.get("consensus_count"), 0) or 0,
        "consensus_spread": safe_float(d.get("consensus_spread"), 0) or 0,
    }


def build_xgb_training_frame():
    """Train on our own graded official snapshots only.

    Target is residual actual Ks - existing projection, so XGBoost can only act
    as a correction layer. It does not replace the core model.
    """
    results = load_json(RESULT_LOG, [])
    rows = []
    for r in results[-XGB_RECENT_TRAIN_LIMIT:]:
        actual = safe_float(r.get("actual"))
        proj = safe_float(r.get("projection"))
        if actual is None or proj is None:
            continue
        if r.get("graded_result") not in ["WIN", "LOSS"]:
            continue
        feat = xgb_feature_row_from_picklike(r)
        feat["target_residual"] = float(clamp(actual - proj, -4.0, 4.0))
        rows.append(feat)
    return pd.DataFrame(rows)


def apply_xgboost_assist(current_features, current_projection, enabled=False):
    """Optional capped XGBoost correction.

    OFF by default. Activates only after enough graded picks and only changes
    the projection by a small capped amount. It cannot affect line source,
    Underdog lock, or strict no-bet gates.
    """
    info = {
        "enabled": bool(enabled),
        "active": False,
        "samples": 0,
        "adjustment": 0.0,
        "message": "XGBoost assist off",
    }
    base = safe_float(current_projection, 0) or 0
    if not enabled:
        return base, info

    df = build_xgb_training_frame()
    info["samples"] = int(len(df))
    if len(df) < XGB_MIN_GRADED_SAMPLES:
        info["message"] = f"Need {XGB_MIN_GRADED_SAMPLES}+ graded picks; found {len(df)}"
        return base, info

    try:
        from xgboost import XGBRegressor
    except Exception as e:
        info["message"] = f"xgboost not installed: {e}"
        return base, info

    try:
        train_df = df.copy()
        X = train_df[XGB_FEATURES].fillna(0.0)
        y = train_df["target_residual"].astype(float)
        model = XGBRegressor(
            n_estimators=160,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X, y)
        cur = pd.DataFrame([current_features])[XGB_FEATURES].fillna(0.0)
        raw_adj = float(model.predict(cur)[0])
        cap = min(XGB_MAX_RESIDUAL_ADJ_KS, abs(base) * XGB_MAX_PERCENT_ADJ)
        adj = float(clamp(raw_adj, -cap, cap))
        info.update({
            "active": True,
            "adjustment": round(adj, 3),
            "message": f"XGBoost residual assist active: raw {raw_adj:+.2f}, capped {adj:+.2f} K from {len(df)} samples",
        })
        return float(clamp(base + adj, 0.0, 15.0)), info
    except Exception as e:
        info["message"] = f"XGBoost assist error: {e}"
        return base, info

def calculate_pick_metrics(sims, line):
    if line is None:
        return {"over_prob": None, "under_prob": None, "fair_prob": None, "pick_side": "NO LINE", "edge": None, "grade": "NO LINE", "ev": None}
    over_prob = float(np.mean(sims > line))
    under_prob = 1 - over_prob
    if over_prob >= under_prob:
        side = "OVER"
        fair = over_prob
    else:
        side = "UNDER"
        fair = under_prob
    edge = (fair - 0.50) * 100
    grade = "S" if fair >= 0.68 else "A" if fair >= 0.60 else "B" if fair >= 0.55 else "C"
    return {"over_prob": over_prob, "under_prob": under_prob, "fair_prob": fair, "pick_side": side, "edge": edge, "grade": grade, "ev": (fair * 100) - ((1 - fair) * 100)}

# =========================
# REAL PROP SOURCES
# =========================
def source_result(source, status, line=None, rows=None, message=""):
    return {"source": source, "status": status, "line": safe_float(line), "rows": rows or [], "message": message}


def clean_real_prop_debug_rows(rows):
    """Display/storage filter: only valid MLB pitcher strikeout prop rows.

    Wrong-sport Underdog rows like LeBron/Shai NBA props are dropped here even
    if they made it through another source's raw/debug output.
    """
    cleaned = []
    nba_name_block = {
        "lebron james", "shai gilgeous alexander", "james harden", "donovan mitchell",
        "anthony edwards", "nikola jokic", "luka doncic", "jayson tatum",
        "stephen curry", "kevin durant", "giannis antetokounmpo", "victor wembanyama"
    }

    for r in rows or []:
        if not isinstance(r, dict):
            continue

        matched = str(r.get("Matched Name", r.get("matched_name", r.get("Player", ""))) or "")
        matched_norm = normalize_name(matched)
        if matched_norm in nba_name_block:
            continue
        if any(n in matched_norm for n in nba_name_block):
            continue

        line = safe_float(
            r.get("Line", r.get("line", r.get("Prop Line", r.get("line_display"))))
        )
        market = str(r.get("Market", r.get("market", "")) or "")
        blob = " ".join(str(v) for v in r.values())[:4000]

        if is_bad_sport_text(blob):
            continue
        if is_valid_k_line(line, allow_integer=False) is None:
            continue
        if is_bad_k_market_text(blob):
            continue

        # Accepted rows usually have Market = Pitcher Strikeouts. For raw rows,
        # require strikeout text in the blob.
        if market:
            if not is_pitcher_k_text(market) and not is_pitcher_k_text(blob):
                continue
        elif not is_pitcher_k_text(blob):
            continue

        cleaned.append(r)

    return cleaned


# v10.8.1 read-only wrapper: logs debug rows but never changes them.
try:
    _original_clean_real_prop_debug_rows_for_debug = clean_real_prop_debug_rows
    def clean_real_prop_debug_rows(rows):
        cleaned = _original_clean_real_prop_debug_rows_for_debug(rows)
        log_raw_prop_debug_rows(cleaned, source_hint="clean_real_prop_debug_rows")
        return cleaned
except NameError:
    pass


def is_half_point_line(line):
    """True for normal no-push prop lines like 4.5, 5.5, 6.5."""
    val = safe_float(line)
    if val is None:
        return False
    return 1.5 <= val <= 12.5 and abs(val % 1 - 0.5) < 1e-9


def is_valid_k_line(line, allow_integer=False):
    """Validate MLB pitcher strikeout prop line.

    Underdog pick'em lines should normally be half-point lines. Integers are accepted only
    for priced sportsbook/alternate markets where pushes can exist.
    """
    val = safe_float(line)
    if val is None:
        return None
    if not (1.5 <= val <= 12.5):
        return None
    if abs(val * 2 - round(val * 2)) > 1e-9:
        return None
    if not allow_integer and not is_half_point_line(val):
        return None
    return float(val)


def extract_half_lines_from_text(text):
    """Pull likely half-point K lines from title/display text, preferring values near strikeout words."""
    import re
    if not text:
        return []
    t = str(text)
    low = t.lower()
    if not any(k in low for k in ["strikeout", "strikeouts", "pitcher k", "pitcher_k"]):
        return []
    vals = []
    # Prefer half numbers because Underdog uses half-lines to avoid pushes.
    for m in re.finditer(r"(?<!\d)(\d{1,2}\.5)(?!\d)", t):
        val = safe_float(m.group(1))
        if is_valid_k_line(val, allow_integer=False) is not None:
            vals.append(float(val))
    return vals

@st.cache_data(ttl=600, show_spinner=False)
def get_odds_events():
    if not ODDS_API_KEY:
        return []
    data = safe_get_json(f"{ODDS_BASE}/sports/baseball_mlb/events", params={"apiKey": ODDS_API_KEY}, timeout=16)
    return data if isinstance(data, list) else []

@st.cache_data(ttl=600, show_spinner=False)
def get_sportsbook_event_pitcher_k_lines(event_id, player_name):
    if not event_id:
        return source_result("Sportsbook", "NO EVENT", rows=[], message="No matching Odds API event id")
    data = safe_get_json(
        f"{ODDS_BASE}/sports/baseball_mlb/events/{event_id}/odds",
        params={"apiKey": ODDS_API_KEY, "regions": "us,us2,uk,eu,au", "markets": ",".join(SPORTSBOOK_PITCHER_K_MARKETS), "oddsFormat": "american"},
        timeout=16
    )
    if not data or (isinstance(data, dict) and data.get("message")):
        return source_result("Sportsbook", "FAILED", rows=[], message="Event odds call failed or plan has no player props")
    rows = []
    for book in data.get("bookmakers", []):
        book_name = book.get("title") or book.get("key") or "Sportsbook"
        for market in book.get("markets", []):
            if market.get("key") not in SPORTSBOOK_PITCHER_K_MARKETS:
                continue
            for outcome in market.get("outcomes", []):
                desc = outcome.get("description") or outcome.get("player") or outcome.get("participant") or outcome.get("name") or ""
                score = name_score(player_name, desc)
                if score < 0.80:
                    continue
                point = safe_float(outcome.get("point"))
                if point is None:
                    continue
                rows.append({"Source": "OddsAPI", "Provider": book_name, "Player": player_name, "Matched Name": desc, "Match Score": round(score, 3), "Market": market.get("key"), "Line": point, "Side": str(outcome.get("name", "")).upper(), "Price": outcome.get("price"), "Last Update": market.get("last_update") or book.get("last_update")})
    if not rows:
        return source_result("Sportsbook", "NO MATCH", rows=[], message="No sportsbook K prop matched this pitcher")
    line_vals = [safe_float(r["Line"]) for r in rows if safe_float(r.get("Line")) is not None]
    consensus = float(np.median(line_vals)) if line_vals else rows[0]["Line"]
    return source_result("Sportsbook", "FOUND", line=consensus, rows=rows, message=f"Found {len(rows)} sportsbook outcomes")

def get_sportsbook_k_data(game_home, game_away, player_name):
    events = get_odds_events()
    event_id = None
    target_teams = {normalize_name(game_home), normalize_name(game_away)}
    for ev in events:
        home = normalize_name(ev.get("home_team"))
        away = normalize_name(ev.get("away_team"))
        if {home, away} == target_teams or (home in target_teams and away in target_teams):
            event_id = ev.get("id")
            break
    return get_sportsbook_event_pitcher_k_lines(event_id, player_name)

@st.cache_data(ttl=600, show_spinner=False)
def get_prizepicks_k_data(player_name):
    data = safe_get_json(PRIZEPICKS_URL, timeout=16)
    if not data:
        return source_result("PrizePicks", "FAILED", message="API failed or returned no JSON")
    players = {}
    for inc in data.get("included", []):
        inc_type = inc.get("type", "")
        attrs = inc.get("attributes", {}) or {}
        if inc_type in ["new_player", "player"]:
            pid = str(inc.get("id"))
            name = attrs.get("name") or attrs.get("display_name") or attrs.get("full_name")
            league = attrs.get("league") or attrs.get("league_name") or attrs.get("sport") or ""
            team = attrs.get("team") or attrs.get("team_name") or ""
            if pid and name:
                players[pid] = {"name": name, "league": league, "team": team}
    rows = []
    for item in data.get("data", []):
        attrs = item.get("attributes", {}) or {}
        stat_type = attrs.get("stat_type") or attrs.get("stat_display_name") or attrs.get("name") or ""
        if not is_pitcher_k_text(stat_type):
            continue
        line_score = safe_float(attrs.get("line_score") or attrs.get("line") or attrs.get("projection"))
        if line_score is None:
            continue
        rel = item.get("relationships", {}) or {}
        pdata = (rel.get("new_player", {}) or {}).get("data") or (rel.get("player", {}) or {}).get("data") or {}
        pid = str(pdata.get("id", ""))
        info = players.get(pid, {})
        pp_name = info.get("name") or attrs.get("player_name") or attrs.get("description") or ""
        league_blob = f"{info.get('league','')} {attrs.get('league','')} {attrs.get('league_name','')} {attrs.get('sport','')}".lower()
        if league_blob.strip() and not any(x in league_blob for x in ["mlb", "baseball"]):
            continue
        score = name_score(player_name, pp_name)
        if score >= 0.80:
            rows.append({"Source": "PrizePicks", "Provider": "PrizePicks", "Player": player_name, "Matched Name": pp_name, "Team": info.get("team", ""), "League": info.get("league", ""), "Market": stat_type, "Line": line_score, "Side": "OVER/UNDER", "Price": None, "Match Score": round(score, 3), "Start Time": attrs.get("start_time"), "Projection ID": item.get("id")})
    if not rows:
        return source_result("PrizePicks", "NO MATCH", message="No fuzzy pitcher strikeout prop match found")
    rows = sorted(rows, key=lambda r: -r.get("Match Score", 0))
    return source_result("PrizePicks", "FOUND", line=rows[0]["Line"], rows=rows, message=f"Found {len(rows)} PrizePicks matches")

def extract_prop_rows_from_any_json(data, player_name, source_name):
    rows = []
    if not data:
        return rows
    objects = flatten_json(data)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        blob = json.dumps(obj, default=str).lower()
        if is_bad_sport_text(blob) or is_bad_k_market_text(blob):
            continue
        if not ("pitcher strikeout" in blob or "pitcher strikeouts" in blob or "pitcher k" in blob or "pitcher_k" in blob or "strikeouts" in blob):
            continue
        candidate_bits = []
        for key in ["player", "player_name", "participant", "participant_name", "name", "description", "display_name", "market_name", "selection", "title"]:
            val = obj.get(key)
            if isinstance(val, dict):
                val = val.get("name") or val.get("full_name") or val.get("display_name")
            if val:
                candidate_bits.append(str(val))
        candidate = " ".join(candidate_bits) or blob[:160]
        score = name_score(player_name, candidate)
        if score < 0.80 and normalize_name(player_name) in normalize_name(blob):
            score = 0.82
        if score < 0.80:
            continue
        line = safe_float(first_value(obj, ["stat_value", "target_value", "over_under_line", "line_score", "line", "point", "handicap"]))
        line = is_valid_k_line(line, allow_integer=True)
        if line is None:
            continue
        side = first_value(obj, ["side", "label", "name", "selection", "outcome", "bet_type"]) or "Over/Under"
        price = safe_float(first_value(obj, ["price", "odds", "american_odds", "american", "over_price", "under_price"]))
        book = first_value(obj, ["sportsbook", "book", "bookmaker", "operator", "source"]) or source_name
        if isinstance(book, dict):
            book = book.get("name") or source_name
        rows.append({"Source": source_name, "Provider": str(book), "Player": player_name, "Matched Name": candidate[:120], "Match Score": round(score, 3), "Market": first_value(obj, ["market", "market_name", "stat", "stat_type", "prop", "category"]) or "Pitcher Strikeouts", "Side": str(side).upper(), "Line": line, "Price": price})
    dedup = {}
    for r in rows:
        key = (r.get("Provider"), r.get("Source"), str(r.get("Side")).lower(), r.get("Line"), r.get("Price"))
        dedup[key] = r
    return list(dedup.values())

def get_underdog_k_data(player_name):
    """Live Underdog parser for MLB pitcher strikeout props.

    v10 upgrade:
    - Still tries the safe relationship path first: line -> over_under -> appearance -> player.
    - If Underdog changes nesting or omits type labels, falls back to a recursive parser.
    - Accepts active Underdog K lines when the player name and strikeout market are clearly present.
    - Keeps NBA/WNBA/fantasy/team props blocked.
    """
    accepted_rows = []
    rejected_rows = []
    last_msg = ""
    target_norm = normalize_name(player_name)

    LINE_TYPES = {"over_under_line", "over_under_lines"}
    OU_TYPES = {"over_under", "over_unders"}
    APP_TYPES = {"appearance", "appearances"}
    PLAYER_TYPES = {"player", "players"}

    def attrs(obj):
        if not isinstance(obj, dict):
            return {}
        out = {}
        a = obj.get("attributes")
        if isinstance(a, dict):
            out.update(a)
        for k, v in obj.items():
            if k not in ["attributes", "relationships", "included", "data"] and k not in out:
                out[k] = v
        return out

    def obj_type(obj, fallback=""):
        return str(obj.get("type") or fallback or "").lower().replace("-", "_") if isinstance(obj, dict) else ""

    def obj_id(obj):
        if not isinstance(obj, dict):
            return None
        val = obj.get("id") or attrs(obj).get("id")
        return str(val) if val not in [None, ""] else None

    def rel_id(obj, rel_names):
        if not isinstance(obj, dict):
            return None
        rels = obj.get("relationships") or {}
        for name in rel_names:
            candidates = [name, name.replace("_", "-"), name.replace("_", "")]
            for cname in candidates:
                if cname not in rels:
                    continue
                node = rels.get(cname)
                data = node.get("data") if isinstance(node, dict) else node
                if isinstance(data, dict):
                    rid = data.get("id")
                    if rid not in [None, ""]:
                        return str(rid)
                if isinstance(data, list) and data:
                    for item in data:
                        if isinstance(item, dict) and item.get("id") not in [None, ""]:
                            return str(item.get("id"))
        return None

    def collect_objects(data):
        objects = []
        def walk(x, parent_key=""):
            if isinstance(x, dict):
                y = dict(x)
                if parent_key and "_parent_key" not in y:
                    y["_parent_key"] = parent_key
                objects.append(y)
                for k, v in x.items():
                    walk(v, k)
            elif isinstance(x, list):
                for item in x:
                    walk(item, parent_key)
        walk(data)
        return objects

    def text_from(*objs):
        parts = []
        wanted = [
            "title", "display_title", "name", "player_name", "full_name", "first_name", "last_name",
            "display_name", "stat", "stat_type", "appearance_stat", "display_stat", "label", "market",
            "market_name", "sport", "league", "sport_name", "league_name", "position", "description",
            "over_under", "over_under_title", "scoring_type", "projection_type"
        ]
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            a = attrs(obj)
            for k in wanted:
                v = a.get(k)
                if isinstance(v, dict):
                    for kk in wanted:
                        if v.get(kk) not in [None, ""]:
                            parts.append(str(v.get(kk)))
                elif v not in [None, ""]:
                    parts.append(str(v))
        return " | ".join(parts)

    def player_name_from(player_obj, appearance_obj=None, line_obj=None, ou_obj=None):
        p = attrs(player_obj) if isinstance(player_obj, dict) else {}
        a = attrs(appearance_obj) if isinstance(appearance_obj, dict) else {}
        l = attrs(line_obj) if isinstance(line_obj, dict) else {}
        o = attrs(ou_obj) if isinstance(ou_obj, dict) else {}
        candidates = [
            p.get("display_name"), p.get("full_name"), p.get("name"), p.get("player_name"),
            p.get("short_name"), p.get("abbreviation"), p.get("abbr_name"),
            (str(p.get("first_name", "")).strip() + " " + str(p.get("last_name", "")).strip()).strip(),
            a.get("player_name"), a.get("full_name"), a.get("display_name"), a.get("title"), a.get("name"),
            a.get("short_name"), a.get("abbreviation"), a.get("abbr_name"),
            l.get("player_name"), l.get("full_name"), l.get("display_name"), l.get("title"), l.get("name"),
            l.get("short_name"), l.get("abbreviation"), l.get("abbr_name"),
            o.get("player_name"), o.get("full_name"), o.get("display_name"), o.get("title"), o.get("name"),
            o.get("short_name"), o.get("abbreviation"), o.get("abbr_name"),
        ]
        for c in candidates:
            if c and normalize_name(c):
                return str(c)
        return ""

    def line_from_obj(*objs):
        # Underdog displayed K lines should come from real line fields only.
        # Do NOT use generic points/point/value/total fields; those caused wrong lines.
        safe_keys = ["stat_value", "line_score", "over_under_line", "target_value"]
        for obj in objs:
            a = attrs(obj)
            for k in safe_keys:
                val = safe_float(a.get(k))
                if is_valid_k_line(val, allow_integer=False) is not None:
                    return float(val), f"{k} half-line from Underdog object"
        text_lines = extract_half_lines_from_text(" | ".join(text_from(o) for o in objs))
        if text_lines:
            return float(text_lines[0]), "half-line from Underdog text"
        return None, "no valid Underdog half-line"

    def blob_from(*objs):
        return " | ".join([text_from(o) for o in objs if isinstance(o, dict)]).lower()

    def is_bad_sport(blob):
        return is_bad_sport_text(blob)

    def is_pitcher_k_blob(blob):
        blob = blob.lower()
        if not any(x in blob for x in ["pitcher strikeout", "pitcher strikeouts", "pitcher_k", "pitcher k", "strikeouts", "strike outs"]):
            return False
        return not is_bad_k_market_text(blob)

    def active_status_ok(*objs):
        status_blob = " ".join(
            str(attrs(o).get(k, ""))
            for o in objs if isinstance(o, dict)
            for k in ["status", "state", "display_status", "over_status", "under_status", "hidden", "active"]
        ).lower()
        if any(x in status_blob for x in ["suspended", "removed", "hidden", "inactive", "closed", "disabled"]):
            return False
        return True

    def underdog_player_score(actual_player, evidence):
        score = max(name_score(player_name, actual_player), name_score(player_name, evidence))
        # Strong fallback for Underdog display names that use first initial + last name.
        # Example: MLB probable pitcher = "Cristopher Sanchez"; Underdog row = "C. Sánchez".
        t_parts = target_norm.split()
        if len(t_parts) >= 2:
            target_initial = t_parts[0][:1]
            target_last = t_parts[-1]
            evidence_norm = normalize_name(evidence)
            # Look for "c sanchez", "c. sanchez", or any blob containing the last name with matching initial.
            if target_last in evidence_norm:
                tokens = evidence_norm.split()
                for i, tok in enumerate(tokens):
                    if tok == target_last and i > 0 and tokens[i - 1][:1] == target_initial:
                        score = max(score, 0.93)
                    if tok == target_last and target_initial in evidence_norm:
                        score = max(score, 0.88)
        if target_norm and target_norm in normalize_name(evidence):
            score = max(score, 0.94)
        return score

    def add_row(line, score, matched, evidence, line_note, path, source_mode):
        accepted_rows.append({
            "Source": "Underdog",
            "Provider": "Underdog",
            "Player": player_name,
            "Matched Name": (matched or evidence[:120]),
            "Match Score": round(float(score), 3),
            "Market": "Pitcher Strikeouts",
            "Side": "OVER/UNDER",
            "Line": float(line),
            "Price": None,
            "Line Evidence": line_note,
            "Parser Mode": source_mode,
            "Underdog Path": path,
        })

    for url in UNDERDOG_URLS:
        data = safe_get_json(url, timeout=18)
        if not data:
            last_msg = f"No JSON from {url}"
            continue

        objects = collect_objects(data)
        by_id_any = {}
        over_unders, appearances, players, line_candidates = {}, {}, {}, []

        for obj in objects:
            typ = obj_type(obj, obj.get("_parent_key", ""))
            oid = obj_id(obj)
            if oid:
                by_id_any[oid] = obj
            if typ in LINE_TYPES or "over_under_line" in typ:
                line_candidates.append(obj)
            elif typ in OU_TYPES or typ == "over_under":
                if oid:
                    over_unders[oid] = obj
            elif typ in APP_TYPES or "appearance" in typ:
                if oid:
                    appearances[oid] = obj
            elif typ in PLAYER_TYPES or typ == "player":
                if oid:
                    players[oid] = obj

        def get_by_id(oid):
            return by_id_any.get(str(oid)) if oid not in [None, ""] else None

        # Relationship parser first.
        if not line_candidates:
            for obj in objects:
                a = attrs(obj)
                if any(a.get(k) not in [None, ""] for k in ["stat_value", "line_score", "over_under_line", "target_value", "line", "points"]):
                    if isinstance(obj.get("relationships"), dict) or is_pitcher_k_blob(json.dumps(obj, default=str).lower()):
                        line_candidates.append(obj)

        for line_obj in line_candidates:
            ou_id = rel_id(line_obj, ["over_under", "overUnders", "over_under_id", "over"])
            ou_obj = over_unders.get(ou_id) or get_by_id(ou_id)

            app_id = rel_id(line_obj, ["appearance", "appearances", "appearance_id"])
            if not app_id and isinstance(ou_obj, dict):
                app_id = rel_id(ou_obj, ["appearance", "appearances", "appearance_id"])
            app_obj = appearances.get(app_id) or get_by_id(app_id)

            player_id = rel_id(line_obj, ["player", "players", "player_id"])
            if not player_id and isinstance(ou_obj, dict):
                player_id = rel_id(ou_obj, ["player", "players", "player_id"])
            if not player_id and isinstance(app_obj, dict):
                player_id = rel_id(app_obj, ["player", "players", "player_id"])
            if not player_id and isinstance(app_obj, dict):
                player_id = attrs(app_obj).get("player_id") or attrs(app_obj).get("playerId")
            player_obj = players.get(str(player_id)) or get_by_id(player_id)

            evidence = text_from(line_obj, ou_obj, app_obj, player_obj)
            blob = evidence.lower()
            if is_bad_sport(blob):
                continue
            if not is_pitcher_k_blob(blob):
                # rejected row hidden intentionally
                continue

            actual_player = player_name_from(player_obj, app_obj, line_obj, ou_obj)
            score = underdog_player_score(actual_player, evidence)
            if score < 0.82:
                # rejected row hidden intentionally
                continue

            chosen_line, line_note = line_from_obj(line_obj, ou_obj)
            if chosen_line is None:
                # rejected row hidden intentionally
                continue
            if not active_status_ok(line_obj, ou_obj):
                continue
            add_row(chosen_line, score, actual_player, evidence, line_note, f"line:{obj_id(line_obj)} -> over_under:{ou_id} -> appearance:{app_id} -> player:{player_id}", "relationship")

        # Recursive fallback parser for new/changed Underdog JSON.
        # This is intentionally looser than relationship mode, but still requires:
        # target player name + strikeout market + sane K line + no bad sport/market words.
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            blob_json = json.dumps(obj, default=str)
            blob_low = blob_json.lower()
            if is_bad_sport(blob_low):
                continue
            if not is_pitcher_k_blob(blob_low):
                continue
            # Try candidate fields and the full object blob so abbreviated Underdog names match daily.
            cand = []
            for k in ["player", "player_name", "participant", "participant_name", "name", "description", "display_name", "title", "short_name", "abbreviation", "abbr_name"]:
                v = attrs(obj).get(k)
                if isinstance(v, dict):
                    v = v.get("name") or v.get("full_name") or v.get("display_name") or v.get("title") or v.get("short_name")
                if v:
                    cand.append(str(v))
            matched = " ".join(cand) or player_name
            score = max(underdog_player_score(matched, blob_json), name_score(player_name, matched))
            if score < 0.82:
                continue
            line, line_note = line_from_obj(obj)
            if line is None:
                continue
            if not active_status_ok(obj):
                continue
            add_row(line, score, matched, blob_json[:200], line_note, f"fallback:{obj_id(obj) or attrs(obj).get('id') or len(accepted_rows)}", "recursive fallback")

        if accepted_rows:
            break

    if not accepted_rows:
        return source_result("Underdog", "NO MATCH", rows=[], message=last_msg or "No active Underdog pitcher-K line matched. Rejected wrong-sport rows are hidden.")

    dedup = {}
    for r in accepted_rows:
        key = (r.get("Underdog Path"), r.get("Line"), r.get("Parser Mode"))
        if key not in dedup or safe_float(r.get("Match Score"), 0) > safe_float(dedup[key].get("Match Score"), 0):
            dedup[key] = r
    accepted_rows = list(dedup.values())

    # Pick the live Underdog board line.
    # v10.8.1 FIX:
    # The original code ranked by the highest line, which can accidentally choose an
    # alternate ladder line. We now prefer the clean relationship parser and choose the
    # first/highest-confidence direct half-line from the stable board instead of max(line).
    primary_rows = [r for r in accepted_rows if r.get("Parser Mode") == "relationship"] or accepted_rows
    half_rows = [r for r in primary_rows if is_half_point_line(r.get("Line"))] or primary_rows

    def row_rank(r):
        rel_bonus = 1 if r.get("Parser Mode") == "relationship" else 0
        direct_bonus = 1 if any(k in str(r.get("Line Evidence", "")).lower() for k in ["stat_value", "line_score", "over_under_line", "target_value"]) else 0
        half_bonus = 1 if is_half_point_line(r.get("Line")) else 0
        score = safe_float(r.get("Match Score"), 0) or 0
        # Do NOT rank by line value. That was the wrong-line bug.
        return (rel_bonus, direct_bonus, half_bonus, round(score, 3))

    ranked = sorted(half_rows, key=row_rank, reverse=True)
    top_rank = row_rank(ranked[0])
    tied = [r for r in ranked if row_rank(r) == top_rank]
    # If several tied relationship rows exist, use the median line, not the highest alt.
    tied_lines = sorted([safe_float(r.get("Line")) for r in tied if safe_float(r.get("Line")) is not None])
    if len(tied_lines) >= 3:
        target_line = tied_lines[len(tied_lines) // 2]
        best_row = sorted(tied, key=lambda r: abs((safe_float(r.get("Line")) or target_line) - target_line))[0]
    else:
        best_row = ranked[0]
    active = safe_float(best_row.get("Line"))

    return source_result(
        "Underdog",
        "FOUND",
        line=float(active),
        rows=sorted(accepted_rows, key=lambda r: (-safe_float(r.get("Match Score"), 0), safe_float(r.get("Line"), 99))),
        message=f"Live Underdog line matched: {float(active):.1f} via {best_row.get('Matched Name')} ({best_row.get('Parser Mode')}); rejected debug rows hidden to prevent wrong-sport noise"
    )

@st.cache_data(ttl=600, show_spinner=False)
def get_sportsgameodds_k_data(player_name):
    if not SPORTSGAMEODDS_API_KEY:
        return source_result("SportsGameOdds", "DISABLED", message="Add SPORTSGAMEODDS_API_KEY to enable")
    endpoints = [f"{SPORTSGAMEODDS_BASE}/events", f"{SPORTSGAMEODDS_BASE}/odds", f"{SPORTSGAMEODDS_BASE}/props"]
    headers = {"X-Api-Key": SPORTSGAMEODDS_API_KEY, "Authorization": f"Bearer {SPORTSGAMEODDS_API_KEY}"}
    all_rows = []
    last_msg = ""
    for url in endpoints:
        data = safe_get_json(url, params={"sport": "baseball", "league": "mlb", "market": "player_pitcher_strikeouts"}, headers=headers, timeout=16)
        if not data:
            last_msg = f"No JSON from {url}"
            continue
        all_rows.extend(extract_prop_rows_from_any_json(data, player_name, "SportsGameOdds"))
    if not all_rows:
        return source_result("SportsGameOdds", "NO MATCH", message=last_msg or "No SportsGameOdds row matched")
    lines = [safe_float(r.get("Line")) for r in all_rows if safe_float(r.get("Line")) is not None]
    return source_result("SportsGameOdds", "FOUND", line=float(np.median(lines)), rows=all_rows, message=f"Found {len(all_rows)} SportsGameOdds rows")

@st.cache_data(ttl=600, show_spinner=False)
def get_opticodds_k_data(player_name):
    if not OPTICODDS_API_KEY:
        return source_result("OpticOdds", "DISABLED", message="Add OPTICODDS_API_KEY to enable")
    endpoints = [f"{OPTICODDS_BASE}/fixtures/odds", f"{OPTICODDS_BASE}/odds", f"{OPTICODDS_BASE}/player-props"]
    headers = {"X-Api-Key": OPTICODDS_API_KEY, "Authorization": f"Bearer {OPTICODDS_API_KEY}"}
    all_rows = []
    last_msg = ""
    for url in endpoints:
        data = safe_get_json(url, params={"sport": "baseball", "league": "mlb", "market": "player_pitcher_strikeouts"}, headers=headers, timeout=16)
        if not data:
            last_msg = f"No JSON from {url}"
            continue
        all_rows.extend(extract_prop_rows_from_any_json(data, player_name, "OpticOdds"))
    if not all_rows:
        return source_result("OpticOdds", "NO MATCH", message=last_msg or "No OpticOdds row matched")
    lines = [safe_float(r.get("Line")) for r in all_rows if safe_float(r.get("Line")) is not None]
    return source_result("OpticOdds", "FOUND", line=float(np.median(lines)), rows=all_rows, message=f"Found {len(all_rows)} OpticOdds rows")

def choose_active_line(sportsbook_data, pp_data, ud_data, sgo_data, optic_data):
    """Choose a safe active line.

    For this app, Underdog is treated as the live source of truth when it has an exact
    half-point pitcher-K match. That prevents the app from showing 5 when Underdog is
    actually showing 4.5. Other sources remain available as backup/consensus.
    """
    candidates = []

    def add(source, line, weight, allow_integer=False):
        val = is_valid_k_line(line, allow_integer=allow_integer)
        if val is not None:
            candidates.append({"Source": source, "Line": val, "Weight": float(weight)})

    # Underdog first: user is comparing the app to live Underdog props.
    ud_line = is_valid_k_line(ud_data.get("line"), allow_integer=False)
    if ud_data.get("status") == "FOUND" and ud_line is not None:
        # Still collect other rows for diagnostics, but do not let consensus round/shift Underdog.
        add("Sportsbook", sportsbook_data.get("line"), 3.0, allow_integer=True)
        add("SportsGameOdds", sgo_data.get("line"), 2.5, allow_integer=True)
        add("OpticOdds", optic_data.get("line"), 2.5, allow_integer=True)
        add("PrizePicks", pp_data.get("line"), 1.5, allow_integer=False)
        add("Underdog", ud_line, 3.5, allow_integer=False)
        raw = [c["Line"] for c in candidates] or [ud_line]
        spread = float(max(raw) - min(raw)) if len(raw) > 1 else 0.0
        return float(ud_line), "Underdog Live Exact", {
            "count": len(candidates),
            "quality": "UNDERDOG_EXACT",
            "spread": round(spread, 2),
            "rows": candidates,
        }

    # Backup mode when Underdog has no exact match.
    add("Sportsbook", sportsbook_data.get("line"), 3.0, allow_integer=True)
    add("SportsGameOdds", sgo_data.get("line"), 2.5, allow_integer=True)
    add("OpticOdds", optic_data.get("line"), 2.5, allow_integer=True)
    add("PrizePicks", pp_data.get("line"), 1.5, allow_integer=False)

    if not candidates:
        return None, "No Valid Real Pitcher-K Line", {"count": 0, "quality": "NO LINE", "spread": None, "rows": []}

    raw_lines = [c["Line"] for c in candidates]
    spread = float(max(raw_lines) - min(raw_lines)) if len(candidates) > 1 else 0.0

    if len(candidates) >= 2 and spread > 1.0:
        priority = {"Sportsbook": 1, "SportsGameOdds": 2, "OpticOdds": 3, "PrizePicks": 4}
        best = sorted(candidates, key=lambda c: priority.get(c["Source"], 99))[0]
        return best["Line"], f"{best['Source']} Only (source disagreement blocked)", {
            "count": len(candidates), "quality": "DISAGREE", "spread": round(spread, 2), "rows": candidates
        }

    expanded = []
    for c in candidates:
        expanded.extend([c["Line"]] * max(1, int(round(c["Weight"] * 2))))
    consensus = float(np.median(expanded))

    # Do not create fake .0 lines from consensus if half-line sources dominate.
    half_candidates = [c["Line"] for c in candidates if is_half_point_line(c["Line"])]
    if half_candidates and not is_half_point_line(consensus):
        counts = {}
        for v in half_candidates:
            counts[v] = counts.get(v, 0) + 1
        consensus = sorted(counts.items(), key=lambda kv: (-kv[1], abs(kv[0] - consensus)))[0][0]

    quality = "STRONG" if len(candidates) >= 3 and spread <= 0.5 else "OK" if len(candidates) >= 2 and spread <= 1.0 else "THIN"
    source = "Cross-Source Consensus" if len(candidates) >= 2 else candidates[0]["Source"]
    return consensus, f"{source} ({quality})", {"count": len(candidates), "quality": quality, "spread": round(spread, 2), "rows": candidates}

# =========================
# CONFIDENCE / SIGNAL
# =========================
# CONFIDENCE / SIGNAL
# =========================
def data_lock_score(lineup_locked, pitcher_confirmed, active_line, consensus_info, ppb, statcast_available, pitch_type_available):
    score = 38
    if pitcher_confirmed:
        score += 15
    if lineup_locked:
        score += 20
    if active_line is not None:
        score += 15
    if consensus_info.get("count", 0) >= 3 and (consensus_info.get("spread") is None or consensus_info.get("spread") <= 0.5):
        score += 9
    elif consensus_info.get("count", 0) >= 2:
        score += 6
    if ppb and ppb < 4.05:
        score += 3
    elif ppb and ppb >= 4.25:
        score -= 5
    if statcast_available:
        score += 5
    if pitch_type_available:
        score += 3
    return int(clamp(score, 0, 100))

def shrink_probability_to_market(model_prob, score=50, lineup_locked=False, pitcher_confirmed=False):
    p = safe_float(model_prob)
    if p is None:
        return None

    # v9.7 market shrink: do not let simulations print fake 70%+ confidence.
    strength = 0.18 + (float(score or 50) / 100.0) * 0.48
    strength += 0.06 if lineup_locked else -0.12
    strength += 0.05 if pitcher_confirmed else -0.10
    strength = clamp(strength, 0.16, 0.82)

    capped = clamp(0.50 + ((p - 0.50) * strength), 0.01, 0.99)

    if not lineup_locked or not pitcher_confirmed:
        capped = min(capped, 0.68)
    elif score < MIN_CONFIRMED_LINEUP_SCORE:
        capped = min(capped, 0.76)

    return clamp(capped, 0.01, 0.99)

def no_bet_gate(active_line, pick_side, fair_prob, ev, gap, score, lineup_locked, pitcher_confirmed, line_source, consensus_info, leash):
    """Final hard filter. If any reason appears, the app must PASS.

    v9.7 is built to win by selectivity: fewer recommendations, stronger edge.
    """
    reasons = []
    consensus_info = consensus_info or {}
    leash = leash or {}
    ppb = safe_float(leash.get("ppb"), 4.0) or 4.0
    recent_ip = safe_float(leash.get("recent_ip"), 5.5) or 5.5

    if active_line is None:
        reasons.append("no real prop line")
    if pick_side not in ["OVER", "UNDER"]:
        reasons.append("no valid side")
    if fair_prob is None or fair_prob < MIN_BETTABLE_PROB:
        reasons.append(f"probability below {int(MIN_BETTABLE_PROB*100)}%")
    if ev is None or ev < MIN_BETTABLE_EV:
        reasons.append(f"EV below {round(MIN_BETTABLE_EV*100,1)}%")
    if gap is None or gap < MIN_BETTABLE_GAP_KS:
        reasons.append(f"edge below {MIN_BETTABLE_GAP_KS} K")
    if score < MIN_BETTABLE_SCORE:
        reasons.append(f"data score below {MIN_BETTABLE_SCORE}")
    if not pitcher_confirmed:
        reasons.append("pitcher not confirmed")

    # No confirmed lineup = never trust an OVER. Unders can survive only with all other gates.
    if not lineup_locked and pick_side == "OVER":
        reasons.append("no confirmed lineup for over")
    elif not lineup_locked:
        reasons.append("lineup not locked")

    if consensus_info.get("quality") in ["NO LINE", "REJECTED"]:
        reasons.append("no validated market consensus")
    if consensus_info.get("rejected"):
        reasons.append("one or more source lines rejected as outliers")
    if consensus_info.get("count", 0) < 2:
        reasons.append("not enough market sources")

    # Pitcher volume/leash is the main K-prop trap.
    if ppb >= 4.15:
        reasons.append("pitcher uses too many pitches per batter")
    if recent_ip < 4.8:
        reasons.append("recent innings too low")
    if leash.get("leash_risk") in ["HIGH_PITCH_COUNT", "SHORT_RECENT_STARTS", "HIGH_RECENT_WORKLOAD"]:
        reasons.append(f"leash risk: {leash.get('leash_risk')}")

    return len(reasons) == 0, reasons

def classify_risk(prob, score, priced, edge_pct, gap, line_source):
    p = safe_float(prob)
    if p is None:
        return "NO MODEL %", "No usable probability"

    pct = p * 100
    # v9.7: stop labeling weak props as playable. Only elite/strong survive visually.
    if pct >= 70 and score >= MIN_ELITE_DATA_SCORE and priced and edge_pct >= MIN_ELITE_NO_VIG_EDGE and gap >= 1.15:
        return "🔥 ELITE WATCH — VERIFY", "All strict real-data, price, gap, and market gates passed"
    if pct >= 64 and score >= MIN_BETTABLE_SCORE and priced and edge_pct >= MIN_BETTABLE_EV * 100 and gap >= MIN_BETTABLE_GAP_KS:
        return "✅ STRONG WATCH", "Playable only after final manual check: lineup, weather, pitcher status"

    notes = []
    if pct < MIN_BETTABLE_PROB * 100:
        notes.append(f"probability under {int(MIN_BETTABLE_PROB*100)}%")
    if score < MIN_BETTABLE_SCORE:
        notes.append(f"data score under {MIN_BETTABLE_SCORE}")
    if not priced:
        notes.append("no real sportsbook price")
    if edge_pct is not None and edge_pct < MIN_ELITE_NO_VIG_EDGE:
        notes.append("no-vig edge not elite")
    if "No Real Line" in str(line_source):
        notes.append("no real prop line")
    if gap is None or gap < MIN_BETTABLE_GAP_KS:
        notes.append(f"gap under {MIN_BETTABLE_GAP_KS} K")
    return "PASS / NO BET", "; ".join(notes) if notes else "Does not clear strict win filter"

def build_signal(proj, line, fair_prob, ev, ppb, score):
    if line is None:
        return "PASS — NO REAL LINE", "pass"
    gap = abs(proj - line)
    side = "OVER" if proj > line else "UNDER"
    ppb = safe_float(ppb, 4.0) or 4.0

    if (
        fair_prob is not None and fair_prob >= 0.68
        and gap >= 1.15
        and ev is not None and ev >= 0.08
        and score >= 92
        and ppb < 4.05
    ):
        return f"🔥 ELITE WATCH {side}", "good"

    if (
        fair_prob is not None and fair_prob >= 0.64
        and gap >= 1.00
        and ev is not None and ev >= 0.06
        and score >= 88
        and ppb < 4.10
    ):
        return f"✅ STRONG WATCH {side}", "good"

    return f"PASS — {side}", "pass"



def bullpen_workload_bf_factor(team_id):
    """Conservative team pitching workload proxy for starter leash.

    It only nudges expected batters faced slightly and never creates a fake edge.
    """
    data = safe_get_json(f"{MLB_BASE}/teams/{team_id}/stats", params={"stats": "season", "group": "pitching"})
    try:
        split = get_first_stat_split(data)
        if not split:
            return 1.0, "Bullpen/team workload unavailable"
        stat = split.get("stat", {})
        ip = baseball_ip_to_float(stat.get("inningsPitched"))
        games = safe_float(stat.get("gamesPlayed"), 0) or 0
        if not ip or not games:
            return 1.0, "Bullpen/team workload unavailable"
        ip_per_game = ip / max(games, 1)
        factor = clamp(1.0 + ((ip_per_game - 8.7) * 0.015), 0.97, 1.03)
        return float(factor), f"Conservative bullpen workload BF factor x{factor:.3f}"
    except Exception:
        return 1.0, "Bullpen/team workload unavailable"

# =========================
# PROJECTION ENGINE
# =========================
def make_projection(row, bankroll, default_odds, use_statcast, use_pitch_type, use_calibration, use_bayesian_markov=True, use_weather=True, use_umpire=True, use_xgboost_assist=False, use_sgo=False, use_optic=False):
    pid = row["pitcher_id"]
    pitcher_name = row["pitcher"]
    hand = row["hand"]

    profile = get_pitcher_profile(pid)
    recent_rows = get_recent_logs(pid)
    leash = build_leash_model(recent_rows)

    lineup_k, lineup_rows, lineup_msg, lineup_locked = calculate_lineup_k_rate(row["game_pk"], row["opp_side"], hand)
    if lineup_k is None:
        lineup_k, fallback_msg = team_k_vs_hand(row["opp_team_id"], hand)
        lineup_rows = []
        lineup_msg = fallback_msg
        lineup_locked = False

    pitcher_k, pitcher_k_source, learn_scale = blend_pitcher_k_rate(profile["Pitcher K%"], recent_rows, pid)

    statcast_profile = get_statcast_pitch_profile(pid, days=365)
    pitcher_k, statcast_note = apply_statcast_csw_adjustment(pitcher_k, statcast_profile, enabled=use_statcast)

    # v9.6 upgrade: prefer true batter-vs-pitch-type matchup when lineup is available.
    matchup_profile = build_pitch_type_matchup_profile(
        statcast_profile,
        lineup_rows if lineup_locked else [],
        enabled=use_pitch_type,
        min_batters=5,
        pitcher_hand=hand,
    )
    if matchup_profile.get("available"):
        pitcher_k, pitch_type_note = apply_advanced_pitch_type_matchup_adjustment(pitcher_k, matchup_profile, enabled=use_pitch_type)
        pitch_type_available = True
        pitch_type_rows = matchup_profile.get("rows", [])
        pitch_type_factor = safe_float(matchup_profile.get("factor"), 1.0) or 1.0
    else:
        # fallback to pitcher-only whiff mix when batter Statcast profiles are thin or lineup is not posted
        pitcher_k, pitch_type_note, pitch_type_available, pitch_type_rows, pitch_type_factor = apply_pitch_type_matchup_adjustment(pitcher_k, statcast_profile, enabled=use_pitch_type)
        if not pitch_type_available:
            pitch_type_note = matchup_profile.get("message", pitch_type_note)

    calibration_profile = build_model_calibration_profile(load_json(RESULT_LOG, []))
    pitcher_k, calibration_note = apply_calibration_adjustment(pitcher_k, calibration_profile, enabled=use_calibration)

    matchup_k = calculate_log5_k_rate(pitcher_k, lineup_k)
    ump_mult, ump_name, umpire_note = umpire_factor(row["game_pk"], enabled=use_umpire)
    park = park_k_factor(row.get("venue"))
    weather_mult, weather_note, weather_details = weather_k_factor(row.get("venue"), row.get("game_time"), enabled=use_weather)
    env_mult = float(clamp(park * ump_mult * weather_mult, 0.94, 1.06))

    bf = leash["expected_bf"]
    bullpen_factor, bullpen_note = bullpen_workload_bf_factor(row.get("team_id"))
    bf = float(clamp(bf * bullpen_factor, 14, 31))
    batter_rates, simulation_source = build_pa_sequence(lineup_rows if lineup_locked else [], bf, lineup_k)

    # v10.7: safer Bayesian + Markov Monte Carlo built around expected BF, not generic 27 outs.
    preliminary_score = data_lock_score(
        lineup_locked=lineup_locked,
        pitcher_confirmed=row.get("pitcher_confirmed"),
        active_line=None,
        consensus_info={"count": 0, "spread": None},
        ppb=leash["ppb"],
        statcast_available=statcast_profile.get("available"),
        pitch_type_available=pitch_type_available,
    )
    if use_bayesian_markov:
        sims, pa_probs, bayesian_markov_note = simulate_bayesian_markov_matchup(
            matchup_k,
            batter_rates,
            expected_bf=bf,
            park=env_mult,
            ump=1.0,
            data_score=preliminary_score,
            lineup_locked=lineup_locked,
            pitcher_confirmed=row.get("pitcher_confirmed"),
            leash=leash,
            sims=BAYESIAN_MARKOV_SIMS,
        )
        simulation_source = simulation_source + " + Bayesian Markov MC"
    else:
        sims, pa_probs = simulate_matchup(matchup_k, batter_rates, park=env_mult, ump=1.0, sims=12000)
        bayesian_markov_note = "Standard Monte Carlo"

    mean = float(np.mean(sims))

    # v10.7 optional XGBoost residual assist. Capped and OFF by default.
    xgb_current_features = xgb_feature_row_from_picklike({
        "projection": mean,
        "pitcher_k": pitcher_k,
        "opp_k": lineup_k,
        "expected_bf": bf,
        "ppb": leash["ppb"],
        "recent_ip": leash["recent_ip"],
        "data_score": preliminary_score,
        "lineup_locked": lineup_locked,
        "pitcher_confirmed": row.get("pitcher_confirmed"),
        "statcast_available": statcast_profile.get("available"),
        "statcast_csw": None if statcast_profile.get("csw") is None else statcast_profile.get("csw") * 100,
        "statcast_whiff": None if statcast_profile.get("whiff") is None else statcast_profile.get("whiff") * 100,
        "pitch_type_matchup_available": pitch_type_available,
        "pitch_type_factor": pitch_type_factor,
        "consensus_count": 0,
        "consensus_spread": 0,
    })
    adjusted_mean, xgb_info = apply_xgboost_assist(xgb_current_features, mean, enabled=use_xgboost_assist)
    if xgb_info.get("active"):
        delta = adjusted_mean - mean
        sims = np.clip(sims + delta, 0, None)
        mean = float(np.mean(sims))

    median = float(np.median(sims))
    p10 = float(np.percentile(sims, 10))
    p90 = float(np.percentile(sims, 90))

    sportsbook_data = get_sportsbook_k_data(row["home_team"], row["away_team"], pitcher_name)
    pp_data = get_prizepicks_k_data(pitcher_name)
    ud_data = get_underdog_k_data(pitcher_name)
    sgo_data = get_sportsgameodds_k_data(pitcher_name) if use_sgo else source_result("SportsGameOdds", "OFF", message="Optional source turned off")
    optic_data = get_opticodds_k_data(pitcher_name) if use_optic else source_result("OpticOdds", "OFF", message="Optional source turned off")

    active_line, active_source, consensus = choose_active_line(sportsbook_data, pp_data, ud_data, sgo_data, optic_data)

    # NOTE: CLV/line tracking updates on refresh because it tracks market movement.
    # Official pick history is only saved when you press "SAVE OFFICIAL BEFORE-GAME SNAPSHOT".
    line_delta = update_clv_snapshot(pitcher_name, active_source, active_line) if active_line is not None else None
    true_line_delta = track_line_delta(pitcher_name, active_source, active_line) if active_line is not None else None

    # v10.8.2 merged upgrade: tiny market-move projection signal from file #2.
    # This is deliberately applied AFTER line selection so it cannot disturb Underdog matching.
    market_move_factor, market_move_note = market_movement_k_factor(line_delta, active_line=active_line, enabled=True)
    if active_line is not None and market_move_factor != 1.0:
        market_shift = float(clamp((market_move_factor - 1.0) * mean, -MARKET_MOVE_K_SHIFT_CAP, MARKET_MOVE_K_SHIFT_CAP))
        sims = np.clip(sims + market_shift, 0, None)
        mean = float(np.mean(sims))
        median = float(np.median(sims))
        p10 = float(np.percentile(sims, 10))
        p90 = float(np.percentile(sims, 90))
    else:
        market_shift = 0.0

    metrics = calculate_pick_metrics(sims, active_line)

    score = data_lock_score(
        lineup_locked=lineup_locked,
        pitcher_confirmed=row.get("pitcher_confirmed"),
        active_line=active_line,
        consensus_info=consensus,
        ppb=leash["ppb"],
        statcast_available=statcast_profile.get("available"),
        pitch_type_available=pitch_type_available
    )

    over_prob_raw = metrics.get("over_prob")
    over_prob = shrink_probability_to_market(over_prob_raw, score, lineup_locked, row.get("pitcher_confirmed")) if over_prob_raw is not None else None
    under_prob = 1 - over_prob if over_prob is not None else None

    if active_line is None:
        pick_side = "NO LINE"
        fair_prob = None
        price = None
        no_vig = None
        ev = None
        kelly = 0.0
        edge_pct = None
        gap = None
    else:
        pick_side = "OVER" if mean > active_line else "UNDER"
        fair_prob = over_prob if pick_side == "OVER" else under_prob
        price = default_odds
        priced_rows = []
        for src in [sportsbook_data, sgo_data, optic_data]:
            priced_rows.extend(src.get("rows", []))
        matching_priced = []
        for r in priced_rows:
            if safe_float(r.get("Line")) == safe_float(active_line) and pick_side in str(r.get("Side", "")).upper():
                matching_priced.append(r)
        if matching_priced:
            best = sorted(matching_priced, key=lambda x: expected_value(fair_prob, x.get("Price")) or -999)[-1]
            price = safe_float(best.get("Price"), default_odds)
            no_vig = paired_no_vig_probability(priced_rows, best)
        else:
            no_vig = american_to_implied(price)
        ev = expected_value(fair_prob, price)
        raw_kelly = kelly_fraction(fair_prob, price)
        kelly = min(raw_kelly, MAX_RECOMMENDED_KELLY)
        edge_pct = ((fair_prob - no_vig) * 100) if no_vig is not None and fair_prob is not None else None
        gap = abs(mean - active_line)

    risk_label, risk_notes = classify_risk(
        fair_prob,
        score,
        priced=(active_line is not None),
        edge_pct=edge_pct if edge_pct is not None else -999,
        gap=gap if gap is not None else 0,
        line_source=active_source
    )

    signal, signal_type = build_signal(mean, active_line, fair_prob or 0, ev, leash["ppb"], score)

    bettable, no_bet_reasons = no_bet_gate(
        active_line=active_line,
        pick_side=pick_side,
        fair_prob=fair_prob,
        ev=ev,
        gap=gap,
        score=score,
        lineup_locked=lineup_locked,
        pitcher_confirmed=row.get("pitcher_confirmed"),
        line_source=active_source,
        consensus_info=consensus,
        leash=leash,
    )

    if not bettable:
        signal_type = "pass"
        if pick_side in ["OVER", "UNDER"]:
            signal = f"PASS — {pick_side}"
        else:
            signal = "PASS"
        risk_notes = (risk_notes + "; " if risk_notes else "") + "No-bet gate: " + "; ".join(no_bet_reasons)

    prop_rows = []
    for src in [sportsbook_data, pp_data, ud_data, sgo_data, optic_data]:
        for r in src.get("rows", []):
            rr = dict(r)
            rr["Model Projection"] = round(mean, 2)
            line = safe_float(rr.get("Line"))
            if line is not None:
                raw_p = poisson_over_probability(mean, line)
                cal_p = shrink_probability_to_market(raw_p, score, lineup_locked, row.get("pitcher_confirmed"))
                lean = "OVER" if mean > line else "UNDER"
                lean_prob = cal_p if lean == "OVER" else 1 - cal_p
                rr["Model Lean"] = lean
                rr["Raw Model Prob %"] = round((raw_p if lean == "OVER" else 1 - raw_p) * 100, 1)
                rr["Model Prob %"] = round(lean_prob * 100, 1)
                rr["Hit Risk"], rr["Risk Notes"] = classify_risk(
                    lean_prob,
                    score,
                    priced=safe_float(rr.get("Price")) is not None,
                    edge_pct=0,
                    gap=abs(mean - line),
                    line_source=rr.get("Source")
                )
            rr["All Real"] = "YES"
            prop_rows.append(rr)

    pick_id = f"{row['date']}_{row['game_pk']}_{pid}_{active_line}_{active_source}"

    return {
        "pick_id": pick_id,
        "created_at": now_iso(),
        "date": row["date"],
        "game_pk": row["game_pk"],
        "game_time": row["game_time"],
        "status": row["status"],
        "venue": row.get("venue"),
        "pitcher_id": str(pid),
        "pitcher": pitcher_name,
        "hand": hand,
        "team": row["team"],
        "opponent": row["opponent"],
        "matchup": row["matchup"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "pitcher_confirmed": bool(row.get("pitcher_confirmed")),
        "lineup_locked": bool(lineup_locked),
        "lineup_note": lineup_msg,
        "pitcher_k": round(pitcher_k, 3),
        "pitcher_k_source": pitcher_k_source,
        "opp_k": round(lineup_k, 3),
        "simulation_source": simulation_source,
        "bayesian_markov_enabled": bool(use_bayesian_markov),
        "bayesian_markov_note": bayesian_markov_note,
        "xgboost_enabled": bool(use_xgboost_assist),
        "xgboost_active": bool(xgb_info.get("active")),
        "xgboost_samples": int(xgb_info.get("samples", 0)),
        "xgboost_adjustment": safe_float(xgb_info.get("adjustment"), 0.0),
        "xgboost_note": xgb_info.get("message"),
        "umpire": ump_name,
        "ump_factor": round(ump_mult, 3),
        "umpire_note": umpire_note,
        "weather_enabled": bool(use_weather),
        "weather_factor": round(weather_mult, 3),
        "weather_note": weather_note,
        "weather_temp_f": weather_details.get("temp_f") if isinstance(weather_details, dict) else None,
        "weather_wind_mph": weather_details.get("wind_mph") if isinstance(weather_details, dict) else None,
        "weather_humidity": weather_details.get("humidity") if isinstance(weather_details, dict) else None,
        "weather_precip_prob": weather_details.get("precip_prob") if isinstance(weather_details, dict) else None,
        "environment_factor": round(env_mult, 3),
        "expected_bf": round(bf, 1),
        "ppb": round(leash["ppb"], 2),
        "leash_risk": leash.get("leash_risk"),
        "bullpen_bf_factor": round(safe_float(bullpen_factor, 1.0), 3),
        "bullpen_note": bullpen_note,
        "recent_ip": round(leash["recent_ip"], 2),
        "last_10_ks": leash["last_10_ks"],
        "projection": round(mean, 2),
        "median": round(median, 2),
        "p10": round(p10, 2),
        "p90": round(p90, 2),
        "learning_scale": round(learn_scale, 3),
        "line": active_line,
        "line_source": active_source,
        "underdog_status": ud_data.get("status"),
        "underdog_line": ud_data.get("line"),
        "underdog_message": ud_data.get("message"),
        "line_delta": line_delta,
        "true_line_delta": true_line_delta,
        "market_move_factor": round(safe_float(market_move_factor, 1.0), 3),
        "market_move_shift": round(safe_float(market_shift, 0.0), 3),
        "market_move_note": market_move_note,
        "consensus_count": consensus.get("count"),
        "consensus_quality": consensus.get("quality"),
        "consensus_spread": consensus.get("spread"),
        "leash_risk": leash.get("leash_risk"),
        "bettable": bettable,
        "no_bet_reasons": no_bet_reasons,
        "odds": price,
        "pick_side": pick_side,
        "over_probability": None if over_prob is None else round(over_prob, 4),
        "under_probability": None if under_prob is None else round(under_prob, 4),
        "fair_probability": None if fair_prob is None else round(fair_prob, 4),
        "edge_ks": None if active_line is None else round(mean - active_line, 2),
        "abs_edge": None if active_line is None else round(abs(mean - active_line), 2),
        "edge_pct": None if edge_pct is None else round(edge_pct, 2),
        "ev": None if ev is None else round(ev, 4),
        "kelly": round(kelly, 4),
        "bet_size": round(bankroll * kelly, 2),
        "data_score": score,
        "risk_label": risk_label,
        "risk_notes": risk_notes,
        "signal": signal,
        "signal_type": signal_type,
        "graded": False,
        "actual": None,
        "win": None,
        "statcast_available": statcast_profile.get("available"),
        "statcast_rows": statcast_profile.get("rows"),
        "statcast_csw": None if statcast_profile.get("csw") is None else round(statcast_profile.get("csw") * 100, 1),
        "statcast_whiff": None if statcast_profile.get("whiff") is None else round(statcast_profile.get("whiff") * 100, 1),
        "statcast_note": statcast_note,
        "pitch_type_matchup_available": pitch_type_available,
        "pitch_type_factor": round(safe_float(pitch_type_factor, 1.0), 3),
        "pitch_type_note": pitch_type_note,
        "calibration_note": calibration_note,
        "calibration_quality": calibration_profile.get("quality_score"),
        "calibration_samples": calibration_profile.get("samples"),
        "prop_rows": prop_rows,
        "lineup_rows": lineup_rows,
        "pitch_type_rows": pitch_type_rows,
        "source_status": {
            "sportsbook": sportsbook_data.get("status"),
            "prizepicks": pp_data.get("status"),
            "underdog": ud_data.get("status"),
            "sportsgameodds": sgo_data.get("status"),
            "opticodds": optic_data.get("status"),
        }
    }

def save_many_once(new_picks):
    picks = load_json(PICK_LOG, [])
    ids = set([p.get("pick_id") for p in picks])
    added = 0
    for p in new_picks:
        if p.get("pick_id") not in ids:
            official = dict(p)
            official["official_snapshot_saved_at"] = now_iso()
            official["snapshot_type"] = "OFFICIAL_BEFORE_GAME"
            official["official_quality_gate"] = "PASS" if official.get("data_score", 0) >= MIN_OFFICIAL_SAVE_SCORE else "LOW_DATA_REVIEW"
            picks.append(official)
            log_long_backtest_row(official)
            ids.add(p.get("pick_id"))
            added += 1
    save_json(PICK_LOG, picks[-10000:])
    return added

# =========================
# GRADING
# =========================
def is_game_final(game_pk):
    sched = safe_get_json(f"{MLB_BASE}/schedule", params={"sportId": 1, "gamePk": game_pk})
    try:
        games = (sched.get("dates") or [{}])[0].get("games") or []
        return bool(games and games[0].get("status", {}).get("abstractGameState") == "Final")
    except Exception:
        return False

def get_actual_pitcher_ks(game_pk, pitcher_id):
    box = safe_get_json(f"{MLB_BASE}/game/{game_pk}/boxscore")
    if not box:
        return None
    for side in ["home", "away"]:
        players = box.get("teams", {}).get(side, {}).get("players", {})
        for p in players.values():
            person = p.get("person", {})
            if str(person.get("id")) == str(pitcher_id):
                return p.get("stats", {}).get("pitching", {}).get("strikeOuts", None)
    return None

def grade_finished_games():
    picks = load_json(PICK_LOG, [])
    results = load_json(RESULT_LOG, [])
    result_ids = set([r.get("pick_id") for r in results])
    graded = 0
    for p in picks:
        if p.get("graded"):
            continue
        if not p.get("game_pk") or not p.get("pitcher_id"):
            continue
        if not is_game_final(p["game_pk"]):
            continue
        actual = get_actual_pitcher_ks(p["game_pk"], p["pitcher_id"])
        if actual is None:
            continue
        p["actual"] = actual
        p["graded"] = True
        p["graded_at"] = now_iso()
        line = safe_float(p.get("line"))
        side = p.get("pick_side")
        if line is not None and side in ["OVER", "UNDER"]:
            win = (actual > line) if side == "OVER" else (actual < line)
            p["win"] = bool(win)
            p["graded_result"] = "WIN" if win else "LOSS"
        else:
            p["win"] = None
            p["graded_result"] = "NO LINE"
        p["new_learning_scale"] = round(update_learning(p["pitcher_id"], p.get("projection"), actual), 3)
        if p.get("pick_id") not in result_ids:
            results.append(dict(p))
            result_ids.add(p.get("pick_id"))
        graded += 1
    save_json(PICK_LOG, picks[-10000:])
    save_json(RESULT_LOG, results[-10000:])
    return graded

def build_signal_tracking():
    results = load_json(RESULT_LOG, [])
    finished = [r for r in results if r.get("graded_result") in ["WIN", "LOSS"]]
    buckets = {}
    def add_bucket(key, row):
        if key not in buckets:
            buckets[key] = {"tag": key, "count": 0, "wins": 0}
        buckets[key]["count"] += 1
        buckets[key]["wins"] += 1 if row.get("graded_result") == "WIN" else 0
    for r in finished:
        tags = [
            f"side={r.get('pick_side')}",
            f"risk={r.get('risk_label')}",
            f"line_source={r.get('line_source')}",
            f"consensus={r.get('consensus_quality')}",
            f"lineup_locked={r.get('lineup_locked')}",
            f"statcast={r.get('statcast_available')}",
            f"pitch_type={r.get('pitch_type_matchup_available')}",
            f"data_score={int((r.get('data_score') or 0)//10)*10}s",
        ]
        for tag in tags:
            add_bucket(tag, r)
    rows = []
    for v in buckets.values():
        count = v["count"]
        wins = v["wins"]
        rows.append({"Signal Tag": v["tag"], "Samples": count, "Wins": wins, "Win Rate": round(wins / count * 100, 1) if count else 0})
    df = pd.DataFrame(rows).sort_values(["Samples", "Win Rate"], ascending=[False, False]) if rows else pd.DataFrame()
    save_json(SIGNAL_TRACKING_FILE, rows)
    return df

# =========================
# RENDERING
# =========================
def render_kpis(picks, bankroll):
    valid = [p for p in picks if p.get("ev") is not None]
    best = sorted(valid, key=lambda x: x.get("ev", -999), reverse=True)[0] if valid else None
    real_line_count = len([p for p in picks if p.get("line") is not None])
    strong_count = len([p for p in picks if p.get("signal_type") == "good"])
    no_line_count = len([p for p in picks if p.get("line") is None])
    statcast_count = len([p for p in picks if p.get("statcast_available")])
    pitch_type_count = len([p for p in picks if p.get("pitch_type_matchup_available")])
    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi-box"><div class="kpi-label">Board Rows</div><div class="kpi-value">{len(picks)}</div><div class="kpi-sub">Current screen</div></div>
      <div class="kpi-box"><div class="kpi-label">Real Lines</div><div class="kpi-value green">{real_line_count}</div><div class="kpi-sub">No fake prop lines</div></div>
      <div class="kpi-box"><div class="kpi-label">No Line</div><div class="kpi-value orange">{no_line_count}</div><div class="kpi-sub">Projection only</div></div>
      <div class="kpi-box"><div class="kpi-label">Strong Signals</div><div class="kpi-value green">{strong_count}</div><div class="kpi-sub">Strict gates</div></div>
      <div class="kpi-box"><div class="kpi-label">Statcast</div><div class="kpi-value">{statcast_count}/{len(picks)}</div><div class="kpi-sub">Pitch-type {pitch_type_count}</div></div>
      <div class="kpi-box"><div class="kpi-label">Bankroll</div><div class="kpi-value green">${bankroll:,.0f}</div><div class="kpi-sub">{california_now().strftime('%I:%M %p PT')}</div></div>
    </div>
    """, unsafe_allow_html=True)
    if best:
        st.markdown(f"""
        <div class="green-card">
          <div class="small-muted">Best EV Play On Current Board</div>
          <div class="big-number green">{best.get('signal')}</div>
          <div>{best.get('pitcher')} — {best.get('pick_side')} {best.get('line')} Ks | EV {round((best.get('ev') or 0)*100,2)}% | Data {best.get('data_score')}/100</div>
        </div>
        """, unsafe_allow_html=True)

def render_pick_card(p):
    prob = p.get("fair_probability")
    prob_pct = int(round(prob * 100)) if prob is not None else 0
    progress_width = max(3, min(100, prob_pct))
    risk = p.get("risk_label", "")
    signal_type = p.get("signal_type", "pass")
    if "85" in risk or signal_type == "good":
        color_class, progress_class, badge = "green", "progress-green", "good-badge"
    elif "PASS" in risk or "NO" in risk:
        color_class, progress_class, badge = "red", "progress-red", "red-badge"
    else:
        color_class, progress_class, badge = "orange", "progress-orange", "yellow-badge"
    line_display = f"{safe_float(p.get('line')):.1f}" if p.get('line') is not None else "NO REAL LINE"
    edge_display = p.get("edge_ks") if p.get("edge_ks") is not None else "—"
    ev_display = f"{(p.get('ev') or 0)*100:.2f}%" if p.get("ev") is not None else "—"
    prob_display = f"{prob_pct}%" if prob is not None else "—"
    # Render-safe Last 10 K bars.
    # NOTE: this avoids standalone raw HTML ever being printed by Streamlit/tunnel caching.
    # The full card below is still rendered with unsafe_allow_html=True.
    bars = "<span class='small-muted'>No recent K log</span>"
    last_ks = p.get("last_10_ks", []) or []
    if last_ks:
        max_k = max(max([safe_int(x, 0) or 0 for x in last_ks]), 1)
        bar_parts = []
        for k_raw in last_ks[:10]:
            k = safe_int(k_raw, 0) or 0
            h = int(20 + (k / max_k) * 42)
            bar_parts.append(
                f"<span class='mini-k-bar-wrap'>"
                f"<span class='mini-k-bar' style='height:{h}px;'></span>"
                f"<span class='mini-k-label'>{k}</span>"
                f"</span>"
            )
        bars = "<div class='mini-k-bars'>" + "".join(bar_parts) + "</div>"
    statcast_txt = "YES" if p.get("statcast_available") else "NO"
    pitch_type_txt = "YES" if p.get("pitch_type_matchup_available") else "NO"
    st.markdown(f"""
    <div class="pick-card">
      <div style="display:grid;grid-template-columns:1.3fr .8fr .9fr 1fr 1fr;gap:18px;align-items:center;">
        <div>
          <div class="player-name">{p.get('pitcher')}</div>
          <div class="small-muted">{p.get('matchup')} | {p.get('hand')}HP</div>
          <div class="small-muted">{p.get('team')} vs {p.get('opponent')}</div>
          <span class="badge {badge}">{p.get('risk_label')}</span>
          <span class="badge">{p.get('line_source')}</span>
        </div>
        <div><div class="small-muted">Projection</div><div class="big-number {color_class}">{p.get('projection')}</div><div class="small-muted">BF {p.get('expected_bf')} | PPB {p.get('ppb')}</div></div>
        <div><div class="small-muted">Line</div><div class="big-number">{line_display}</div><div class="small-muted">Edge: {edge_display} K</div></div>
        <div>
          <div class="small-muted">Pick</div><div class="big-number {color_class}">{p.get('pick_side')}</div>
          <div class="small-muted">Fair Prob</div><div class="{color_class}" style="font-size:26px;font-weight:900;">{prob_display}</div>
          <div class="progress-wrap"><div class="{progress_class}" style="width:{progress_width}%;"></div></div>
        </div>
        <div>
          <div class="small-muted">Signal</div><div class="{color_class}" style="font-size:20px;font-weight:950;">{p.get('signal')}</div>
          <div class="small-muted" style="margin-top:8px;">EV</div><div style="font-size:22px;font-weight:900;">{ev_display}</div>
          <div class="small-muted">Bet Size</div><div style="font-size:22px;font-weight:900;">${p.get('bet_size')}</div>
        </div>
      </div>
      <div class="hr-soft"></div>
      <div style="display:grid;grid-template-columns:.7fr .7fr .7fr .7fr .7fr .7fr 2.2fr;gap:14px;align-items:end;">
        <div><div class="small-muted">Data Score</div><div style="font-size:22px;font-weight:900;">{p.get('data_score')}/100</div></div>
        <div><div class="small-muted">Pitcher K%</div><div style="font-size:22px;font-weight:900;">{p.get('pitcher_k')}</div></div>
        <div><div class="small-muted">Opp K%</div><div style="font-size:22px;font-weight:900;">{p.get('opp_k')}</div></div>
        <div><div class="small-muted">Statcast</div><div style="font-size:22px;font-weight:900;">{statcast_txt}</div></div>
        <div><div class="small-muted">Pitch-Type</div><div style="font-size:22px;font-weight:900;">{pitch_type_txt}</div></div>
        <div><div class="small-muted">CLV Δ</div><div style="font-size:22px;font-weight:900;">{p.get('line_delta')}</div></div>
        <div><div class="small-muted">Last 10 Ks</div>{bars}</div>
      </div>
      <div class="small-muted" style="margin-top:12px;">Risk Notes: {p.get('risk_notes')}</div>
      <div class="small-muted">Statcast: {p.get('statcast_note')} | Pitch Type: {p.get('pitch_type_note')} | Calibration: {p.get('calibration_note')}</div>
      <div class="small-muted">Weather: {p.get('weather_note')} | Umpire: {p.get('umpire_note')}</div>
      <div class="small-muted">Advanced Sim: {p.get('bayesian_markov_note')} | XGBoost: {p.get('xgboost_note')}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# APP
# =========================
st.markdown("""
<div class="hero-panel">
  <div class="big-title">🔥 MLB STRIKEOUT PROP ENGINE v10.8 WEATHER + UMPIRE CAPS</div>
  <div class="sub-title">Strict Win Filter + MLB-only Underdog line lock → Refresh → Save → Grade</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")
    day_mode = st.radio("Game Feed", ["Today + Tomorrow", "Today", "Tomorrow"], index=0)
    bankroll = st.number_input("Bankroll", min_value=1.0, value=1000.0, step=50.0)
    default_odds = st.number_input("Default Odds if sportsbook price missing", value=-110.0, step=5.0)
    hide_no_line = st.checkbox("Hide No Real Line picks", value=False)
    only_strong = st.checkbox("Show only strong signals", value=True)
    st.divider()
    st.header("Model Upgrades")
    use_statcast = st.checkbox("Use Statcast pitcher CSW/whiff", value=True)
    use_pitch_type = st.checkbox("Use pitch-type whiff mix", value=True)
    use_calibration = st.checkbox("Use historical calibration", value=True)
    use_bayesian_markov = st.checkbox("Use Bayesian Markov Monte Carlo", value=True)
    use_weather = st.checkbox("Use live weather adjustment", value=True)
    use_umpire = st.checkbox("Use capped umpire tendency", value=True)
    use_xgboost_assist = st.checkbox("Experimental: capped XGBoost assist", value=False)
    use_sgo = st.checkbox("Optional: SportsGameOdds API", value=False)
    use_optic = st.checkbox("Optional: OpticOdds API", value=False)
    if st.button("🧹 Clear Streamlit Cache + Reload Live Lines", use_container_width=True):
        st.cache_data.clear()
        st.session_state.loaded_picks = []
        st.session_state.last_refresh_time = None
        st.success("Cache cleared. Now click REFRESH LIVE BOARD again.")
    st.caption("Refresh does not save official picks. Save only when the board looks right. Optional paid APIs stay OFF unless you have keys.")

dates = target_dates(day_mode)

if "loaded_picks" not in st.session_state:
    st.session_state.loaded_picks = []
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = None
if "last_saved_count" not in st.session_state:
    st.session_state.last_saved_count = 0

col_refresh, col_save = st.columns(2)

with col_refresh:
    refresh_btn = st.button("🔄 REFRESH LIVE BOARD — Do Not Save Yet", use_container_width=True)

with col_save:
    save_btn = st.button("💾 SAVE OFFICIAL BEFORE-GAME SNAPSHOT", use_container_width=True)

if refresh_btn:
    all_rows = []
    for d in dates:
        all_rows.extend(extract_probable_pitchers(d))

    projections = []
    progress = st.progress(0)

    for i, row in enumerate(all_rows):
        try:
            projections.append(
                make_projection(
                    row,
                    bankroll=bankroll,
                    default_odds=default_odds,
                    use_statcast=use_statcast,
                    use_pitch_type=use_pitch_type,
                    use_calibration=use_calibration,
                    use_bayesian_markov=use_bayesian_markov,
                    use_weather=use_weather,
                    use_umpire=use_umpire,
                    use_xgboost_assist=use_xgboost_assist,
                    use_sgo=use_sgo,
                    use_optic=use_optic
                )
            )
        except Exception as e:
            log_source_request("make_projection", "ERROR", f"{row.get('pitcher')}: {e}")
        progress.progress((i + 1) / max(1, len(all_rows)))

    st.session_state.loaded_picks = projections
    st.session_state.last_refresh_time = now_iso()
    st.success(f"Refreshed {len(projections)} pitchers. Nothing officially saved yet.")

if save_btn:
    if not st.session_state.get("loaded_picks"):
        st.warning("Refresh the live board first, inspect the lines, then save the official before-game snapshot.")
    else:
        added = save_many_once(st.session_state.loaded_picks)
        st.session_state.last_saved_count = added
        st.success(f"Saved official before-game snapshot. Added {added} new rows.")

saved = load_json(PICK_LOG, [])

# IMPORTANT:
# - If you have refreshed this session, the screen shows refreshed live board.
# - If not, it shows saved official snapshots for the selected dates.
if st.session_state.get("loaded_picks"):
    board = st.session_state.loaded_picks
    board_status = "LIVE REFRESHED BOARD — NOT OFFICIAL UNLESS SAVED"
else:
    board = [p for p in saved if p.get("date") in dates]
    board_status = "SAVED OFFICIAL SNAPSHOTS"

if hide_no_line:
    board = [p for p in board if p.get("line") is not None]
if only_strong:
    board = [p for p in board if p.get("signal_type") == "good"]

st.info(f"{APP_VERSION} | {board_status} | Last refresh: {st.session_state.get('last_refresh_time') or 'Not refreshed this session'} | Last save added: {st.session_state.get('last_saved_count', 0)}")

render_kpis(board, bankroll)

def display_clean_real_prop_rows(rows, **kwargs):
    cleaned = clean_real_prop_debug_rows(rows)
    if cleaned:
        st.dataframe(pd.DataFrame(cleaned), use_container_width=True, hide_index=True)
    else:
        st.info("No rejected/NBA debug rows shown. Only valid MLB pitcher strikeout lines will appear here.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "TOP PLAYS",
    "ALL PLAYERS",
    "REAL PROP BOARD",
    "STATCAST",
    "AFTER GAMES / LEARNING",
    "SETTINGS"
])

with tab1:
    st.markdown('<div class="section-title-pro">Top Plays</div>', unsafe_allow_html=True)
    if not board:
        st.info("Click 🔄 Refresh Live Board first.")
    else:
        top = sorted(
            board,
            key=lambda x: (
                x.get("signal_type") == "good",
                x.get("ev") if x.get("ev") is not None else -999,
                x.get("fair_probability") if x.get("fair_probability") is not None else 0
            ),
            reverse=True
        )
        for p in top:
            render_pick_card(p)

with tab2:
    st.markdown('<div class="section-title-pro">All Players</div>', unsafe_allow_html=True)
    if board:
        show = pd.DataFrame([{k: v for k, v in p.items() if k not in ["prop_rows", "lineup_rows", "pitch_type_rows"]} for p in board])
        cols = [
            "date", "pitcher", "matchup", "hand", "projection", "line", "pick_side",
            "fair_probability", "edge_ks", "ev", "signal", "risk_label",
            "line_source", "underdog_line", "underdog_status", "underdog_message", "data_score", "lineup_locked", "pitcher_confirmed",
            "statcast_available", "pitch_type_matchup_available", "pitch_type_factor", "bayesian_markov_enabled", "xgboost_active", "xgboost_samples", "xgboost_adjustment", "bettable", "leash_risk"
        ]
        cols = [c for c in cols if c in show.columns]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No players loaded.")

with tab3:
    st.markdown('<div class="section-title-pro">Real Prop Rows + Underdog Debug</div>', unsafe_allow_html=True)
    rows = []
    for p in board:
        for r in p.get("prop_rows", []):
            rr = dict(r)
            rr["Pitcher"] = p.get("pitcher")
            rr["Projection"] = p.get("projection")
            rr["Data Score"] = p.get("data_score")
            rows.append(rr)
    rows = clean_real_prop_debug_rows(rows)
    if rows:
        df_rows = pd.DataFrame(rows)
        preferred = [c for c in ["Pitcher", "Source", "Parser Mode", "Matched Name", "Line", "Market", "Line Evidence", "Underdog Path", "Match Score", "Reject Reason", "Projection", "Model Lean", "Model Prob %"] if c in df_rows.columns]
        other = [c for c in df_rows.columns if c not in preferred]
        st.dataframe(df_rows[preferred + other], use_container_width=True, hide_index=True)
    else:
        st.warning("No valid MLB pitcher strikeout prop rows found. Rejected NBA/basketball rows are hidden.")

with tab4:
    st.markdown('<div class="section-title-pro">Statcast + Pitch-Type</div>', unsafe_allow_html=True)
    if board:
        stat_rows = []
        pitch_rows = []
        lineup_rows = []
        for p in board:
            stat_rows.append({
                "Pitcher": p.get("pitcher"),
                "Statcast Available": p.get("statcast_available"),
                "Statcast Rows": p.get("statcast_rows"),
                "CSW%": p.get("statcast_csw"),
                "Whiff%": p.get("statcast_whiff"),
                "Pitch-Type Available": p.get("pitch_type_matchup_available"),
                "Pitch-Type Factor": p.get("pitch_type_factor"),
                "Pitch-Type Note": p.get("pitch_type_note"),
                "Weather Factor": p.get("weather_factor"),
                "Weather Note": p.get("weather_note"),
                "Umpire": p.get("umpire"),
                "Umpire Factor": p.get("ump_factor"),
                "Umpire Note": p.get("umpire_note"),
                "Environment Factor": p.get("environment_factor"),
            })
            for r in p.get("pitch_type_rows", []):
                rr = dict(r)
                rr["Pitcher"] = p.get("pitcher")
                pitch_rows.append(rr)
            for r in p.get("lineup_rows", []):
                rr = dict(r)
                rr["Pitcher"] = p.get("pitcher")
                rr["Matchup"] = p.get("matchup")
                lineup_rows.append(rr)
        st.subheader("Pitcher Statcast Summary")
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
        st.subheader("Pitch-Type Rows")
        if pitch_rows:
            st.dataframe(pd.DataFrame(pitch_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No pitch-type rows loaded yet.")
        st.subheader("Lineup Batter K Inputs")
        if lineup_rows:
            st.dataframe(pd.DataFrame(lineup_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No posted lineup rows loaded yet.")
    else:
        st.info("Load the board first.")

with tab5:
    st.markdown('<div class="section-title-pro">After Games — Grade + Learn</div>', unsafe_allow_html=True)
    if st.button("✅ AFTER GAMES — Grade Results + Update Learning", use_container_width=True):
        graded = grade_finished_games()
        st.success(f"Graded {graded} finished official snapshots and updated learning.")
    results = load_json(RESULT_LOG, [])
    if results:
        df = pd.DataFrame(results)
        finished = df[df["graded_result"].isin(["WIN", "LOSS"])] if "graded_result" in df.columns else pd.DataFrame()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Graded", len(finished))
        if not finished.empty:
            c2.metric("Win Rate", f"{(finished['graded_result'].eq('WIN').mean()*100):.1f}%")
            c3.metric("Avg EV", f"{(finished['ev'].dropna().mean()*100 if 'ev' in finished.columns and not finished['ev'].dropna().empty else 0):.2f}%")
            c4.metric("Avg Edge", f"{(finished['abs_edge'].dropna().mean() if 'abs_edge' in finished.columns and not finished['abs_edge'].dropna().empty else 0):.2f}")
            cal = build_model_calibration_profile(results)
            c5.metric("Calibration", f"{cal.get('quality_score', 0)}/100")
        else:
            c2.metric("Win Rate", "N/A")
            c3.metric("Avg EV", "N/A")
            c4.metric("Avg Edge", "N/A")
            c5.metric("Calibration", "N/A")
        st.dataframe(df.tail(200), use_container_width=True)
        st.markdown('<div class="section-title-pro">Signal Tracking</div>', unsafe_allow_html=True)
        sig = build_signal_tracking()
        if not sig.empty:
            st.dataframe(sig, use_container_width=True, hide_index=True)
        else:
            st.info("Signal tracking starts after graded wins/losses.")
    else:
        st.info("No graded history yet. Save official snapshots before games, then grade after games finish.")

with tab6:
    st.markdown('<div class="section-title-pro">Settings / Saved Files</div>', unsafe_allow_html=True)
    st.code(STORAGE_DIR)
    st.write("Pick Log:")
    st.code(PICK_LOG)
    st.write("Result Log:")
    st.code(RESULT_LOG)
    st.write("Learning File:")
    st.code(LEARN_FILE)
    st.write("CLV File:")
    st.code(CLV_FILE)
    st.write("Long Backtest File:")
    st.code(LONG_BACKTEST_FILE)
    st.subheader("Advanced Model Status")
    xgb_train_df = build_xgb_training_frame()
    st.write(f"XGBoost training samples available: {len(xgb_train_df)} / {XGB_MIN_GRADED_SAMPLES} needed")
    st.caption("XGBoost is a capped residual assist only. It never overrides Underdog lines or no-bet gates.")
    st.subheader("Source Status")
    if board:
        src_rows = []
        for p in board:
            rr = {"Pitcher": p.get("pitcher")}
            rr.update(p.get("source_status", {}))
            src_rows.append(rr)
        st.dataframe(pd.DataFrame(src_rows), use_container_width=True, hide_index=True)
    req = load_json(REQUEST_LOG_FILE, [])
    if req:
        st.subheader("Recent Source Requests / Errors")
        st.dataframe(pd.DataFrame(req).tail(75), use_container_width=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Clear Current Date-Range Official Snapshots"):
            picks = load_json(PICK_LOG, [])
            picks = [p for p in picks if p.get("date") not in dates]
            save_json(PICK_LOG, picks)
            st.warning("Cleared current date-range official snapshots.")
    with col_b:
        if st.button("Clear Request Logs"):
            save_json(REQUEST_LOG_FILE, [])
            st.warning("Request logs cleared.")
    with col_c:
        if st.button("Clear ALL Logs"):
            save_json(PICK_LOG, [])
            save_json(RESULT_LOG, [])
            save_json(LEARN_FILE, {})
            save_json(CLV_FILE, {})
            save_json(SIGNAL_TRACKING_FILE, [])
            save_json(LONG_BACKTEST_FILE, [])
            save_json(LINE_HISTORY_FILE, {})
            save_json(LINEUP_CACHE_FILE, {})
            st.error("All logs cleared.")

st.caption("Workflow: Refresh live board → inspect lines → save official before-game snapshot → after games, grade and learn.")


# =========================
# v10.8.1 RAW PROP DEBUG UI
# =========================
try:
    with st.expander("🔎 Raw Prop Debug Table (Read-Only)", expanded=False):
        st.caption("Read-only view of parsed prop rows. This does not change lines, projections, probabilities, EV, or picks.")
        render_raw_prop_debug_table()
except Exception as e:
    st.warning(f"Raw prop debug table unavailable: {e}")



# =========================
# v10.8.1 HIT-RATE DASHBOARD UI
# =========================
try:
    with st.expander("📊 Hit-Rate Dashboard (Read-Only)", expanded=False):
        st.caption("Read-only summary of already-graded results. This does not change lines, projections, EV, or picks.")
        render_hit_rate_dashboard()
except Exception as e:
    st.warning(f"Hit-rate dashboard unavailable: {e}")





# =========================
# v10.8.3 FULL PRO UI MAIN SECTION
# =========================
try:
    st.markdown('<div class="section-title-pro">Full Pro Dashboard</div>', unsafe_allow_html=True)
    pro_rows = get_current_board_rows_for_ui()
    bankroll_value = globals().get("bankroll", globals().get("BANKROLL", 1000))
    render_full_pro_header(pro_rows, bankroll_value)
    render_full_pro_board(pro_rows, max_cards=10)
except Exception as e:
    st.warning(f"Full Pro Dashboard unavailable: {e}")
