import os
import json
import re
from typing import TypedDict, Dict, Any, Callable

import numpy as np
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


# -------------------------
# Environment and LLM setup
# -------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not found in environment. Create a .env file with GOOGLE_API_KEY=YOUR_KEY."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.0,
)


# -------------------------
# Knowledge base loader
# -------------------------
KB_FILE = os.path.join(os.path.dirname(__file__), "KnowledgeBase.json")
if not os.path.exists(KB_FILE):
    raise FileNotFoundError(
        f"Knowledge base file not found at {KB_FILE}. Ensure 'KnowledgeBase.json' is present."
    )

with open(KB_FILE, "r", encoding="utf-8") as f:
    CROP_KNOWLEDGE_BASE: Dict[str, Any] = json.load(f)


# -------------------------
# Agent state definition
# -------------------------
class AgentState(TypedDict):
    user_query: str
    crop_name: str
    user_inputs: Dict[str, Any]
    optimized_values: Dict[str, Any]
    remedies_text: str


# -------------------------
# Helper functions
# -------------------------
def parse_range(range_str: str) -> tuple[float, float]:
    """Parse a numeric range like '6.0-8.5' or '50-70%' or '100-150 kg/ha' to (low, high)."""
    if not range_str or not isinstance(range_str, str):
        return (np.nan, np.nan)
    # Extract the first pair of numbers "a-b"
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)", range_str)
    if not match:
        # Single number fallback
        single = re.findall(r"-?\d+(?:\.\d+)?", range_str)
        if len(single) == 1:
            value = float(single[0])
            return (value, value)
        return (np.nan, np.nan)
    low = float(match.group(1))
    high = float(match.group(2))
    if low > high:
        low, high = high, low
    return (low, high)


def get_midpoint(low: float, high: float) -> float:
    return float((low + high) / 2.0)


def clamp(value: float, low: float, high: float) -> float:
    if np.isnan(low) or np.isnan(high):
        return value
    return float(max(low, min(high, value)))


def to_float_safe(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def extract_soil_texture_factor(user_text: str) -> int:
    """Heuristic extraction of soil texture for BARLEY lime requirement.
    Factor: sandy=2, sandy loam=3, loam/clay=4 (default=4).
    """
    text = (user_text or "").lower()
    if re.search(r"\bsandy\b", text) and re.search(r"\bloam\b", text):
        return 3  # sandy loam
    if re.search(r"\bsandy\b", text):
        return 2
    if re.search(r"\bclay\b|\bloam\b", text):
        return 4
    return 4


def get_optimal_range(kb: Dict[str, Any], crop_key: str, metric_key: str) -> tuple[float, float]:
    crop = kb.get(crop_key, {})
    metrics = crop.get("optimal_metrics", {})
    range_str = metrics.get(metric_key) or metrics.get(metric_key.lower()) or ""
    return parse_range(range_str)


def get_min_of_range(kb: Dict[str, Any], crop_key: str, metric_key: str) -> float:
    low, high = get_optimal_range(kb, crop_key, metric_key)
    return low


def optimize_linear_toward_mid(user_value: float, low: float, high: float, factor: float) -> float:
    if np.isnan(user_value) or np.isnan(low) or np.isnan(high):
        return user_value
    midpoint = get_midpoint(low, high)
    return float(user_value + (midpoint - user_value) * factor)


# Basic multilingual/regional synonym map to improve robustness when LLM misses
SYNONYMS_MAP: Dict[str, str] = {
    # Rice
    "nel": "rice", "chawal": "rice", "dhaan": "rice", "paddy": "rice", "oryza sativa": "rice",
    # Wheat
    "gehu": "wheat", "triticum aestivum": "wheat",
    # Corn
    "makka": "corn", "maize": "corn", "zea mays": "corn",
    # Potato
    "aloo": "potato", "alu": "potato", "solanum tuberosum": "potato",
    # Soybean
    "soya": "soybean", "soyabean": "soybean", "soyabin": "soybean", "glycine max": "soybean", "arhar": "soybean",
    # Sugarcane / Cotton / Barley / Sunflower / Tomato (common Hindi)
    "ganna": "sugarcane", "saccharum officinarum": "sugarcane",
    "kapas": "cotton", "gossypium hirsutum": "cotton",
    "jau": "barley", "hordeum vulgare": "barley",
    "surajmukhi": "sunflower", "helianthus annuus": "sunflower",
    "tamatar": "tomato", "solanum lycopersicum": "tomato",
}


def canonicalize_crop(user_text: str, crop: str) -> str:
    text = (user_text or "").lower()
    c = (crop or "").lower().strip()
    if c in CROP_TO_TOOL:
        return c
    # direct synonym
    if c in SYNONYMS_MAP:
        return SYNONYMS_MAP[c]
    # scan text for any synonym token
    for syn, target in SYNONYMS_MAP.items():
        if re.search(rf"\b{re.escape(syn)}\b", text):
            return target
    return c


def build_optimized_payload(crop_key: str, optimized: Dict[str, float]) -> Dict[str, Any]:
    crop_info = CROP_KNOWLEDGE_BASE.get(crop_key, {})
    tips = crop_info.get("fertilizer_tips", "")
    pests = crop_info.get("pest_control", "")
    plan = crop_info.get("plan", "")
    schemes = crop_info.get("gov_schemes", "")
    equipment = crop_info.get("equipment", "")

    remedies_parts = []
    if tips:
        remedies_parts.append(f"Fertilizer tips: {tips}")
    if pests:
        remedies_parts.append(f"Pest/disease control: {pests}")
    if plan:
        remedies_parts.append(f"Cultivation plan: {plan}")
    if schemes:
        remedies_parts.append(f"Government schemes: {schemes}")
    if equipment:
        remedies_parts.append(f"Recommended equipment: {equipment}")

    remedies_text = "\n".join(remedies_parts).strip()

    return {
        "optimized_values": optimized,
        "remedies_text": remedies_text,
    }


# -------------------------
# Optimization tools (10 crops)
# -------------------------
def _get_user_inputs(state: AgentState) -> Dict[str, float]:
    inputs = state.get("user_inputs", {}) or {}
    return {
        "soil_ph": to_float_safe(inputs.get("soil_ph")),
        "soil_moisture": to_float_safe(inputs.get("soil_moisture")),
        "n": to_float_safe(inputs.get("n")),
        "p": to_float_safe(inputs.get("p")),
        "k": to_float_safe(inputs.get("k")),
    }


def _apply_zero_rule(value: float, kb_key: str, metric: str) -> float | None:
    # If user value is exactly 0 for N, P, K, return min of optimal range as per rule
    if metric in ("n", "p", "k") and (not np.isnan(value)) and value == 0:
        return get_min_of_range(CROP_KNOWLEDGE_BASE, kb_key, metric)
    return None


def optimize_barley(state: AgentState) -> Dict[str, Any]:
    crop_key = "barley"
    ui = _get_user_inputs(state)
    # Ranges
    ph_low, ph_high = get_optimal_range(CROP_KNOWLEDGE_BASE, crop_key, "soil_ph")
    mo_low, mo_high = get_optimal_range(CROP_KNOWLEDGE_BASE, crop_key, "soil_moisture")
    n_low, n_high = get_optimal_range(CROP_KNOWLEDGE_BASE, crop_key, "n")
    p_low, p_high = get_optimal_range(CROP_KNOWLEDGE_BASE, crop_key, "p")
    k_low, k_high = get_optimal_range(CROP_KNOWLEDGE_BASE, crop_key, "k")

    # Optimizations
    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.2)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    # Clamp to ranges
    opt_ph = clamp(opt_ph, ph_low, ph_high)
    opt_mo = clamp(opt_mo, mo_low, mo_high)
    opt_n = clamp(opt_n, n_low, n_high)
    opt_p = clamp(opt_p, p_low, p_high)
    opt_k = clamp(opt_k, k_low, k_high)

    # Lime calculation
    soil_factor = extract_soil_texture_factor(state.get("user_query", ""))
    lime_required = (opt_ph - ui["soil_ph"]) * soil_factor if not np.isnan(ui["soil_ph"]) else np.nan

    optimized = {
        "soil_ph": round(opt_ph, 2) if not np.isnan(opt_ph) else opt_ph,
        "soil_moisture": round(opt_mo, 2) if not np.isnan(opt_mo) else opt_mo,
        "n": round(opt_n, 2) if not np.isnan(opt_n) else opt_n,
        "p": round(opt_p, 2) if not np.isnan(opt_p) else opt_p,
        "k": round(opt_k, 2) if not np.isnan(opt_k) else opt_k,
        "lime_required_t_ha": round(lime_required, 2) if not np.isnan(lime_required) else lime_required,
    }

    return build_optimized_payload(crop_key, optimized)


def optimize_cotton(state: AgentState) -> Dict[str, Any]:
    crop_key = "cotton"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (5.8, 8.0)
    mo_low, mo_high = (60.0, 75.0)
    n_low, n_high = (100.0, 150.0)
    p_low, p_high = (50.0, 70.0)
    k_low, k_high = (80.0, 100.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.2)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(clamp(opt_mo, mo_low, mo_high), 2),
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


def optimize_corn(state: AgentState) -> Dict[str, Any]:
    crop_key = "corn"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (5.8, 7.0)
    mo_low, mo_high = (60.0, 70.0)
    n_low, n_high = (125.0, 150.0)
    p_low, p_high = (50.0, 60.0)
    k_low, k_high = (30.0, 40.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.15)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(clamp(opt_mo, mo_low, mo_high), 2),
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


def optimize_potato(state: AgentState) -> Dict[str, Any]:
    crop_key = "potato"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (5.5, 6.0)
    mo_low, mo_high = (65.0, 80.0)
    n_low, n_high = (100.0, 150.0)
    p_low, p_high = (150.0, 200.0)
    k_low, k_high = (150.0, 250.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.15)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(clamp(opt_mo, mo_low, mo_high), 2),
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


def optimize_rice(state: AgentState) -> Dict[str, Any]:
    crop_key = "rice"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (5.5, 6.5)
    # Moisture optimized toward 100 (flooded saturation)
    target_moisture = 100.0
    n_low, n_high = (100.0, 120.0)
    p_low, p_high = (40.0, 50.0)
    k_low, k_high = (40.0, 50.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.15)
    if np.isnan(ui["soil_moisture"]):
        opt_mo = np.nan
    else:
        opt_mo = ui["soil_moisture"] + (target_moisture - ui["soil_moisture"]) * 0.15
        opt_mo = clamp(opt_mo, 0.0, 100.0)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(opt_mo, 2) if not np.isnan(opt_mo) else opt_mo,
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


def optimize_soybean(state: AgentState) -> Dict[str, Any]:
    crop_key = "soybean"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (6.0, 7.0)
    mo_low, mo_high = (50.0, 70.0)
    n_low, n_high = (15.0, 25.0)
    p_low, p_high = (40.0, 60.0)
    k_low, k_high = (80.0, 100.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.15)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(clamp(opt_mo, mo_low, mo_high), 2),
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


def optimize_sugarcane(state: AgentState) -> Dict[str, Any]:
    crop_key = "sugarcane"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (6.0, 6.5)
    mo_low, mo_high = (70.0, 80.0)
    n_low, n_high = (150.0, 250.0)
    p_low, p_high = (60.0, 80.0)
    k_low, k_high = (100.0, 150.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.15)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(clamp(opt_mo, mo_low, mo_high), 2),
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


def optimize_sunflower(state: AgentState) -> Dict[str, Any]:
    crop_key = "sunflower"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (6.0, 6.8)
    mo_low, mo_high = (60.0, 70.0)
    n_low, n_high = (60.0, 90.0)
    p_low, p_high = (40.0, 60.0)
    k_low, k_high = (40.0, 70.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.15)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(clamp(opt_mo, mo_low, mo_high), 2),
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


def optimize_tomato(state: AgentState) -> Dict[str, Any]:
    crop_key = "tomato"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (6.0, 6.8)
    mo_low, mo_high = (65.0, 75.0)
    n_low, n_high = (150.0, 200.0)
    p_low, p_high = (80.0, 100.0)
    k_low, k_high = (150.0, 200.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.15)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(clamp(opt_mo, mo_low, mo_high), 2),
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


def optimize_wheat(state: AgentState) -> Dict[str, Any]:
    crop_key = "wheat"
    ui = _get_user_inputs(state)
    ph_low, ph_high = (6.0, 7.0)
    mo_low, mo_high = (60.0, 70.0)
    n_low, n_high = (100.0, 150.0)
    p_low, p_high = (40.0, 60.0)
    k_low, k_high = (30.0, 50.0)

    opt_ph = optimize_linear_toward_mid(ui["soil_ph"], ph_low, ph_high, 0.15)
    opt_mo = optimize_linear_toward_mid(ui["soil_moisture"], mo_low, mo_high, 0.15)

    n_zero_rule = _apply_zero_rule(ui["n"], crop_key, "n")
    opt_n = n_zero_rule if n_zero_rule is not None else optimize_linear_toward_mid(ui["n"], n_low, n_high, 0.1)

    p_zero_rule = _apply_zero_rule(ui["p"], crop_key, "p")
    opt_p = p_zero_rule if p_zero_rule is not None else optimize_linear_toward_mid(ui["p"], p_low, p_high, 0.1)

    k_zero_rule = _apply_zero_rule(ui["k"], crop_key, "k")
    opt_k = k_zero_rule if k_zero_rule is not None else optimize_linear_toward_mid(ui["k"], k_low, k_high, 0.1)

    optimized = {
        "soil_ph": round(clamp(opt_ph, ph_low, ph_high), 2),
        "soil_moisture": round(clamp(opt_mo, mo_low, mo_high), 2),
        "n": round(clamp(opt_n, n_low, n_high), 2),
        "p": round(clamp(opt_p, p_low, p_high), 2),
        "k": round(clamp(opt_k, k_low, k_high), 2),
    }
    return build_optimized_payload(crop_key, optimized)


# Crop name mapping for router
CROP_TO_TOOL: Dict[str, str] = {
    "barley": "optimize_barley",
    "cotton": "optimize_cotton",
    "corn": "optimize_corn",
    "maize": "optimize_corn",
    "potato": "optimize_potato",
    "rice": "optimize_rice",
    "paddy": "optimize_rice",
    "soy": "optimize_soybean",
    "soybean": "optimize_soybean",
    "soyabean": "optimize_soybean",
    "sugarcane": "optimize_sugarcane",
    "sunflower": "optimize_sunflower",
    "tomato": "optimize_tomato",
    "wheat": "optimize_wheat",
}


# -------------------------
# Nodes
def parse_input(state: AgentState) -> AgentState:
    # Get user query from the agent state
    user_query = state.get("user_query", "")

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You extract crop() and numeric inputs from a user's query. Strictly output a JSON object only, with the exact schema:
{"crop": <string>, "inputs": {"soil_ph": <number or null>, "soil_moisture": <number or null>, "n": <number or null>, "p": <number or null>, "k": <number or null>}, "tool": <string>}

Rules:
- Crop must be one of: barley, cotton, corn, potato, rice, soybean, sugarcane, sunflower, tomato, wheat.
- Normalize synonyms, regional names of crops, any language name of crop, and scientific names of crops to exactly one of the above.
- If uncertain or no valid mapping, again Normalize synonyms, regional names, and scientific names to exactly one of the barley, cotton, corn, potato, rice, soybean, sugarcane, sunflower, tomato, wheat, if this doesnt work see synonyms map below and try again.
- Extract numeric values for soil_ph, soil_moisture (percent without %), N, P, K in kg/ha. If missing, use null.
- Output ONLY the JSON object, nothing else.
- IMPORTANT: For rice, make sure to include 'paddy' as a synonym.

SYNONYMS = {
    # Rice
    "chawal": "rice",
    "dhaan": "rice",
    "paddy": "rice",
    "oryza sativa": "rice",
    "arisi": "rice",         # Tamil
    "nel": "rice",           # Tamil
    "chaula": "rice",        # Odia
    "dhan": "rice",          # Odia

    # Wheat
    "gehu": "wheat",
    "triticum aestivum": "wheat",
    "gothumai": "wheat",     # Tamil
    "gahu": "wheat",         # Odia

    # Corn / Maize
    "makka": "corn",
    "maize": "corn",
    "zea mays": "corn",
    "makkacholam": "corn",   # Tamil
    "maka": "corn",          # Odia

    # Potato
    "aloo": "potato",
    "solanum tuberosum": "potato",
    "urulaikizhangu": "potato",  # Tamil
    "alu": "potato",              # Odia

    # Soybean
    "soya": "soybean",
    "arhar": "soybean",
    "glycine max": "soybean",
    "soyabean": "soybean",        # Tamil
    "soyabin": "soybean",         # Odia

    # Sugarcane
    "ganna": "sugarcane",
    "saccharum officinarum": "sugarcane",
    "karumbu": "sugarcane",       # Tamil
    "akhu": "sugarcane",          # Odia

    # Cotton
    "kapas": "cotton",
    "gossypium hirsutum": "cotton",
    "paruthi": "cotton",          # Tamil
    "kapas": "cotton",            # Odia

    # Barley
    "jau": "barley",
    "hordeum vulgare": "barley",
    "varagu": "barley",           # Tamil
    "jau": "barley",              # Odia

    # Sunflower
    "surajmukhi": "sunflower",
    "helianthus annuus": "sunflower",
    "sooriyakanthi": "sunflower", # Tamil
    "suryamukhi": "sunflower",    # Odia

    # Tomato
    "tamatar": "tomato",
    "solanum lycopersicum": "tomato",
    "thakkali": "tomato",         # Tamil
    "tamato": "tomato",           # Odia
}


Tool mapping:
rice -> optimize_rice
corn -> optimize_corn
wheat -> optimize_wheat
potato -> optimize_potato
soybean -> optimize_soybean
sugarcane -> optimize_sugarcane
cotton -> optimize_cotton
barley -> optimize_barley
sunflower -> optimize_sunflower
tomato -> optimize_tomato
            """.strip()
        ),
        ("human", "User query: {q}\nReturn only the JSON object.")
    ])

    chain = prompt | llm
    # Safe LLM call
    try:
        response = chain.invoke({"q": user_query})
        content = response.content if hasattr(response, "content") else str(response)
    except Exception:
        content = ""

    # Extract JSON safely
    json_match = re.search(r"\{[\s\S]*\}$", content.strip())
    data = {"crop": "", "inputs": {"soil_ph": None, "soil_moisture": None, "n": None, "p": None, "k": None}}
    if json_match:
        try:
            data = json.loads(json_match.group(0))
        except Exception:
            pass

    # Fallback: regex parsing if crop empty
    crop_name = (data.get("crop") or "").strip().lower()
    # Canonicalize with synonym support so inputs like 'nel' map to 'rice'
    crop_name = canonicalize_crop(user_query, crop_name)
    inputs = data.get("inputs") or {}
    if not crop_name:
        text = user_query.lower()
        # pick first crop name found in text
        for key in CROP_TO_TOOL.keys():
            if re.search(rf"\b{re.escape(key)}\b", text):
                crop_name = key
                break
        # Extract numbers by regex if missing
        def find_num(pattern: str) -> float | None:
            m = re.search(pattern, text, flags=re.I)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
            return None

        inputs = {
            "soil_ph": inputs.get("soil_ph"),
            "soil_moisture": inputs.get("soil_moisture"),
            "n": inputs.get("n"),
            "p": inputs.get("p"),
            "k": inputs.get("k"),
        }
        if inputs.get("soil_ph") is None:
            inputs["soil_ph"] = find_num(r"ph\s*([0-9]+(?:\.[0-9]+)?)")
        if inputs.get("soil_moisture") is None:
            inputs["soil_moisture"] = find_num(r"moisture\s*([0-9]+(?:\.[0-9]+)?)")
        if inputs.get("n") is None:
            inputs["n"] = find_num(r"\bn\s*([0-9]+(?:\.[0-9]+)?)")
        if inputs.get("p") is None:
            inputs["p"] = find_num(r"\bp\s*([0-9]+(?:\.[0-9]+)?)")
        if inputs.get("k") is None:
            inputs["k"] = find_num(r"\bk\s*([0-9]+(?:\.[0-9]+)?)")

    # Normalize inputs
    normalized_inputs = {
        "soil_ph": to_float_safe(inputs.get("soil_ph")),
        "soil_moisture": to_float_safe(inputs.get("soil_moisture")),
        "n": to_float_safe(inputs.get("n")),
        "p": to_float_safe(inputs.get("p")),
        "k": to_float_safe(inputs.get("k")),
    }

    new_state: AgentState = {
        "user_query": user_query,
        "crop_name": crop_name,
        "user_inputs": normalized_inputs,
        "optimized_values": {},
        "remedies_text": "",
    }
    return new_state


def router(state: AgentState) -> str:
    crop = (state.get("crop_name") or "").lower().strip()
    # Fallback to a safe default tool so the graph continues even on unknown crops
    return CROP_TO_TOOL.get(crop, "optimize_rice")


def final_response(state: AgentState) -> AgentState:
    optimized = state.get("optimized_values", {}) or {}
    remedies = state.get("remedies_text", "")
    crop = state.get("crop_name", "")

    # Guard: if we have no crop or no optimized values, show a clear message so output is never blank
    if (not str(crop).strip()) or (not optimized):
        new_state = dict(state)
        new_state["remedies_text"] = "Unknown crop, please enter a valid crop name."
        return new_state

    # Prompt to format a concise helpful response
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Format a concise, friendly agronomy recommendation as plain text."),
        (
            "human",
            """
Crop: {crop}
Optimized metrics (soil pH, soil moisture %, N kg/ha, P kg/ha, K kg/ha): {optimized}
Context and remedies:
{remedies}

Requirements:
- Present a short summary line of the key targets.
- Then list 3-5 actionable steps, including fertilization and irrigation.
- Keep under 180 words.
""".strip(),
        ),
    ])
    chain = prompt | llm
    # Safe call with fallback formatting
    text = ""
    try:
        response = chain.invoke({"crop": crop, "optimized": optimized, "remedies": remedies})
        text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        text = ""
    if not text or not text.strip():
        # Deterministic fallback formatter
        lines = []
        if crop:
            lines.append(f"Crop: {crop}")
        if optimized:
            lines.append(
                "Targets -> pH: {soil_ph}, Moisture %: {soil_moisture}, N: {n} kg/ha, P: {p} kg/ha, K: {k} kg/ha".format(
                    soil_ph=optimized.get("soil_ph", "-"),
                    soil_moisture=optimized.get("soil_moisture", "-"),
                    n=optimized.get("n", "-"),
                    p=optimized.get("p", "-"),
                    k=optimized.get("k", "-"),
                )
            )
        if remedies:
            lines.append("")
            lines.append(remedies)
        text = "\n".join(lines).strip()

    # Store final text in remedies_text to conform to state keys
    new_state = dict(state)
    new_state["remedies_text"] = text.strip()
    return new_state  # optimized_values must already be present from tool nodes


# -------------------------
# Build graph
# -------------------------
graph = StateGraph(AgentState)

graph.add_node("parse_input", parse_input)
graph.add_node("optimize_barley", lambda s: {**s, **optimize_barley(s)})
graph.add_node("optimize_cotton", lambda s: {**s, **optimize_cotton(s)})
graph.add_node("optimize_corn", lambda s: {**s, **optimize_corn(s)})
graph.add_node("optimize_potato", lambda s: {**s, **optimize_potato(s)})
graph.add_node("optimize_rice", lambda s: {**s, **optimize_rice(s)})
graph.add_node("optimize_soybean", lambda s: {**s, **optimize_soybean(s)})
graph.add_node("optimize_sugarcane", lambda s: {**s, **optimize_sugarcane(s)})
graph.add_node("optimize_sunflower", lambda s: {**s, **optimize_sunflower(s)})
graph.add_node("optimize_tomato", lambda s: {**s, **optimize_tomato(s)})
graph.add_node("optimize_wheat", lambda s: {**s, **optimize_wheat(s)})
graph.add_node("final_response", final_response)

graph.set_entry_point("parse_input")

# Router as conditional edge
graph.add_conditional_edges(
    source="parse_input",
    path=lambda s: router(s),
    path_map={
        "optimize_barley": "optimize_barley",
        "optimize_cotton": "optimize_cotton",
        "optimize_corn": "optimize_corn",
        "optimize_potato": "optimize_potato",
        "optimize_rice": "optimize_rice",
        "optimize_soybean": "optimize_soybean",
        "optimize_sugarcane": "optimize_sugarcane",
        "optimize_sunflower": "optimize_sunflower",
        "optimize_tomato": "optimize_tomato",
        "optimize_wheat": "optimize_wheat",
        END: END,
    },
)

# All tool nodes go to final_response
for node_name in [
    "optimize_barley",
    "optimize_cotton",
    "optimize_corn",
    "optimize_potato",
    "optimize_rice",
    "optimize_soybean",
    "optimize_sugarcane",
    "optimize_sunflower",
    "optimize_tomato",
    "optimize_wheat",
]:
    graph.add_edge(node_name, "final_response")

graph.add_edge("final_response", END)

app = graph.compile()


def run_agent_from_structured_inputs(
    crop: str,
    soil_type: str,
    soil_ph: str | float | None,
    soil_moisture_percent: str | float | None,
    n_kg_ha: str | float | None,
    p_kg_ha: str | float | None,
    k_kg_ha: str | float | None,
) -> dict:
    """Invoke the agent from structured inputs and return the final state dict."""
    user_text = (
        f"Crop: {crop}; soil type: {soil_type}; "
        f"soil_pH: {soil_ph}; soil_moisture: {soil_moisture_percent}% ; "
        f"N: {n_kg_ha} kg/ha; P: {p_kg_ha} kg/ha; K: {k_kg_ha} kg/ha"
    )
    initial_state: AgentState = {
        "user_query": user_text,
        "crop_name": "",
        "user_inputs": {},
        "optimized_values": {},
        "remedies_text": "",
    }
    return app.invoke(initial_state)


def run_agent_from_text(user_text: str) -> dict:
    """Invoke the agent from a raw user text and return the final state dict."""
    initial_state: AgentState = {
        "user_query": user_text,
        "crop_name": "",
        "user_inputs": {},
        "optimized_values": {},
        "remedies_text": "",
    }
    return app.invoke(initial_state)

def main() -> None:
    print("Enter crop name (barley, cotton, corn/maize, potato, rice/paddy, soybean/soya, sugarcane, sunflower, tomato, wheat):")
    crop = input().strip()
    
    print("Enter soil type (e.g., sandy, sandy loam, loam, clay):")
    soil_type = input().strip()

    print("Enter soil pH (e.g., 5.8):")
    soil_ph = input().strip()

    print("Enter soil moisture % (e.g., 45):")
    soil_moisture = input().strip()

    print("Enter Nitrogen N (kg/ha):")
    n = input().strip()

    print("Enter Phosphorus P (kg/ha):")
    p = input().strip()

    print("Enter Potassium K (kg/ha):")
    k = input().strip()

    user_text = (
        f"Crop: {crop}; soil type: {soil_type}; "
        f"soil_pH: {soil_ph}; soil_moisture: {soil_moisture}%; "
        f"N: {n} kg/ha; P: {p} kg/ha; K: {k} kg/ha"
    )

    initial_state: AgentState = {
        "user_query": user_text,
        "crop_name": "",
        "user_inputs": {},
        "optimized_values": {},
        "remedies_text": "",
        "tool": "",  # keep if you added LLM-driven tool routing
    }
    final_state = app.invoke(initial_state)
    output_text = final_state.get("remedies_text", "")
    clean_text = output_text.replace("**", "")
        
    print("\n=== Recommendation ===\n")
    print(clean_text)
    crop = "tomato"  # example crop key

    if crop in CROP_KNOWLEDGE_BASE:
        crop_data = CROP_KNOWLEDGE_BASE[crop]
        print("6. Fertilizer Tips:", crop_data.get("fertilizer_tips", "N/A"))
        print("7.Pest Control:", crop_data.get("pest_control", "N/A"))
        print("8.Plan:", crop_data.get("plan", "N/A"))
        print("9. Government Schemes:", crop_data.get("gov_schemes", "N/A"))


if __name__ == "__main__":
    main()


