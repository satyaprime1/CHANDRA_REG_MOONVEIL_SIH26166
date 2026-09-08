"""
MoonVeil MK1 — Streamlit Interactive Presentation Dashboard

Launch via:
    streamlit run app.py
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from main import run_moonveil
from config import default_moonveil_config
from src.visualization import (
    draw_matches,
    draw_reliability_matches,
    draw_registration_overlay,
    draw_spatial_distribution,
)

st.set_page_config(
    page_title="MoonVeil MK1 — Chandrayaan-2 Image Registration",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌙 MoonVeil MK1 — Chandrayaan-2 Image Registration")
st.caption("SIH Problem Statement SIH26166 — Sub-pixel Optical Image Correspondence, Geometric Verification & Registration")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Pipeline Configuration")

sensor_ref = st.sidebar.selectbox("Reference Sensor", ["OHRC", "TMC", "IIRS"], index=0)
sensor_src = st.sidebar.selectbox("Source Sensor", ["OHRC", "TMC", "IIRS"], index=0)

st.sidebar.subheader("Matching Algorithm")
matcher_choice = st.sidebar.radio(
    "Select Feature Matcher",
    ["SIFT (Sub-pixel Refinement)", "AI Matcher (DIS Dense Flow / LightGlue)"],
    index=0,
)

st.sidebar.subheader("Preprocessing Settings")
use_clahe = st.sidebar.checkbox("Enable CLAHE Contrast Enhancement", value=False)
use_illum = st.sidebar.checkbox("Enable Illumination Normalization", value=False)

st.sidebar.subheader("Registration Decision Gate")
min_inliers = st.sidebar.number_input("Min Inliers Required", min_value=4, max_value=100, value=8)
max_rmse = st.sidebar.number_input("Max Allowable RMSE (px)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)

# --- DATA INPUT ---
st.header("1. Input Data Selection")

data_source = st.radio(
    "Choose Input Source",
    ["Use Built-in Demo Pair (Known Ground-Truth Synthetic Pair)", "Upload Custom Image Files"],
    horizontal=True,
)

ref_path = None
src_path = None

if data_source == "Use Built-in Demo Pair (Known Ground-Truth Synthetic Pair)":
    demo_dir = Path(__file__).parent / "data" / "_synthetic_demo"
    if (demo_dir / "reference.tif").exists() and (demo_dir / "source.tif").exists():
        ref_path = demo_dir / "reference.tif"
        src_path = demo_dir / "source.tif"
        st.info("Loaded demo pair: `data/_synthetic_demo/reference.tif` and `source.tif`")
    else:
        st.error("Demo files not found in data/_synthetic_demo/")
else:
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        uploaded_ref = st.file_uploader("Upload Reference Image (TIFF/PNG/JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])
    with col_u2:
        uploaded_src = st.file_uploader("Upload Source Image (TIFF/PNG/JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])

    if uploaded_ref and uploaded_src:
        tmp_dir = tempfile.mkdtemp()
        ref_path = Path(tmp_dir) / uploaded_ref.name
        src_path = Path(tmp_dir) / uploaded_src.name
        with open(ref_path, "wb") as f:
            f.write(uploaded_ref.getbuffer())
        with open(src_path, "wb") as f:
            f.write(uploaded_src.getbuffer())

if ref_path and src_path:
    # Build Config
    cfg = default_moonveil_config()
    cfg.preprocessing.contrast_method = "CLAHE" if use_clahe else None
    cfg.preprocessing.illumination_normalization = use_illum
    cfg.registration_gate.min_trusted_matches = min_inliers
    cfg.registration_gate.max_rmse = max_rmse
    cfg.features.subpixel_refinement = True

    if "AI Matcher" in matcher_choice:
        cfg.ai_matcher.enabled = True
        cfg.ai_matcher.model_name = "auto"
    else:
        cfg.ai_matcher.enabled = False

    if st.button("🚀 Run MoonVeil Registration Pipeline", type="primary"):
        with st.spinner("Processing image pair through pipeline..."):
            result = run_moonveil(ref_path, src_path, sensor_ref, sensor_src, config=cfg)
            st.session_state["result"] = result

if "result" in st.session_state:
    res = st.session_state["result"]
    ev = res.evaluation

    st.markdown("---")
    st.header("2. Registration Results & Metrics")

    # Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        if ev.registration_success:
            st.success("SUCCESS", icon="✅")
        else:
            st.error("REJECTED", icon="🚫")
        st.caption("Decision Gate")

    with m2:
        st.metric("Verified Inliers", f"{ev.verified_inliers} / {ev.candidate_matches}")
    with m3:
        st.metric("Inlier Ratio", f"{ev.inlier_ratio:.1%}")
    with m4:
        st.metric("RMSE", f"{ev.rmse:.2f} px")
    with m5:
        st.metric("Spatial Coverage", f"{ev.spatial_coverage:.1%}")
    with m6:
        st.metric("Reliability Score", f"{ev.overall_reliability:.1f}", delta=ev.reliability_status)

    if not ev.registration_success:
        st.warning(f"⚠️ Registration Refused: {res.registration.failure_reason}")

    # Tabs for Visualizations
    st.markdown("---")
    st.header("3. Interactive Visual Inspection")

    tab1, tab2, tab3, tab4 = st.columns(4)

    t1, t2, t3, t4 = st.tabs([
        "🔍 Feature Matches & Reliability",
        "🌐 Spatial Grid Coverage",
        "🖼️ Warped Registration Overlay",
        "📄 Metadata & Metrics JSON",
    ])

    with t1:
        st.subheader("Match Candidate Reliability Classification")
        if res.candidate_matches:
            vis_matches = draw_reliability_matches(res)
            st.image(vis_matches[:, :, ::-1], caption="Green: HIGH | Yellow: MEDIUM | Blue: LOW | Red: REJECTED Outlier", use_column_width=True)
        else:
            st.info("No matches available to display.")

    with t2:
        st.subheader("Final 4x4 Spatial Distribution Grid over Trusted Matches")
        vis_spatial = draw_spatial_distribution(res)
        st.image(vis_spatial[:, :, ::-1], caption=f"Spatial Coverage: {ev.spatial_coverage:.1%}", use_column_width=True)

    with t3:
        st.subheader("Source Image Warped & Registered to Reference Frame")
        if res.registration.registered_image is not None:
            vis_overlay = draw_registration_overlay(res)
            st.image(vis_overlay[:, :, ::-1], caption="Red/Cyan Channel Overlay of Registered Image Pair", use_column_width=True)
        else:
            st.warning("Registration was not approved; warped overlay unavailable.")

    with t4:
        st.subheader("PDS4 & Evaluation Metadata")
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.write("Reference Image Metadata:")
            st.json(res.reference.metadata)
        with meta_col2:
            st.write("Source Image Metadata:")
            st.json(res.source.metadata)

        st.subheader("Full JSON Metrics Export")
        metrics_dict = {
            "pair_id": res.pair_id,
            "sensor_reference": res.sensor_reference,
            "sensor_source": res.sensor_source,
            "candidate_matches": ev.candidate_matches,
            "verified_inliers": ev.verified_inliers,
            "inlier_ratio": ev.inlier_ratio,
            "rmse": ev.rmse,
            "spatial_coverage": ev.spatial_coverage,
            "overall_reliability": ev.overall_reliability,
            "reliability_status": ev.reliability_status,
            "registration_success": ev.registration_success,
        }
        st.json(metrics_dict)
        st.download_button(
            "📥 Download Metrics JSON",
            data=json.dumps(metrics_dict, indent=2),
            file_name=f"{res.pair_id}_metrics.json",
            mime="application/json",
        )
