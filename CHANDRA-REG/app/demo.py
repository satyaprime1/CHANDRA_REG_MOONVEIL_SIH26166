import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import json
import sys
from PIL import Image

# ============================================================
# STREAMLIT CLOUD IMPORT FIX
# demo.py is inside:
# CHANDRA-REG/app/demo.py
#
# src is inside:
# CHANDRA-REG/src/
#
# So we add CHANDRA-REG to Python's import path.
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ============================================================
# Import CHANDRA-REG Core Pipeline Modules
# ============================================================

from src.preprocessing.synthetic_generator import create_synthetic_pair
from src.classical.sift_matcher import match_sift
from src.deep_matching.loftr_matcher import match_loftr
from src.evaluation.metrics import evaluate_registration
from src.evaluation.visualizer import save_registration_visualization


# ============================================================
# Set Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="CHANDRA-REG | ISRO SIH Prototype",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom CSS Styling
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Application Header
# ============================================================

st.markdown(
    "<div class='main-title'>🌙 CHANDRA-REG: Multi-Modal Lunar Image Registration</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>ISRO Smart India Hackathon (PS 26166) — "
    "Illumination, Sun Angle & Scale Invariant Correspondence Engine</div>",
    unsafe_allow_html=True
)


# ============================================================
# Sidebar Controls
# ============================================================

st.sidebar.header("🎛️ Pipeline Configuration")

input_mode = st.sidebar.radio(
    "Input Source",
    [
        "Upload Custom Pair",
        "Synthetic Lunar Test Pair"
    ]
)

matcher_choice = st.sidebar.selectbox(
    "Feature Matching Engine",
    [
        "LoFTR (Deep Vision Transformer)",
        "SIFT (Classical)"
    ]
)

preprocessing_choice = st.sidebar.selectbox(
    "Illumination Preprocessing Filter",
    [
        "None",
        "CLAHE (Adaptive Equalization)",
        "Wallis (Local Variance Filter)",
        "Phase Congruency"
    ]
)


# ============================================================
# Convert selections to internal keys
# ============================================================

matcher_key = "loftr" if "LoFTR" in matcher_choice else "sift"

prep_map = {
    "None": "none",
    "CLAHE (Adaptive Equalization)": "clahe",
    "Wallis (Local Variance Filter)": "wallis",
    "Phase Congruency": "phase_congruency"
}

preprocessing_key = prep_map[preprocessing_choice]


# ============================================================
# Transformation Settings
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Transformation Settings")

if input_mode == "Synthetic Lunar Test Pair":

    scale_val = st.sidebar.slider(
        "Scale Factor",
        0.5,
        2.5,
        1.25,
        0.05
    )

    rot_val = st.sidebar.slider(
        "Rotation Angle (deg)",
        -180.0,
        180.0,
        15.0,
        5.0
    )

    illum_val = st.sidebar.slider(
        "Solar Angle Shift (deg)",
        0.0,
        90.0,
        45.0,
        5.0
    )

else:

    scale_val = 1.0
    rot_val = 0.0
    illum_val = 0.0


# ============================================================
# Run Button
# ============================================================

run_button = st.sidebar.button(
    "🚀 Run Registration Engine",
    type="primary",
    use_container_width=True
)


# ============================================================
# Data Ingestion Logic
# ============================================================

ref_img = None
moving_img = None
H_gt = None


# ============================================================
# CUSTOM IMAGE UPLOAD
# ============================================================

if input_mode == "Upload Custom Pair":

    col_up1, col_up2 = st.columns(2)

    # --------------------------------------------------------
    # Reference Image
    # --------------------------------------------------------

    with col_up1:

        up_ref = st.file_uploader(
            "Upload Reference Image (e.g. LRO NAC / TMC)",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ]
        )

        if up_ref:

            ref_img = cv2.imdecode(
                np.frombuffer(
                    up_ref.read(),
                    np.uint8
                ),
                cv2.IMREAD_GRAYSCALE
            )

            if ref_img is not None:

                st.image(
                    ref_img,
                    caption="Reference Image",
                    use_container_width=True
                )

            else:

                st.error(
                    "Unable to read the reference image."
                )


    # --------------------------------------------------------
    # Moving Image
    # --------------------------------------------------------

    with col_up2:

        up_mov = st.file_uploader(
            "Upload Moving Target Image (e.g. OHRC / IIRS)",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ]
        )

        if up_mov:

            moving_img = cv2.imdecode(
                np.frombuffer(
                    up_mov.read(),
                    np.uint8
                ),
                cv2.IMREAD_GRAYSCALE
            )

            if moving_img is not None:

                st.image(
                    moving_img,
                    caption="Moving Image",
                    use_container_width=True
                )

            else:

                st.error(
                    "Unable to read the moving image."
                )


# ============================================================
# SYNTHETIC LUNAR TEST PAIR
# ============================================================

else:

    try:

        pair = create_synthetic_pair(
            scale=scale_val,
            rotation_deg=rot_val,
            illumination_angle_deg=illum_val
        )

        ref_img = pair["reference"]
        moving_img = pair["moving"]
        H_gt = pair["H_GT"]

        col_preview1, col_preview2 = st.columns(2)

        with col_preview1:

            st.image(
                ref_img,
                caption="Synthetic Reference Image (800x800)",
                use_container_width=True
            )

        with col_preview2:

            st.image(
                moving_img,
                caption=(
                    f"Synthetic Moving Image "
                    f"(Scale: {scale_val}x, "
                    f"Rot: {rot_val}°, "
                    f"Solar: {illum_val}°)"
                ),
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "Failed to generate synthetic lunar image pair."
        )

        st.exception(e)


# ============================================================
# Execute Registration Pipeline
# ============================================================

if run_button or (
    input_mode == "Synthetic Lunar Test Pair"
):

    if ref_img is not None and moving_img is not None:

        try:

            with st.spinner(
                "Processing image correspondence & "
                "estimating geometric transformation..."
            ):

                # =================================================
                # FEATURE MATCHING
                # =================================================

                if matcher_key == "loftr":

                    match_res = match_loftr(
                        ref_img,
                        moving_img
                    )

                else:

                    match_res = match_sift(
                        ref_img,
                        moving_img,
                        preprocessing=preprocessing_key
                    )


                # =================================================
                # INLIER POINT EXTRACTION
                # =================================================

                if match_res["success"]:

                    inliers_pts = match_res["pts_ref"][
                        match_res["inliers_mask"]
                    ]

                else:

                    inliers_pts = np.array([])


                # =================================================
                # EVALUATION
                # =================================================

                eval_res = evaluate_registration(

                    H_gt=H_gt,

                    H_est=match_res["H_est"],

                    pts_ref_inliers=inliers_pts,

                    num_raw_matches=match_res[
                        "num_raw_matches"
                    ],

                    inlier_count=match_res[
                        "inlier_count"
                    ],

                    img_shape=ref_img.shape
                )


            # ====================================================
            # PERFORMANCE SCORECARD
            # ====================================================

            st.markdown("---")

            st.subheader(
                "📊 Performance & Metric Scorecard"
            )

            m1, m2, m3, m4 = st.columns(4)


            # ----------------------------------------------------
            # Inlier Count
            # ----------------------------------------------------

            with m1:

                st.metric(
                    "Inlier Match Count",
                    f"{eval_res['inlier_count']} pts",
                    delta=(
                        f"{eval_res['num_raw_matches']} "
                        "raw matches"
                    )
                )


            # ----------------------------------------------------
            # Inlier Ratio
            # ----------------------------------------------------

            with m2:

                st.metric(
                    "Inlier Ratio",
                    f"{eval_res['inlier_ratio'] * 100:.1f}%",
                    delta="Cleanliness"
                )


            # ----------------------------------------------------
            # RMSE
            # ----------------------------------------------------

            with m3:

                rmse = eval_res[
                    "reprojection_rmse"
                ]

                if rmse != float("inf"):

                    rmse_str = f"{rmse:.2f} px"

                else:

                    rmse_str = "N/A"


                st.metric(
                    "Corner Reprojection RMSE",
                    rmse_str,
                    delta=(
                        "Sub-pixel Precision"
                        if rmse < 0.5
                        else "Coarse Fit"
                    )
                )


            # ----------------------------------------------------
            # Spatial Uniformity
            # ----------------------------------------------------

            with m4:

                st.metric(
                    "Spatial Uniformity (SUI)",
                    (
                        f"{eval_res['spatial_uniformity_index']:.3f}"
                    ),
                    delta="Range: 0.0 to 1.0"
                )


            # ====================================================
            # VISUAL OUTPUT TABS
            # ====================================================

            st.markdown(
                "### 🖼️ Registration Output & Alignment Analysis"
            )

            tab1, tab2, tab3 = st.tabs(
                [
                    "Registered Alignment Output",
                    "Absolute Error Heatmap",
                    "Inlier Distribution Map"
                ]
            )


            # ====================================================
            # REGISTER MOVING IMAGE
            # ====================================================

            h, w = ref_img.shape[:2]

            if (
                match_res["success"]
                and match_res["H_est"] is not None
            ):

                registered_img = cv2.warpPerspective(
                    moving_img,
                    match_res["H_est"],
                    (w, h)
                )

                diff_map = cv2.absdiff(
                    ref_img,
                    registered_img
                )

            else:

                registered_img = np.zeros_like(
                    ref_img
                )

                diff_map = np.zeros_like(
                    ref_img
                )


            # ====================================================
            # TAB 1 — REGISTERED ALIGNMENT
            # ====================================================

            with tab1:

                c1, c2 = st.columns(2)

                with c1:

                    st.image(
                        ref_img,
                        caption="1. Reference Target Frame",
                        use_container_width=True
                    )

                with c2:

                    st.image(
                        registered_img,
                        caption=(
                            "2. Registered / "
                            "Aligned Output Image"
                        ),
                        use_container_width=True
                    )


            # ====================================================
            # TAB 2 — ABSOLUTE ERROR HEATMAP
            # ====================================================

            with tab2:

                fig, ax = plt.subplots(
                    figsize=(8, 6)
                )

                im = ax.imshow(
                    diff_map,
                    cmap="inferno"
                )

                ax.set_title(
                    "Absolute Pixel Difference Error Map "
                    "(Dark = Perfect Alignment)"
                )

                ax.axis("off")

                fig.colorbar(
                    im,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04
                )

                st.pyplot(
                    fig,
                    clear_figure=True
                )

                plt.close(fig)


            # ====================================================
            # TAB 3 — INLIER DISTRIBUTION
            # ====================================================

            with tab3:

                fig_pts, ax_pts = plt.subplots(
                    figsize=(8, 6)
                )

                ax_pts.imshow(
                    ref_img,
                    cmap="gray"
                )

                if len(inliers_pts) > 0:

                    ax_pts.scatter(
                        inliers_pts[:, 0],
                        inliers_pts[:, 1],
                        c="lime",
                        s=12,
                        alpha=0.8,
                        label=(
                            f"Inliers "
                            f"({len(inliers_pts)})"
                        )
                    )

                    ax_pts.legend(
                        loc="upper right"
                    )

                ax_pts.set_title(
                    "Spatially Distributed "
                    "Inlier Feature Keypoints"
                )

                ax_pts.axis("off")

                st.pyplot(
                    fig_pts,
                    clear_figure=True
                )

                plt.close(fig_pts)


            # ====================================================
            # EXPORT BUTTONS
            # ====================================================

            st.markdown("---")

            col_dl1, col_dl2 = st.columns(2)


            # ----------------------------------------------------
            # Registered Image Download
            # ----------------------------------------------------

            with col_dl1:

                is_success, buffer = cv2.imencode(
                    ".png",
                    registered_img
                )

                if is_success:

                    st.download_button(
                        "💾 Download Registered Image (PNG)",
                        data=buffer.tobytes(),
                        file_name=(
                            "chandra_registered_output.png"
                        ),
                        mime="image/png",
                        use_container_width=True
                    )


            # ----------------------------------------------------
            # Metrics JSON Download
            # ----------------------------------------------------

            with col_dl2:

                metrics_json = json.dumps(
                    eval_res,
                    indent=2
                )

                st.download_button(
                    "📄 Download Metrics Report (JSON)",
                    data=metrics_json,
                    file_name=(
                        "registration_metrics.json"
                    ),
                    mime="application/json",
                    use_container_width=True
                )


        # ========================================================
        # ERROR HANDLING
        # ========================================================

        except Exception as e:

            st.error(
                "❌ Registration pipeline failed."
            )

            st.exception(e)


    else:

        st.info(
            "Please upload both Reference and Moving "
            "images to run registration."
        )
