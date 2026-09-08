# CHANDRA-REG: Chandrayaan-2 Multi-Modal Image Correspondence & Registration

Software prototype for multi-modal, illumination-invariant, and scale-invariant image registration using Chandrayaan-2 (OHRC, TMC-2, IIRS) and reference lunar imagery (LRO NAC, SELENE). (ISRO SIH Problem Statement 26166).

## Pipeline Architecture
1. **Synthetic Pair Generator**: Generates controlled test cases with ground-truth homography matrices.
2. **Preprocessing**: Illumination normalization & Phase Congruency frequency domain extraction.
3. **Feature Extractor & Matcher**: SIFT + MAGSAC++ baseline and LoFTR / LightGlue deep matcher.
4. **Spatial Uniformity & Refinement**: ANMS grid bucketing and sub-pixel local NCC alignment.
5. **Evaluation Engine**: Computes Reprojection RMSE, Inlier Ratio, and Spatial Uniformity Index (SUI).

## Quick Start (Python 3.11)
```bash
# Run baseline benchmark
python main.py --mode benchmark

# Run single synthetic pair registration test
python main.py --mode synthetic

# Run unit test suite
python -m pytest tests/
```