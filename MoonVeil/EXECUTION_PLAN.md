# MoonVeil MK1 — Execution Plan (JARVIS's working playbook)

Purpose of this file: so I can proceed module-by-module — build, run,
test, debug, explain — without needing fresh instructions each time. When
told to "continue" or "next", I consult the STATE block below, execute the
next incomplete step, update STATE, and report back plainly.

No real ISRO data yet. Every module is validated first against synthetic
data (a real image + a known synthetic transform) so correctness is proven
independent of dataset availability. Real-pair testing slots in later
without touching the code.

---

## STATE (update after every module)

```
current_module: NONE
last_completed: FULL SKELETON COMPLETE.
  - src/visualization.py: draw_matches, draw_reliability_matches,
    draw_registration_overlay, draw_spatial_distribution — all draw-only,
    return BGR images, never save. 1/1 smoke test passes.
  - src/registration.py: added should_register() decision gate alongside
    register_image(). Structured to reject on: geometry failure, too few
    trusted matches, low inlier ratio, poor spatial coverage, high RMSE.
  - main.py: run_moonveil() wires the ENTIRE pipeline end-to-end, correctly
    following the blueprint's "Important Correction" ordering — candidate
    spatial distribution is computed BEFORE geometry (feeds reliability's
    spatial component), and a SEPARATE final spatial distribution is
    computed AFTER trusted-match filtering (this is what's reported in
    evaluation/decision-gate). CLI via argparse matches the documented
    contract exactly, writes metrics.json per pair to outputs/<pair_id>/.
  - tests/test_main_e2e.py: 2/2 end-to-end tests pass —
      Case A (good synthetic pair): 671 candidates -> 651 inliers (97%),
        RMSE 0.50px, overall 93.8 (HIGH), registration SUCCESS.
      Case C (two unrelated textures, no real correspondence): pipeline
        completes with NO crash and correctly reports
        "REGISTRATION NOT RELIABLE: ... Only 7 RANSAC inliers; minimum is 8."
  - Live CLI run confirmed against a synthetic demo pair (now relocated to
    data/_synthetic_demo/, clearly labeled as NOT real ISRO data, so it can
    never be mistaken for real validation).
next_up: REAL DATA. All 12 modules + full pipeline are built and tested on
  synthetic ground truth. Nothing further should be built until real
  OHRC/TMC/IIRS pairs are available to tune thresholds against.
blocking_issues: none — but real-data threshold tuning is the critical path
real_data_available: NO
notes: config.py had several default_*_config() factory functions
  accidentally dropped during sequential str_replace edits earlier in the
  session (each new block replaced its anchor rather than appending) —
  caught immediately via a config-import sanity check before declaring the
  skeleton done, and fixed. All 8 config classes + their defaults verified
  present via grep before final regression.
  Full regression: 12/12 test files green (types, loader, preprocessing,
  features, matching, geometry, spatial, reliability, registration,
  evaluation, visualization, main_e2e).
  Reference "known-good" synthetic numbers (for sanity-checking real data
  behavior later): ~670 candidates, ~650 inliers (97%), RMSE ~0.5px,
  100% spatial coverage, overall reliability ~94 (HIGH).
```

---

## Build order & per-module protocol

For EVERY module, the same five-step protocol applies:

1. **Implement** — write the module per its contract in
   Module_Interfaces_&_Data_Schemas, nothing more, no scope creep.
2. **Self-test** — run a synthetic sanity test in `tests/`.
3. **Explain** — plain-language: what goes in, what comes out, key
   parameters, what can make it fail.
4. **Debug if needed** — fix in place before moving on. Never proceed on a
   failing module.
5. **Update STATE** — mark done, move `next_up` forward.

Do NOT skip ahead. Do NOT batch multiple modules into one giant file.
Do NOT introduce dashboard, AI matcher, or sub-pixel logic before the
baseline pipeline runs end-to-end.

---

### 1. `src/types.py`
- Dataclasses: ImageData, ImagePair, PreprocessedImage, FeatureSet,
  MatchCandidate, GeometricResult, MatchReliability, SpatialDistribution,
  RegistrationResult, EvaluationResult, MoonVeilResult, PipelineStatus,
  TrustedCorrespondenceSet, SensorType enum.
- Test: instantiate each dataclass with dummy values, confirm no errors.
- No image-processing logic here at all.

### 2. `src/data_loader.py`
- `load_image(path, sensor, band=None) -> ImageData`
- Test: load a real (non-ISRO, any sample) TIFF/PNG → correct ImageData;
  missing file → FileNotFoundError; bad sensor string → ValueError.

### 3. `src/preprocessing.py`
- `preprocess_image(image_data, config) -> PreprocessedImage`
- Never mutates original array. Test: same output dims unless resize_factor
  set; normalized range check (0-255 uint8).

### 4. `src/features.py`
- `detect_features(image, config) -> FeatureSet` (SIFT baseline)
- Test: non-empty keypoints/descriptors on any real image; empty case
  raises ValueError.

### 5. `src/matching.py`
- `match_features(source, reference, config) -> list[MatchCandidate]`
- BFMatcher + KNN(k=2) + Lowe ratio test.
- Test: same image vs itself → high match count, ratio near-perfect.

### 6. `src/geometry.py`
- `estimate_geometry(matches, config) -> GeometricResult`
- `calculate_reprojection_error`, `calculate_rmse` (can live here).
- Test: synthetic known-transform pair → recovered homography close to
  ground truth; too-few-matches → failed GeometricResult, not an exception.

### 7. `src/spatial.py`
- `analyze_spatial_distribution(points, image_shape, config) -> SpatialDistribution`
- Test: evenly spread synthetic points → high coverage_ratio; clustered
  points → low coverage_ratio.

### 8. `src/reliability.py`
- `calculate_match_reliability(matches, geometry, spatial, config) -> list[MatchReliability]`
- Weighted composite score, 0-100, status HIGH/MEDIUM/LOW/REJECTED.
- Test: known-good synthetic inlier → high score; known-bad/outlier → low
  score. Weight sum validated == 1.0.

### 9. `src/registration.py`
- `register_image(source, transformation_matrix, output_shape) -> RegistrationResult`
- Does NOT judge trustworthiness — that's upstream via `should_register`.
- Test: synthetic known transform → warped image aligns with reference
  within small pixel tolerance.

### 10. `src/evaluation.py`
- `evaluate(...) -> EvaluationResult` — pure aggregation, no image ops.
- Test: feed known inputs, check arithmetic (inlier_ratio, RMSE, etc.)
  matches hand-calculated expected values.

### 11. `src/visualization.py`
- `draw_matches`, `draw_registration_overlay`, `draw_spatial_distribution`,
  `draw_reliability_matches` — all return images, never save directly.
- Test: functions run without error and return arrays of expected shape.

### 12. `should_register` decision gate + `main.py` (`run_moonveil`)
- Wires the full pipeline per the dependency order in the blueprint.
- Test: full synthetic end-to-end run produces a MoonVeilResult with
  registration_success=True and sane metrics.

### 13. CLI wiring (`argparse` in main.py)
- Matches the CLI contract in Module_Interfaces doc §41.

### 14. — GATE — Real data integration
- Only once (1)-(13) pass on synthetic data. When real OHRC/TMC/IIRS pairs
  arrive: drop them into `data/<SENSOR>/pair_XX/`, run the CLI, tune
  thresholds in config.py against real behaviour. No code restructuring
  expected — if it is needed, that itself is a signal to flag to the user.

### 15. — Optional, only after 14 succeeds —
- AI matcher comparison (`src/ai_matcher.py`), Streamlit dashboard,
  sub-pixel refinement experiment. Each independently discardable per the
  blueprint's hard-stop conditions.

---

## Standing rules I hold myself to

- Never sacrifice a working baseline for an in-progress enhancement.
- Never fake a metric or claim sub-pixel accuracy without evidence.
- Always show synthetic-test evidence before calling a module "done".
- Always explain each module in plain terms once it works.
- Stop and flag to the user (rather than silently improvising) if a
  module's real-world behavior contradicts its written contract.
