import os
from src.preprocessing.isro_parser import ingest_isro_zip, load_isro_browse_image, extract_tile_from_strip
from src.deep_matching.loftr_matcher import match_loftr
from src.evaluation.metrics import evaluate_registration
from src.evaluation.visualizer import save_registration_visualization

def run_isro_real_test():
    tmc_zip = r'C:\Users\SODHAN KRISHNA\Downloads\ch2_tmc_nca_20260813T0627378526_d_img_d18.zip'
    ohr_zip = r'C:\Users\SODHAN KRISHNA\Downloads\ch2_ohr_ncp_20260103T0609041371_d_img_d18.zip'

    print("================================================================================")
    print("        ISRO CHANDRAAYAN-2 REAL DATASET INGESTION & REGISTRATION               ")
    print("================================================================================")

    print("Ingesting TMC-2 PDS4 Zip...")
    tmc_meta = ingest_isro_zip(tmc_zip)
    print(f"  Sensor: {tmc_meta['sensor']} | Dimensions: {tmc_meta['lines']} x {tmc_meta['samples']} | Type: {tmc_meta['data_type']}")

    print("Ingesting OHRC PDS4 Zip...")
    ohr_meta = ingest_isro_zip(ohr_zip)
    print(f"  Sensor: {ohr_meta['sensor']} | Dimensions: {ohr_meta['lines']} x {ohr_meta['samples']} | Type: {ohr_meta['data_type']}")

    tmc_full = load_isro_browse_image(tmc_meta)
    ohr_full = load_isro_browse_image(ohr_meta)

    print(f"TMC-2 Browse Image Shape: {tmc_full.shape}")
    print(f"OHRC  Browse Image Shape: {ohr_full.shape}")

    # Extract spatial tile crops from orbital strips
    ref_tile = extract_tile_from_strip(tmc_full, center_y_ratio=0.5, tile_size=800)
    mov_tile = extract_tile_from_strip(ohr_full, center_y_ratio=0.5, tile_size=800)

    print(f"Extracted TMC-2 Tile: {ref_tile.shape}")
    print(f"Extracted OHRC  Tile: {mov_tile.shape}")

    # Run LoFTR Registration on Real Chandrayaan-2 TMC-2 vs OHRC Pair!
    print("Running LoFTR Deep Transformer Registration on Real TMC-2 vs OHRC...")
    res = match_loftr(ref_tile, mov_tile)

    print("\n================================================================================")
    print("             REAL ISRO CHANDRAAYAN-2 REGISTRATION RESULTS                       ")
    print("================================================================================")
    print(f" Match Success      : {res['success']}")
    print(f" Raw Matches        : {res['num_raw_matches']}")
    print(f" Inliers Count      : {res['inlier_count']}")
    print(f" Inlier Ratio       : {res['inlier_ratio']*100:.1f}%")

    eval_res = evaluate_registration(None, res['H_est'], res['pts_ref'], res['num_raw_matches'], res['inlier_count'], ref_tile.shape)
    out_file = save_registration_visualization(ref_tile, mov_tile, res, eval_res, 'outputs/isro_real_tmc_vs_ohrc.png')
    print(f" Visualization PNG : {out_file}")
    print("================================================================================\n")

if __name__ == "__main__":
    run_isro_real_test()