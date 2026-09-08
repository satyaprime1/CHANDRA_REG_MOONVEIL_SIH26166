# MoonVeil MK1

Reliability-aware lunar image correspondence and registration system for
Chandrayaan-2 optical imagery (OHRC / TMC-2 / IIRS), built for SIH26166.

**Status:** repository scaffolded. Modules being implemented in dependency
order — see `EXECUTION_PLAN.md` for the exact build/test/debug sequence.

## Required real data (not included in this repo)

Place curated ISRO image pairs under:

```
data/OHRC/pair_XX/{reference,source}.tif
data/TMC/pair_XX/{reference,source}.tif
data/IIRS/pair_XX/{reference,source}.tif
```

Obtain via ISRO's PRADAN/ISSDC distribution. Do not commit raw ISRO
archives to version control — see `.gitignore`.

## Running (once implemented)

```bash
pip install -r requirements.txt
python main.py --reference data/OHRC/pair_01/reference.tif \
                --source data/OHRC/pair_01/source.tif \
                --sensor OHRC
```

## Architecture

See the project blueprints for full detail. Short version: load → preprocess
→ detect features → match → verify geometry (RANSAC) → score reliability →
reject unreliable matches → register → evaluate → visualize.
