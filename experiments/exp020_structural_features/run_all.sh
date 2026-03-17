#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

echo "===== Step 1: Preprocess (background removal + features) ====="
python experiments/exp020_structural_features/step1_preprocess.py

echo ""
echo "===== Step 2: Train structural model ====="
python experiments/exp020_structural_features/step2_train.py

echo ""
echo "===== Step 3: Inference + Stacking ====="
python experiments/exp020_structural_features/step3_inference.py

echo ""
echo "===== All done! ====="
