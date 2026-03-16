#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

echo "===== Step 1: Pseudo-Label Round 2 ====="
python experiments/exp018_triple_stack/step1_pseudo_round2.py

echo ""
echo "===== Step 2: Train 3 Architectures ====="
python experiments/exp018_triple_stack/step2_train_all.py

echo ""
echo "===== Step 3: Inference + Stacking ====="
python experiments/exp018_triple_stack/step3_inference_stack.py

echo ""
echo "===== All done! ====="
