#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

echo "===== Step 1: Video Model Training ====="
python experiments/exp019_physics_kd/step1_video_model.py

echo ""
echo "===== Step 2: Soft-Label Image Model Training ====="
python experiments/exp019_physics_kd/step2_soft_image_model.py

echo ""
echo "===== Step 3: Inference + Blending ====="
python experiments/exp019_physics_kd/step3_inference.py

echo ""
echo "===== All done! ====="
