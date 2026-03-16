#!/bin/bash
# exp018 프로세스가 끝나면 exp019 자동 시작
EXP018_PID=$1

echo "Waiting for exp018 (PID: $EXP018_PID) to finish..."
while kill -0 $EXP018_PID 2>/dev/null; do
    sleep 30
done

echo "exp018 finished! Starting exp019..."
echo ""
bash /home/pascal/Ora/kimjunhee/workspace/dacon/experiments/exp019_physics_kd/run_all.sh >> /home/pascal/Ora/kimjunhee/workspace/dacon/experiments/exp019_physics_kd/train.log 2>&1
echo "exp019 done!"
