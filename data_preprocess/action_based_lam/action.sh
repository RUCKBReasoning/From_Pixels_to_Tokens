export CUDA_VISIBLE_DEVICES=0
timestamp=$(date +"%Y%m%d-%H%M%S")

python action_tokenizer.py \
    --data-root-dir /path_to/RoboTwin \
    --data-mix data-mix \
    --results-dir ./results_dir \
    --num-steps 100000 \
    --use_swanlab False \
    --save-every 50000 \
