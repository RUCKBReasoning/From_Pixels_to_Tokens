export CUDA_VISIBLE_DEVICES=2
torchrun --standalone --nnodes 1 --nproc-per-node 1 main.py fit \
    --config config/lam-stage-2.yaml \
