iter_budget=10000
PADDED_SEED=$(printf "%03d" $SEED)

python experiments/run_gradient_restore.py \
        --config experiments/config.yaml \
        --out_folder results/addrem_50mm/restore \
        --niters $(($iter_budget / 20)) \
        --local_step_iters 20 \
        --reservoir_size 5

python experiments/addrem_enumerate.py \
    --config experiments/config.yaml \
    --out_folder "results/addrem_50mm/brute" \
    --niters $iter_budget
