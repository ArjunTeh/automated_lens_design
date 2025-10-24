iter_budget=10000
PADDED_SEED=$(printf "%03d" $SEED)

python experiments/rjmcmc.py \
    --config experiments/config.yaml \
    --out_folder "results/mh_compare/mh" \
    --niters $iter_budget

python experiments/run_gradient_restore.py \
        --config experiments/config.yaml \
        --out_folder results/mh_compare/restore \
        --niters $(($iter_budget / 20)) \
        --local_step_iters 20 \
        --reservoir_size 5 \
