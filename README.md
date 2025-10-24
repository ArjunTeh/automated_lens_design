# Automated Compound Lens Design
This is the code to accompany the paper [Automated design of compound lenses with discrete-continuous optimization](https://imaging.cs.cmu.edu/automated_lens_design/index.html)

# Installation
To install all the dependencies use
```bash
pip install -r requirements.txt
```
We recommend using virtual environments since the code relies on some deprecated features of the dependent libraries.

This codebase primarily relies on JAX and the equinox library for computation and plotly for plotting

# Examples
There are two scripts that run experiments presented in the paper. For the brute force search, run `source run_brute_force_comparison.sh` and for comparison with Metropolis-Hastings, run `source run_mh_comparison.sh`.

The main logic for the algorithm in the paper is implemented in `jump_restore` function in `gradient_restore.py` (line 270).
