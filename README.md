# Bandit Convex Optimization with Unbounded Memory

This repository supplements the paper "Bandit Convex Optimization with Unbounded Memory" by providing an implementation of the algorithms presented in it and offering empirical results.

# Usage

1. Change directory to `bco-um`.
2. To run Experiment 1 (testing different values of rho), run `python bco_test_1.py --d {dimension} --rho {rho value} --T {time horizon} --iterations {number of times to run}`. Alternatively, running `python bco_test_1.py` without the additional arguments will run the experiment with default values.
3. To run Experiment 2 (comparing BCO-UM to BCO-M), run `python bco_test_2.py --d {dimension} --rho {value for rho} --T {time horizon} --iterations {number of times to run} --mem_length {length of memory for BCO-M}`. Again, running `python bco_test_2.py` without the additional arguments will run the experiment with default values.
4. For both experiments, if you wish to experiment with multiple values of rho or multiple values for mem_length, this can be done by separating each value with a space. For example, if you wish to run Experiment 1 with rho values 0.5, 0.3, and 0.1, run the command `python bco_test_1.py --d 2 --rho 0.5 0.3 0.1 --T 1000 --iterations 25`.
5. Plots will be stored in the `plots` directory. 
