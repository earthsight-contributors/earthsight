EarthSight Satellite Simulations.

These are heavily based off of the open source simulations from Serval, provided at:
https://github.com/ConnectedSystemsLab/Serval/tree/main/filters. For the purposes of double blind review,
our code is anonymized. For clearness, we have removed code that is not crucial for the execution of our program.

To get started, execute the following commands in a Python environment.

1. python3 -m pip install requirements.txt
2. python3 run.py --mode earthsight --learning stl --scenario naturaldisaster --hardware tpu

For a rough estimate of the computational results on TPU, use:

3. python3 benchmark_filter_eval.py

or

4. python3 plot_trees.py to plot potential execution paths

Key components for EarthSight
Formula execution order - see src/multitask_formula.py and formula.py
Lookahead - see src/lookaheadsimulation.py
Threshold updates - see src/formula.py threshold_adjuster() function

We have example sheets loaded into scenarios.py for simulation traces to support reproduction of our results.
