from typing import List
import pandas as pd #type: ignore
from src.filter import Filter
from src.scenarios import all_filters, run_natural_disaster_scenario, run_military_scenario, run_combined_scenario
from src.multitask_formula import create_model_registry_from_filters, evaluate_formula_dnf_multitask
from src.query import SpatialQueryEngine, Query

import numpy as np
import random
import src.formula as fx
from matplotlib.patches import Polygon
import src.multitask_formula as mtl

def simulate_serval(formula, simulated_assignment):
    compute_time = 0
    filter_results = {}

    for filter_group, group_priority in formula:
        for filter_id, _ in filter_group:
            if filter_id not in filter_results:
                f = Filter.get_filter(filter_id)
                compute_time += f.time
                filter_results[filter_id] = simulated_assignment[filter_id]
            if not filter_results[filter_id]:
                break
        else:
            return group_priority, compute_time
    return 0, compute_time

def simulate_earthsight_stl(formula, simulated_assignment):
    unique_filters = set(f for term in formula for f, _ in term[0])
    _, compute_time, _, priority = fx.evaluate_formula_dnf(
        formula, unique_filters,
        lower_threshold=0.0,
        upper_threshold=1.0,
        simulated_assignment=simulated_assignment,
        mode=2
    )
    return priority, compute_time

def simulate_earthsight_multitask(formula, simulated_assignment, registry):
    _, compute_time, _, priority = evaluate_formula_dnf_multitask(
        formula,
        registry.copy(),
        lower_threshold=0.0,
        upper_threshold=1.0,
        simulated_assignment=simulated_assignment,
        debug=False
    )
    return priority, compute_time

def run_benchmark(queries: List[Query], registry) -> pd.DataFrame:
    results = {
        'mode': [],
        'compute_time': [],
        'correct': [],
        'false_positive_high_priority': [],
        'ground_truth_high_priority': [],
    }

    formula = []

    for q in queries:
        for f_seq in q.filter_categories:
            if not f_seq:
                continue  # Skip empty formulas
            term = ([(f, True) for f in f_seq], q.priority_tier)
            formula.append(term)

    if not formula:
        # Return empty DataFrame with correct columns if no formulas
        return pd.DataFrame(results)

    unique_filters = set(var for term, pri in formula for var, polarity in term)
    simulated_assignment = {
        f: (random.random() < Filter.get_filter(f).pass_probs['pass'])
        for f in unique_filters
    }

    gp = fx.ground_truth_priority(formula, simulated_assignment)

    if gp > 0:
        print("hi")

    for mode, sim_fn in [
        ("serval", simulate_serval),
        ("earthsight_stl", simulate_earthsight_stl),
        ("earthsight_multitask", lambda f, a: simulate_earthsight_multitask(f, a, registry)),
    ]:
        predicted, compute_time = sim_fn(formula, simulated_assignment)
        results['mode'].append(mode)
        results['compute_time'].append(compute_time)
        results['correct'].append(predicted == gp)
        results['false_positive_high_priority'].append(predicted > 2 and gp < 1)
        results['ground_truth_high_priority'].append(gp > 2)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Setup filters
    Filter.add_filters(all_filters)
    for f in all_filters:
        f.time = f.time * 5
        f.pass_probs['pass'] *= 5 # simulates a higher pass rate so the benchmark results are more illustrative

    # Create model registry
    registry, _ = create_model_registry_from_filters(all_filters)

    # Load scenarios
    q_military = run_military_scenario()
    q_natural = run_natural_disaster_scenario()
    q_combined = run_combined_scenario()

    qe = SpatialQueryEngine()
    qe.load_queries(q_natural + q_military)

    # Sample coordinates and fetch queries
    coordinates = [(random.uniform(-180, 180), random.uniform(-90, 90)) for _ in range(10000)]
    all_queries = []
    for coord in coordinates:
        queries = qe.get_queries_at_coord(coord, min_pri=1, max_pri=10)
        if queries:  # Only add if there are actual queries
            all_queries.append(queries)

    # Initialize an empty DataFrame to collect all results
    all_results = []

    # Run benchmark for each set of queries
    for i, queries in enumerate(all_queries):
        if i % 100 == 0:
            print(f"Processing benchmark {i}/{len(all_queries)}")
            
        df_result = run_benchmark(queries, registry.copy())
        if not df_result.empty:
            # Add instance ID for tracking
            df_result['instance_id'] = i
            all_results.append(df_result)
    
    # Combine all results
    if all_results:
        df_all_results = pd.concat(all_results, ignore_index=True)
        
        # Compute time summary
        time_summary = df_all_results.groupby('mode')['compute_time'].agg(['mean', 'median', 'std', 'min', 'max'])
        print("\n=== Compute Time Summary (in arbitrary units) ===")
        print(time_summary.round(2).to_string())
        
        # Count of benchmarks per mode
        benchmark_counts = df_all_results.groupby('mode').size()
        print("\n=== Number of Benchmarks per Mode ===")
        print(benchmark_counts.to_string())

        # Accuracy metrics
        accuracy = df_all_results.groupby('mode')['correct'].mean().round(4)
        print("\n=== Accuracy (fraction correct) ===")
        print((accuracy * 100).round(2).astype(str) + '%')

        # Amplification ratio: how many false high-priority predictions vs actual ground truth high-priority items
        high_priority_counts = df_all_results.groupby('mode')['ground_truth_high_priority'].sum()
        false_positive_counts = df_all_results.groupby('mode')['false_positive_high_priority'].sum()
        
        print("\n=== High Priority Items (Ground Truth) ===")
        print(high_priority_counts.to_string())
        
        print("\n=== False Positive High Priority Items ===")
        print(false_positive_counts.to_string())
        
        amplification_summary = false_positive_counts / high_priority_counts.replace(0, 1)
        
        print("\n=== False Positive Amplification Ratio (Extra vs Ground Truth High Priority) ===")
        print((amplification_summary.round(2).astype(str) + 'x'))
    else:
        print("No valid benchmarks were run.")