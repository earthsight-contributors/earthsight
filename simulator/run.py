from typing import List
import pandas as pd #type: ignore
from src.metrics import Metrics
from src.station import Station
from src.satellite import Satellite
from src.earthsightgs import EarthSightGroundStation
from src.utils import Print, Time, Location
from src.simulator import Simulator
from src.earthsightsatellite import EarthsightSatellite
from src.scheduler import EarthSightScheduler
from src.filter import Filter
from src.scenarios import run_natural_disaster_scenario, run_military_scenario, run_combined_scenario, run_coverage_scaling_scenario, get_all_filters
from src.multitask_formula import create_model_registry_from_filters
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import statistics
import json
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="serval", help="Mode to run: serval or earthsight")
    parser.add_argument("--scenario", default="naturaldisaster", help="Scenario type: naturaldisaster, intelligence, or combined")
    parser.add_argument("--learning", default="mtl", help="mtl or stl")
    parser.add_argument("--hardware", default="tpu", help="Hardware to run on: tpu or cpu")
    args = parser.parse_args()

    if args.hardware.lower() in ["tpu", "edgetpu", "edge tpu", "coral"]:
        print("Running on Edge TPU")
        args.hardware = "tpu"
    else:
        print("Running on GPU")
        args.hardware = "gpu"

    print("Running simulations for {} scenario in mode {} with {} learning on {}".format(args.scenario, args.mode, args.learning, args.hardware))
    all_filters = get_all_filters(hardware=args.hardware)

    Filter.add_filters(all_filters)
    for filter in all_filters:
        filter.false_negative_rate = 0.03

    registry = create_model_registry_from_filters(all_filters)[0] if args.learning == "mtl" else None

    if args.scenario == "naturaldisaster":
        queries = run_natural_disaster_scenario()
    elif args.scenario == "intelligence":
        queries = run_military_scenario()
    elif args.scenario == "combined":
        queries = run_combined_scenario()
    else:
        queries = run_coverage_scaling_scenario([20])

    stations = pd.read_json("referenceData/planet_stations.json")
    groundStations: 'List[Station]' = []

    satellites = Satellite.load_from_tle("referenceData/planet_tles.txt")
    satellites = [EarthsightSatellite(i, mode=args.mode, mtl_registry=registry) for i in satellites]

    for satellite in satellites:
        satellite.compute_power = 30 * 1000 if args.hardware == 'gpu' else 2 * 1000 # Set compute power for hardware options

    startTime = Time().from_str("2025-02-01 14:00:00")
    endTime = Time().from_str("2025-02-05 14:00:00")

    sim = Simulator(60, startTime, endTime, satellites, groundStations)
    scheduler = EarthSightScheduler(queries, satellites, groundStations, sim.time, serval_mode=True)

    for id, row in stations.iterrows():
        s = Station(row["name"], id, Location().from_lat_long(row["location"][0], row["location"][1]))
        groundStations.append(EarthSightGroundStation(s, scheduler=scheduler))   

    Metrics.metr()
    sim.run()

    # FAST ANALYTICS
    # Convert the rcv_data into a JSON-safe form
    rcv_json = {sat.id: data for sat, data in EarthSightGroundStation.rcv_data.items()}
    cmpt_json = {sat.id: sat.computation_queue_sizes for sat in satellites}

    # Initialize storage for analytics

    percent_annotated = []
    unavoidable_delay_by_priority = {i: [] for i in range(-1, 11)}  # store list of unavoidable delays (in minutes)
    delay_by_priority = {i: [] for i in range(-1, 11)}  # store list of delays (in minutes)
    count_by_priority = {i: 0 for i in range(-1, 11)}   # counts per priority
    sum_by_priority = {i: 0 for i in range(-1, 11)}     # total delay per priority (in seconds)

    delay_high_pri = []
    delay_low_pri = []
    delay_hi_total = 0
    delay_low_total = 0
    count_hi_total = 0
    count_low_total = 0

    for sat, data in EarthSightGroundStation.rcv_data.items():
        # Percent annotated
        total_annotated = data['count_1'] + data['count_0']
        total_received = total_annotated + data['count_-1']
        percent_annotated.append(total_annotated / total_received if total_received else 0)

        # High- and low-priority averages
        if data['count_1']:
            delay_high_pri.append(data[1] / data['count_1'] / 60)  # convert to minutes
            delay_hi_total += data[1]
            count_hi_total += data['count_1']
        if data['count_0']:
            delay_low_pri.append(data[0] / data['count_0'] / 60)
            delay_low_total += data[0]
            count_low_total += data['count_0']

        # Populate extended statistics per priority
        for i in range(-1, 11):
            count = data.get(f'count_{i}', 0)
            total_delay = data.get(i, 0)
            unavoidable_delay = data.get(f'unavoidable_delay_{i}', 0)
            if count > 0:
                avg_delay_minutes = (total_delay / count) / 60
                avg_unavoidable_delay_minutes = (unavoidable_delay / count) / 60
                delay_by_priority[i].append(avg_delay_minutes)
                unavoidable_delay_by_priority[i].append(avg_unavoidable_delay_minutes)
                count_by_priority[i] += count
                sum_by_priority[i] += total_delay

    # Report results
    print("Running simulations for {} scenario in mode {} with {} learning on {}".format(args.scenario, args.mode, args.learning, args.hardware))

    print("========== Summary Statistics ==========")
    
    print("Percent Annotated: ", statistics.mean(percent_annotated))

    print("\nHigh Priority Delay (Mean, Std Dev) [Minutes]:")
    if delay_high_pri:
        print("  ", statistics.mean(delay_high_pri), statistics.stdev(delay_high_pri))
    else:
        print("  No high-priority data.")

    print("Low Priority Delay (Mean, Std Dev) [Minutes]:")
    if delay_low_pri:
        print("  ", statistics.mean(delay_low_pri), statistics.stdev(delay_low_pri))
    else:
        print("  No low-priority data.")

    if count_hi_total > 0:
        print("Weighted Average Delay (High Priority) [Minutes]:", delay_hi_total / count_hi_total / 60)
    if count_low_total > 0:
        print("Weighted Average Delay (Low Priority) [Minutes]:", delay_low_total / count_low_total / 60)

    print("\nDelay at Satellite by Priority Level [-1 to 10] (Mean, Std Dev) [Minutes]:")
    for i in range(0, 11):
        delays = delay_by_priority[i]
        if delays:
            mean_d = statistics.mean(delays)
            std_d = statistics.stdev(delays) if len(delays) > 1 else 0.0
            p90 = np.percentile(delays, 90)
            p99 = np.percentile(delays, 99)
            print(f"  Priority {i}: Mean = {mean_d:.2f}, Std Dev = {std_d:.2f}, 90th = {p90:.2f}, 99th = {p99:.2f}")

    
    print("\nUnavoidable Delay at Satellite by Priority Level [-1 to 10] (Mean, Std Dev) [Minutes]:")
    for i in range(0, 11):
        delays = unavoidable_delay_by_priority[i]
        if delays:
            mean_d = statistics.mean(delays)
            std_d = statistics.stdev(delays) if len(delays) > 1 else 0.0
            p90 = np.percentile(delays, 90)
            p99 = np.percentile(delays, 99)
            print(f"  Priority {i}: Mean = {mean_d:.2f}, Std Dev = {std_d:.2f}, 90th = {p90:.2f}, 99th = {p99:.2f}")


    # Save to file
    with open("rcv_data.json", "w") as f:
        json.dump(rcv_json, f, indent=2)

    with open ("compute_queues.json", "w") as f:
        json.dump(cmpt_json, f, indent=2)

    # analyze computation times
    computation_times = EarthsightSatellite.computation_times
    # find mean, std, 25, 50, 75, 90, 99 percentiles
    mean = statistics.mean(computation_times)
    std = statistics.stdev(computation_times)
    p25 = np.percentile(computation_times, 25)
    p50 = np.percentile(computation_times, 50)
    p75 = np.percentile(computation_times, 75)
    p90 = np.percentile(computation_times, 90)
    p99 = np.percentile(computation_times, 99)
    print("========== Computation Times ==========")
    print("Mean: ", mean, "Std: ", std, "25th: ", p25, "50th: ", p50, "75th: ", p75, "90th: ", p90, "99th: ", p99)
    print("========================================")


    print("Power generated:", EarthsightSatellite.power_generation)
    print("Power consumed:", EarthsightSatellite.power_consumptions)

    #json the compute times
    

    # average nonzero queue length
    avg_queue_length = []
    p90_queue_length = []
    for satellite in satellites:
        avg_queue_length.append(statistics.mean([i for i in satellite.computation_queue_sizes]))
        p90_queue_length.append(np.percentile([i for i in satellite.computation_queue_sizes], 90))

    print("__________________________________Average Queue Lengths:____________________________________")
    print(statistics.mean(avg_queue_length), statistics.stdev(avg_queue_length), statistics.mean(p90_queue_length), statistics.stdev(p90_queue_length))


        


    