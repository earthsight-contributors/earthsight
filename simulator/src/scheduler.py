import pickle
from src.satellite import Satellite
from src.schedule import Schedule, ScheduleItem
from src.utils import Time
from src.lookaheadsimulation import LookaheadSimulator
from src.query import SpatialQueryEngine, Query
import src.scenarios as sx
import random


class LookaheadRuntime(object):
    def __init__(self, satellites, groundstations, query_engine, current_time) -> None:
        self.satellites = satellites
        self.groundstations = groundstations
        self.qe = query_engine
        self.current_time = current_time
        self.lookahead_results = None
        self.sim : LookaheadSimulator = None
    
    def refresh_results(self):
        temp = self.lookahead_results is not None
        self.lookahead_results = None
        refresh_until : Time = self.current_time.copy()
        refresh_until.add_seconds(60 * 60 * 6)
        sim = LookaheadSimulator(60, self.current_time, refresh_until, self.satellites, self.groundstations, engine=self.qe)
        sim.run()
        self.lookahead_results = sim.transmission_log
        self.sim = sim

        # save the results to a file
        f = "log/results.pkl"

        if temp:
            try:
                with open(f, "wb") as pk:
                    pickle.dump(self.lookahead_results, pk)
            except Exception:
                pass

    def extend_results(self, new_time):
        if self.lookahead_results is None:
            self.refresh_results()

        # check if the new time is greater than the current time
        if new_time > self.current_time:
            self.sim.endTime = new_time
            self.sim.run()
            self.lookahead_results = self.sim.transmission_log

    def prune_past_results(self, cutoff_time, satellites=None):
        if not satellites:
            satellites = self.satellites

        for sat in satellites:
            bad_idx = []
            for i in range(len(self.lookahead_results[sat.id])):
                if self.lookahead_results[sat.id][i][1] < cutoff_time:
                    bad_idx.append(i)
                else:
                    break

            for i in sorted(bad_idx,reverse=True):
                self.lookahead_results[sat.id].pop(i)
            
    def get_results(self, sat, time):
        f = "log/results.pkl"

        try:
            with open(f, "rb") as pk:
                self.lookahead_results = pickle.load(pk)
        except Exception:
            pass


        if self.lookahead_results is None:
            self.refresh_results()

        # pickle the results
        

        buffer_time = time.copy()
        buffer_time.add_seconds(60 * 20) # 20 mins

        max_extend_time = time.copy()
        max_extend_time.add_seconds(60 * 60 * 6) # 6 hours
        while ((sat.id not in self.lookahead_results  \
                or len(self.lookahead_results[sat.id]) == 0 \
                or self.lookahead_results[sat.id][-1][0] <= buffer_time) 
                and self.sim.endTime < max_extend_time):
            
            # 
            self.sim.endTime.add_seconds(60 * 60)
            self.extend_results(self.sim.endTime)

            print("Extending results for satellite {} at time {}".format(sat.node.id, time))

        for t, sched in self.lookahead_results[sat.id]:
            if t > buffer_time:
                return t, sched

        print(self.lookahead_results)
        raise Exception("No schedule found for satellite at time {}".format(time))
            

class EarthSightScheduler(object):
    def __init__(self, queries, satellites, stations, sim_time, serval_mode = False) -> None:
        self.queries = queries
        self.satellites = satellites
        self.stations = stations
        self.qe = SpatialQueryEngine()
        self.qe.load_queries(queries)
        self.sim_time = sim_time
        self.runtime = None
        self.serval_mode = serval_mode
        

    def schedule(self, sat : Satellite, start, length) -> Schedule:
        """
        Schedule queries to be executed by satellites.
        """

        endTime : Time = start.copy()
        endTime.add_seconds(length)
        schedule = Schedule(tasklist=[], startTime=start, endTime=endTime) # length in seconds
        length_in_minutes = length // 60
        image_count = 45*length_in_minutes
        current_time : Time = start

        if self.serval_mode:
            pri = 2
        else:
            if not self.runtime:
                self.runtime = LookaheadRuntime(self.satellites, self.stations, self.qe, current_time)

            
            time, sched = self.runtime.get_results(sat, current_time)
            pri = 1
            while sched[pri] == 0 and pri <= 10:
                pri += 1

            # pri is the lowest priority such that an image of priority pri was received
       
        fname = "log/{}_{}_schedule.pkl".format(sat.node.id, str(start).replace(":", "-"))
        # fname = "schedule_cache/{}_{}_schedule.pkl".format(sat.node.id, str(start).replace(":", "-"))

        try:
            with open(fname, "rb") as pk:
                schedule.tasklist = pickle.load(pk)
            return schedule
        except Exception:
            pass  

        for i in range(image_count):
            # calculate satellite position
            future_coord = sat.calculate_orbit(current_time).to_coords()
            queries : list[Query] = self.qe.get_queries_at_coord(future_coord, min_pri=pri, max_pri=10)
            if random.random() < 0.2 and not queries:
                queries.add(sx.create_combined_monitoring_query())

            formula = [([(f, True) for f in f_seq], q.priority_tier) for q in queries for f_seq in q.filter_categories]
            
            schedule_item = ScheduleItem(items=[formula])
            schedule.add_task(schedule_item)
            current_time.add_seconds(1.25)

        # save schedule to file
        with open(fname, "wb") as f:
            pickle.dump(schedule.tasklist, f)
        return schedule
