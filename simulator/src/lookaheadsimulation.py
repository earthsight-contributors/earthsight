import copy
from typing import Dict, List, Optional, no_type_check, TYPE_CHECKING
from time import time as time_now
from typing import Dict, List
from src.utils import Time, Print, PriorityQueueWrapper
from src.routing import Routing
from src.satellite import Satellite
from src.station import Station
from src.data import Data
from src.packet import PriorityPacket
from src.topology import Topology
from src.node import Node
from src.transmission import Transmission
from src import log
import concurrent.futures
from src.nodeDecorator import NodeDecorator
from src.query import SpatialQueryEngine
from src.formula import overall_confidence_dnf
from src.receiveGS import ReceiveGS
from src.utils import Time
from src.earthsightsatellite import EarthsightSatellite
import pickle
    
class LookaheadSimulator:
    """
    Main class that runs simulator

    Attributes:
        timestep (float/seconds) - timestep in seconds. CURRENTLY MUST BE AN INTEGER NUMBER
        startTime (Time) - Time object to when to start
        endTime (Time) - Time object for when to end (simulation will end on this timestep. i.e if end time is 12:00 and timeStep is 10, last run will be 11:59:50)
        satList (List[Satellite]) - List of Satellite objects
        gsList (List[Station]) - List of Station objects
        topologys (Dict[str, Topology]) - dictionary of time strings to Topology objects
        recreated (bool) - wether or not this simulation was recreated from saved file. If true, then don't compute for t-1 timestep
    """
    sim_id = 0
    def __init__(self, timeStep: float, startTime: Time, endTime: Time, satList: 'List[Satellite]', gsList: 'List[Station]', engine : SpatialQueryEngine = None) -> None:
        self.timeStep = timeStep
        self.startTime = startTime
        self.endTime = endTime
        self.time = self.startTime.copy()

        self.id = LookaheadSimulator.sim_id
        LookaheadSimulator.sim_id += 1

        self.transmission_log = {sat.id: [] for sat in satList}
        self.satList = [LookaheadSatellite(sat, engine=engine, lookahead_time=self.time, sim_id=self.sim_id) for sat in satList]
        self.gsList = [LookaheadGS(gs, self.transmission_log, self.time) for gs in gsList]

        
    @staticmethod
    def parallel_sat_loads(sat, timestep):
        sat.load_data(timestep)
        sat.load_packet_buffer()

    @staticmethod
    def parallel_gs_loads(gs, timestep):
        gs.load_data(timestep)
        gs.load_packet_buffer()

    @staticmethod
    def parallel_propogation(sat, time):
        sat.update_orbit(time)


    def run(self) -> None:
        """
        At inital, load one time step of data into object
        then schedule based off of new data
        send info
        """

        while self.time < self.endTime:
            s = time_now()
            print("Looking ahead to:", self.time.to_str())
            

            # for now, single threaded. easily parallelzable with threadpoolexecutor
            for sat in self.satList:
                LookaheadSimulator.parallel_propogation(sat, self.time)

            for sat in self.satList:
                LookaheadSimulator.parallel_sat_loads(sat, self.timeStep)

            for gs in self.gsList:
                LookaheadSimulator.parallel_gs_loads(gs, self.timeStep)

            topology = Topology(self.time, self.satList, self.gsList)
            routing = Routing(topology, self.timeStep, lookahead=True)
            Transmission(routing.bestDownLinks, topology, self.satList, self.gsList, self.timeStep, uplink=False)

            self.time.add_seconds(self.timeStep)
            print("Timestep took", time_now() - s)
            

        for k, v in self.transmission_log.items():
                print(k, v)

        log.close_logging_file()

class LookaheadSatellite(NodeDecorator):
    def __init__(
        self,
        original_sat: EarthsightSatellite,
        engine: SpatialQueryEngine,
        lookahead_time: Time,
        sim_id: int = 0
    ) -> None:
        fresh_node =  Satellite(name="Lookahead " + original_sat.name, id=str(original_sat.get_id()) + "_" + str(sim_id), tle=original_sat.tle)
        super().__init__(fresh_node)

        # attributes for lookahead satellite
        self.node = fresh_node
        self.transmitPacketQueue = PriorityQueueWrapper()
        self.engine: SpatialQueryEngine = engine
        self.lookahead_time = lookahead_time
        self.image_size = 1000
        self.currentMWs = float('inf') # no power limits for lookahead
        self.priority_counts = {i : 0 for i in range(1, 11)}

        for pkt in original_sat.transmitPacketQueue.queue:
            if 1 <= pkt.priority <= 10:
                self.priority_counts[pkt.priority] += self.image_size

    def populate_cache(self, time) -> None:
        """
        Populates the cache with images
        """
        queries = self.engine.get_queries_at_coord(self.node.position.to_coords())
        # seaparate queries by priority into a dict
        query_dict = {i : [] for i in range(1, 11)}
        for q in queries:
            query_dict[q.priority_tier].append(q)
        
        all_fail_prob = 1
        for pri, quer in query_dict.items():
            formula = [[(f, True) for f in f_seq] for q in quer for f_seq in q.filter_categories]
            # filters_used = [f for f_seq in formula for f in f_seq]
            prob = overall_confidence_dnf(formula, [])
            self.priority_counts[pri] += prob * self.image_size
            all_fail_prob -= prob

        self.priority_counts[1] += all_fail_prob * self.image_size
 
    def load_packet_buffer(self) -> None:
        """
        Loads the packet buffer with data
        """
        pri = 10
        while len(self.transmitPacketQueue) < 600 and any(self.priority_counts.values()):
            while self.priority_counts[pri] > 0 and pri > 0:
                self.transmitPacketQueue.appendleft(PriorityPacket(priority=pri, infoSize = min(self.image_size, self.priority_counts[pri]), relevantNode=self.node))
                self.priority_counts[pri] = max(0, self.priority_counts[pri] - self.image_size)
                if self.priority_counts[pri] < 0:
                    self.priority_counts[pri] = 0
            pri -=  1

    def get_cache_size(self) -> int:
        return self.cache_size

    def percent_of_memory_filled(self) -> float:
        return len(self.dataQueue) / 10000000

    def load_data(self, timeStep: float) -> None:
        self.generate_power(timeStep)
        time_copy = self.lookahead_time.copy()
        img_r = timeStep / 45
        for _ in range(45):
            self.populate_cache(time_copy)
            time_copy.add_seconds(img_r)


class LookaheadGS(ReceiveGS):
    def __init__(self, node, transmission_log, time):
        super().__init__(node)
        self.transmission_log = transmission_log
        self.time = time

    def receive_packet(self, pck):
        # transmission log:
        # dict[Node : List[time, dict[priority : count]]]

        if not 1 <= pck.priority <= 10:
            return
        relevant_id = int(pck.relevantNode.id.split("_")[0])
        node_log = self.transmission_log[relevant_id]
        if node_log and -180 <= Time.difference_in_seconds(self.time, node_log[-1][0]) <= 180:
            node_log[-1][0] = self.time.copy()
            node_log[-1][1][pck.priority] += pck.infoSize
            
        else:
            node_log.append([self.time, {i: 0 for i in range(1, 11)}])
            node_log[-1][1][pck.priority] += pck.infoSize 
        