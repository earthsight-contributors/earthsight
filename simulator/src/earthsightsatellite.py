from datetime import datetime
from src.image import Image, evaluate_image
from . import node
from . import log
from .utils import FusedPriorityQueue, Time
from src.metrics import Metrics
from src.packet import PriorityPacket, Packet
from src.schedule import Schedule
from collections import deque
from queue import PriorityQueue
from src.nodeDecorator import NodeDecorator

class EarthsightSatellite(NodeDecorator):
    computation_times = []
    power_consumptions = 0
    power_generation = 0

    def __init__(
        self,
        node: "node.Node",
        start_time: datetime = datetime(2025, 1, 1),
        mode = "earthsight",
        mtl_registry = None
    ) -> None:

        super().__init__(node)
        self.mode = mode
        self.node = self.get_node()
        self.cache_size = 0
        self.computation_schedule = deque()
        self.computation_time_cache = 0

        self.mtl_registry = mtl_registry
        self.schedule_request = [None]
        self.computationQueue = deque()
        self.prioritizedQueue = PriorityQueue()
        self.deprioritizedQueue = PriorityQueue()

        self.computation_queue_sizes = []

        self.transmitPacketQueue : FusedPriorityQueue = FusedPriorityQueue(
            schedule_request=self.schedule_request,
            priority_queue=self.prioritizedQueue,
            compute_queue=self.computationQueue,
            low_priority_queue=self.deprioritizedQueue
        )

        self.normalPowerConsumption = 2.13 * 1000 # (ADACS) (units are in milliwatt hours)
        self.currentMWs = 100 * 1000
        self.compute_power = 10 * 1000 # 10 W
        self.receivePowerConsumption = 1 * 1000 # 1 W
        self.transmitPowerConsumption = 5 * 1000 # 5 W
        self.camera_power = 4.5 * 1000 # 4.5 W
        self.powerGeneration = 10 * 1000 #
        self.maxMWs = 4000000
        
        self.start_time = Time().from_datetime(start_time)
        self.image_rate = .75 # images per second
        self.analytics = []
        self.coords = self.node.position.to_coords()
        self.scheduled_until = None
        self.schedule_req_time = None

    def populate_cache(self, timeStep: float) -> int:
        """
        Populates the cache with images
        """
        coords = self.node.position.to_coords()
        images_captured = int(timeStep * self.image_rate)
        self.currentMWs -= self.camera_power * timeStep
        Metrics.metr().images_captured += images_captured
        collection_time = log.get_logging_time()

        log.Log("node", self, {"computation schedule length": len(self.computation_schedule), "images_captured": images_captured, "time": collection_time})


        for _ in range(images_captured):
            image = Image(10, time=collection_time, coord=coords, name="Target Image")


            if self.computation_schedule:
                Metrics.metr().hipri_captured += 1
                tasks = self.computation_schedule.popleft()
                image.score, image.compute_time, image.descriptor = evaluate_image(formula=tasks.items[0], mode=self.mode, registry=self.mtl_registry)
                if image.descriptor > 1:
                    Metrics.metr().hipri_sent += 1


            self.cache_size += image.size
            if image.compute_time > 0:
                EarthsightSatellite.computation_times.append(image.compute_time)
                Metrics.metr().hipri_computed += 1
                self.transmitPacketQueue.put_compute(Packet(relevantData=image, relevantNode=self.node, descriptor="image"))
                log.Log("node", self, {"image": image.id, "score": image.score, "compute_time": image.compute_time})
            else:
                image.descriptor = 0 if self.computation_schedule else -1
                self.transmitPacketQueue.put_low_priority(PriorityPacket(priority=image.descriptor + 2, relevantData=image, descriptor="image", relevantNode=self.node))
        
        return images_captured
            
    def get_cache_size(self) -> int:
        """
        Returns the size of the cache
        """
        return self.cache_size

    def do_computation(self) -> None:
        """
        Does the computation
        """
        while (len(self.computationQueue) > 0):
            if self.computation_time_cache <= 0:
                break
            
            pkt : Packet = self.computationQueue.popleft()
            image : Image = pkt.relevantData[0]
            self.computation_time_cache -= image.compute_time

            self.images_processed_per_timestep += 1
            
            if image.score > 0:
                self.transmitPacketQueue.put_priority(PriorityPacket(priority=image.score, relevantData=image, descriptor="image", relevantNode=self.node))
            else:
                self.transmitPacketQueue.put_low_priority(PriorityPacket(priority=5, relevantData=image, descriptor="image", relevantNode=self.node))

    def percent_of_memory_filled(self) -> float:
        return len(self.dataQueue) / 10000000

    def receive_packet(self, pck: Packet) -> None:
        """
        Receives a packet
        """
        schedule: Schedule = pck.relevantData[0]
        self.scheduled_until : Time = schedule.end
        self.computation_schedule.extend(schedule.toQueue())
        log.Log("Node {}: Scheduled with length {}".format(self.id, len(self.computation_schedule)))

    def should_request_schedule(self, timestep: float) -> bool:
        schedule_horizon = log.get_logging_time().copy()
        schedule_horizon.add_seconds(60*60*4) # 6 hours
        if self.scheduled_until and self.scheduled_until >= schedule_horizon:
            # schedule is valid for 6 hours already
            return False
        
        if self.transmitPacketQueue.has_schedule_request():
            # already requested a schedule
            self.transmitPacketQueue.schedule_request[0].relevantData = [log.get_logging_time()]
            return False
            
        
        if self.schedule_req_time and Time.difference_in_seconds(log.get_logging_time(), self.schedule_req_time) <= 60*10:
            # give gs 10 minutes to respond before resending
            return False

        return True
        

    def load_data(self, timeStep: float) -> None:
        """
        Loads data from the cache into the node
        """
        if self.should_request_schedule(timeStep):
            time = log.get_logging_time() if self.scheduled_until is None else self.scheduled_until
            self.transmitPacketQueue.put_schedule(PriorityPacket(priority=11, relevantData=time, descriptor="schedule request", relevantNode=self))
            self.schedule_req_time = log.get_logging_time()

        self.computation_queue_sizes.append(len(self.computationQueue))

        self.images_processed_per_timestep = 0
        self.time = log.loggingCurrentTime
        ims_captured = self.populate_cache(timeStep)
        EarthsightSatellite.power_generation += self.powerGeneration
        # Do computation
        if self.computation_time_cache < timeStep:
            if self.currentMWs > self.compute_power * timeStep / 4: # load in 15 second increments
                # timestep groups afforded by the power
                steps = min(int(self.currentMWs / (self.compute_power * timeStep)), 4)

                self.computation_time_cache += timeStep * steps / 4

                prevmw = self.currentMWs

                self.currentMWs -= self.compute_power * timeStep * steps / 4
                EarthsightSatellite.power_consumptions += self.compute_power * timeStep * steps / 4
                # print("Computing: ", prevmw, self.computation_time_cache, self.currentMWs, timeStep, self.compute_power)
            else:
                pass
                # print("Not enough power to compute: ", self.currentMWs, timeStep, self.compute_power)
        log.Log(
            "Computation time cache", self, {
                "time_cache": self.computation_time_cache}
        )
        self.do_computation()
        self.analytics.append([len(self.computation_schedule), len(self.dataQueue), len(self.computationQueue), len(self.transmitPacketQueue), self.percent_of_memory_filled(), self.computation_time_cache, self.currentMWs, self.images_processed_per_timestep, ims_captured])
