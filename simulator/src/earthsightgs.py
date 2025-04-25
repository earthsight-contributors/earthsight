from src.image import Image
from src.receiveGS import ReceiveGS
from src.packet import Packet
from src.log import get_logging_time
from src.utils import Time
from src.node import Node
from . import log


class EarthSightGroundStation(ReceiveGS):

    rcv_data = {}
    delays = {i:[] for i in range(-1, 11)}
    
    def __init__(self, node: Node, scheduler) -> None:
        """
        Decorator object for a node object, normally used on a station object.
        It will make it so that the node object can only receive, and not transmit.
        """
        super().__init__(node)
        self.upload_bandwidth = 1000000
        self.scheduler = scheduler

    def get_upload_bandwidth(self) -> int:
        """
        Returns the upload bandwidth
        """
        return self.upload_bandwidth


    def has_data_to_transmit(self) -> bool:
        return len(self.transmitPacketQueue) > 0
    
    ### Add a method that builds a schedule and sends it to the satellite
    def make_schedule(self, satellite, time_start, length) -> None:
        """
        Builds a schedule and sends it to the satellite
        """
        return self.scheduler.schedule(satellite, time_start, length)

    def receive_packet(self, pck: 'Packet') -> None:
        if type(pck.descriptor) != str:
            print(pck, "Unknown packet type")
            print(pck.descriptor)
            log.Log("Unknown packet type", pck, "Unknown packet type")
            raise ValueError("Unknown packet type")
        elif pck.descriptor.startswith("schedule request"):
            schedule = self.make_schedule(satellite=pck.relevantNode, time_start=pck.relevantData[0], length=60*60*6)
            print("Schedule fullness: ", schedule.percentage_requiring_compute())
            pkt = Packet(relevantData=schedule, relevantNode=pck.relevantNode, descriptor="schedule") # relevant node is the target node
            self.transmitPacketQueue.appendleft(pkt)
        elif pck.descriptor == "image":
            sat = pck.relevantNode
            if sat not in EarthSightGroundStation.rcv_data:
                EarthSightGroundStation.rcv_data[sat] = {
                    **{i: 0 for i in range(-1, 11)},
                    **{'count_{}'.format(i): 0 for i in range(-1, 11)},
                    **{'unavoidable_delay_{}'.format(i): 0 for i in range(-1, 11)}
                }

            if type(pck.relevantData) != Image:
                image = pck.relevantData[0]
            else:
                image : Image = pck.relevantData

            collection_time = image.time
            current_time = get_logging_time()
            delay = Time.difference_in_seconds(current_time, collection_time)
            unavoidable_delay = Time.difference_in_seconds(current_time, image.earliest_possible_transmit_time)


            ground_truth = image.descriptor            
            EarthSightGroundStation.rcv_data[sat][ground_truth] += delay
            EarthSightGroundStation.rcv_data[sat]['unavoidable_delay_' + str(ground_truth)] += unavoidable_delay
            EarthSightGroundStation.rcv_data[sat]['count_' + str(ground_truth)] += 1
            EarthSightGroundStation.delays[ground_truth].append(delay)
        else:
            print(pck, "Unknown packet type")
            super().receive_packet(pck)


        

    