from typing import TYPE_CHECKING
from itertools import chain
import random # type: ignore
from time import time as timeNow
import numpy as np

from src.links import Link
from src.log import Log
from src.utils import Print
import const

if TYPE_CHECKING:
    from src.satellite import Satellite
    from src.station import Station
    from src.node import Node
    from src.packet import Packet
    from typing import List, Dict, Optional
    from src.topology import Topology

class CurrentTransmission:
    def __init__(self, sending: 'Node', receivingNodes: 'List[Node]', channel: 'int') -> None:
        self.sending = sending
        self.receivingNodes = receivingNodes
        self.receivingChannel = channel

        self.packets: 'List[Packet]' = []
        self.packetsTime: 'List[tuple[float]]' = [] #List of startTimes and endTimes for each packet relevative to the start of the timestep
        self.PER: 'Dict[Node, float]' = {} #the PER for each node. Should be set to 1 if the node isn't scheduled to receive the packet
        self.SNR: 'Dict[Node, float]' = {}
        
class Transmission:
    """
    This class is what sends the data between nodes
    """
    def __init__(self, links: 'Dict[Node, Dict[Node, Link]]', topology: 'Topology', satList: 'List[Satellite]', gsList: 'List[Station]', timeStep: 'int', uplink = False) -> None:
        self.links = links ##links should be a case of Dict[Satellite][Station] = links
        self.linkList = list( chain( *(list(d.values()) for d in self.links.values()) ))
        self.nodes = [i for i in chain(satList, gsList)]
        self.topology = topology
        self.timeStep = timeStep
        self.satList = satList
        self.gsList = gsList
        self.uplink = uplink

        transmissions = self.get_new_transmissions()
        self.transmit(transmissions)
        
    def transmit(self, transmissions: 'List[CurrentTransmission]'):
        #so here's how this works 
        #we have each device which has been scheduled from x time to y time
        #so let's do this, for each reception device's channel, we store a list of each packet's (startTime, endTime)
        #we then find any collisions in the startTime and endTime

        #receiving is a dict[node][channel] = List[ (packet, (startTime, endTime), PER, SNR) ]
        
        receiving = {}
        #s = timeNow()
        for transmission in transmissions:
            for node in transmission.receivingNodes:
                for i in range(len(transmission.packets)):
                    lst = receiving.setdefault(node, {})
                    chanList = lst.setdefault(transmission.receivingChannel, [])
                    chanList.append((transmission.packets[i], transmission.packetsTime[i], transmission.PER[node], transmission.SNR[node], str(transmission.sending), str(transmission.packets[i])))
                    #receiving[node][transmission.receivingChannel].append((transmission.packets[i], transmission.packetsTime[i], transmission.PER[node], str(transmission.sending), str(transmission.packets[i])))
        
        #print("Time to create receiving dict", timeNow() - s)
        
        #print receiving but call the repr function for each object
        #now let's go through each receiving and find any overlapping times 
        #t = timeNow()
        for receiver in receiving.keys():
            for channel, blocks in receiving[receiver].items():
                if len(blocks) == 0:
                    continue

                for block in blocks:
                    packet = block[0]
                    PER = block[2]
                    
                    #let's check if this packet gets dropped by PER
                    
                    if random.random() <= PER:
                        #print("Packet dropped", packet, receiver)
                        pass
                        #Log("Packet dropped", packet)
                    else:
                        #print("Packet received", packet, receiver)
                        time = block[1][1] - block[1][0]
                        if receiver.has_power_to_receive(time):
                            receiver.use_receive_power(time)
                            receiver.receive_packet(packet)

    def get_new_transmissions(self) -> 'Dict[int, List]':
        devicesTransmitting = {}
        currentTransmissions = []
        
        for link in self.linkList:
            #print("Link has {} start times".format(len(link.startTimes)), *link.nodeSending)
            for idx in range(len(link.startTimes)):
                sending = link.nodeSending[idx]
                startTime = link.startTimes[idx]
                channel = link.channels[idx]
                endTime = min(link.endTimes[idx], self.timeStep)
                
                #let's do this to avoid duplicate sending - maybe think of a better way to handle this??
                if sending in devicesTransmitting:
                    #check if this is from the same  or another, if its in another - raise an exception
                    if link is devicesTransmitting[sending]:
                        pass
                    else:
                        raise Exception("{} is transmitting on two links at the same time".format(sending))
                devicesTransmitting[sending] = link
                
                receiving = []
                datarate = 0
                if sending.beamForming:
                    receiving = [link.get_other_object(sending)]
                    per = {receiving[0]: link.PER}
                    snr = {receiving[0]: link.snr}
                    datarate = link.get_relevant_datarate(sending)
                else:
                    listOfLinks = self.topology.nodeUpLinks[sending] if self.uplink else self.topology.nodeDownLinks[sending]
                    receiving = [i.get_other_object(sending) for i in listOfLinks]
                    per = {i.get_other_object(sending): i.PER for i in listOfLinks}
                    snr = {i.get_other_object(sending): i.snr for i in listOfLinks}
                    receiving = [i for i in receiving if i.receiveAble]
                    
                    #lst = [i.get_relevant_datarate(sending) for i in listOfLinks if i.get_other_object(sending).receiveAble]
                    #datarate = min(lst)
                    datarate = link.get_relevant_datarate(sending)
                    for i in listOfLinks:
                        if i.get_relevant_datarate(sending) < datarate or not i.is_listening():
                            per[i.get_other_object(sending)] = 1
                    #Log("Sending", sending, "receiving", *receiving, "channel", channel, "datarate", datarate, "PER", per, "SNR", snr, "totalPackets")
                
                trns = CurrentTransmission(sending, receiving, channel)
                # print(sending)
                #now let's assign the packets within this transmission
                currentTime = startTime
                # print(currentTime, endTime, len(sending.transmitPacketQueue))
                tp = 0
                while currentTime < endTime and len(sending.transmitPacketQueue) > 0:
                    timeForNext = 0.1
                    if currentTime + timeForNext <= endTime and sending.has_power_to_transmit(timeForNext):

                        tp += 1
                        sending.use_transmit_power(timeForNext)
                        pck = sending.send_data()
                        trns.packets.append(pck)
                        trns.packetsTime.append((currentTime, currentTime + timeForNext))
                        currentTime = currentTime + timeForNext

                        if self.uplink:
                            break # only transmit one packet because it has one destination satellite - don't send packets meant for sat1 and sat2 to sat1.
                    else:
                        if currentTime + timeForNext > endTime:
                            pass
                            # print("Time for next packet exceeds end time, breaking")
                        else:
                            print("Not enough power to transmit, breaking")
                        break
                # if tp > 0:
                #     print("transmitted", tp)
                # Log("Sending", sending, "receiving", *receiving, "channel", channel, "datarate", datarate, "PER", per, "SNR", snr, "totalPackets", len(trns.packets))
                assert len(trns.packets) == len(trns.packetsTime)
                trns.PER = per
                trns.SNR = snr
                currentTransmissions.append(trns)
                        
        return currentTransmissions
        
        
