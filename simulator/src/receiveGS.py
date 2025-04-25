from typing import TYPE_CHECKING
from src.station import Station
from src.utils import Location
from src.packet import Packet
from src.log import Log
from src.nodeDecorator import NodeDecorator
import const

if TYPE_CHECKING:
    from src.node import Node

class ReceiveGS(NodeDecorator):
    """
    Class that models station that only receives data and doesn't transmit anything
    """
    def __init__(self, node: 'Node') -> None:
        """
        Decorator object for a node object, normally used on a station object.
        It will make it so that the node object can only receive, and not transmit.
        """
        super().__init__(node)
        ##self._node is a station object, so set transmit
        self.transmitAble = False
        self.receiveAble = True
        self.waitForAck = False
        self.sendAcks = False
        self.groundTransmitAble = True
        self.groundReceiveAble = True

    #@profile
    def receive_packet(self, pck: 'Packet') -> None:
        """
        Code to receive packet and add it to packet buffer

        Arguments:
            pck (Packet) - packet received
        """
        if "ack" in pck.descriptor:
            self.receive_ack(pck)
        elif self.sendAcks:
            ##this assumes that once this get's here, the packet is done
            #self.receivePacketQueue.appendleft(pck)
            self.generate_ack(pck)
        Log("Iot Received packet:", pck, self)
        if const.INCLUDE_UNIVERSAL_DATA_CENTER:
            self.receivePacketQueue.appendleft(pck)
    def load_data(self, timeStep: float) -> None:
        """
        It only receives data, so don't load or do anything
        """
        if const.INCLUDE_UNIVERSAL_DATA_CENTER:
            self.convert_receive_buffer_to_data_objects()
        ##Process data objects

    def load_packet_buffer(self) -> None:
        """
        Doesn't have any data to transmit, so pass
        """
        if const.INCLUDE_UNIVERSAL_DATA_CENTER:
            self.convert_data_objects_to_transmit_buffer()

    def get_upload_bandwidth(self) -> None:
        return 5000
