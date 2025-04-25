# Code Guide

This document will walk you through all of the code in the simulator, describing all of the components and important things to pay attention to.

## TLDR

Earthsight uses run.py

Implement your simulation into the system, using run.py, and create your own version of satellite or ground station classes.

## Filesystem

In the outer directory, there are three main files:

- main.py - this is the main runner which the simulation is run from
- const.py - this is where configuration variables are set
- sim.yml - this is a conda config file. You should may the environment off of this, but we also provide a full requirements.txt that you may use also.
- src/:
  - data.py - this is an object that represents an object that is being collected and then transmitted such as an image or a file
  - EarthsightSatellite.py - this is an example of a satellite that can take images and then transmit it to the ground.
  - EarthsightGroundsation.py - this is an example of a ground station which can only transmit data that it collects to a satellite but can't receive any data. This object randomly creates roughly 30 bits of data every hour.

## Main functionality walkthrough

We will start in example.py and then work our way through all of the functionality in the simulator

After the imports in example.py, it takes in the stations from the tinygs stations.json that they report and then converts that into a list of ground stations.

    groundStations: 'List[Station]' = []

This is the same list that will be used throughout the simulator. While there are many instances where the simulator defines a list of stations, they should all be references to this original object. The code continues on by looping through all of the lines in the json file and creates a station object for each one:

    s = Station(row["name"], id, Location().from_lat_long(row["location"][0], row["location"][1]))

This line here will create a station object with the name, id, and location given in the file. The constructor for the station object is:

    Station(name: str, id: int, loc: Location, transmitAble: bool = True, receiveAble: bool = True, packetBuffer: int = 2147483646, maxMemory: int = 2147483646)

The location should be a Location object which can be found in utils.py. This type standardizes the location of all of the objects across the simulator. It stores all of the locations in ITRF form. If you want more information on this frame, take a look here: https://www.iers.org/IERS/EN/DataProducts/ITRS/itrs.html#:~:text=The%20ITRS%20is%20an%20ideal,GPS%2C%20SLR%2C%20and%20DORIS. _All distance units stored in the simulator are in meters_. The other attributes, transmitAble and receiveAble are booleans describing wether or not the ground station has the functionality to both transmit and receive, respectively. The packetBuffer and maxMemory variables describes the size in bits of the size of the object's packetBuffer and the total memory capacity of the object. _Everything in the simulator is done in bits, so be careful of your units_

The name, id, location, packetBuffer, and maxMemory get passed into the node.py constructor which is in the form of:

    Node(name: str, id: int, pos: Location, packetBuffer: int = 2147483647, maxMemory: int = 2147483646)

Within the node.py constructor, in addition to the variables passed in through the constructor, there are other variables assigned, which are normally set by the decorator objects:

- nChannels - the number of receiving channels this object has
- sendAcks - whether this device sends acks when a packet gets received
- waitForAck - whether this device should delete the packet once it's sent or wait for an ack to be received before deleting

Additionally, there are multiple features to deal with simulating power capabilities in the simulator. All of these variables are stored in milliwatts for energy consumption and milliwatt seconds/millijoules for battery capacity. The usage of these variables is determined by const.INCLUDE_POWER_CALCULATIONS. All of these variables are by default 0, indicating that it takes no energy to do anything.

- maxMWs - max battery capacity in milliwatt seconds
- currentMWs - current battery capacity in milliwatt seconds
- normalPowerConsumption - normal power consumption in milliwatts. This is stuff like heaters, gps, normal compute, etc.
- transmitPowerConsumption - power consumption in milliwatts when transmitting
- receivePowerConsumption - power consumption in milliwatts when receiving
- powerGeneration - power generation in milliwatt seconds

Continuing back to example.py, you can see we randomly assign 75% of the stations to IotDevices and ReceiveGS. Both of these classes extend the NodeDecorator class which follows a decorator design pattern on node objects. Both of these classes should at least implement the following 3 methods, which each new user-created device should either overload or take from another decorator. The methods are:

1. load*data(self, timeStep) - \_timeStep and all time units are either floats in seconds or utils.Time objects*
2. load_packet_buffer(self)
3. receive_packet(self, pck)

If these methods are not overloaded, the default method in node.py will be called (which does give a few too many angry messages). Additionally, more methods and default variables can be overloaded, i.e. powerGeneration in iotSatellite.py. I will elaborate on these methods in more detail later.

Continuing on, after we create the list of stations, we load up the tle files in the method Satellite.load_from_tle which takes in a 3 line TLE file. You can add additional keyword arguments to this method which will be passed to the satellite constructor:

    Satellite(name: str, id: int, tle: str = "", beamForming: bool = False, packetBuffer: int = 2147483646, maxMemory: int = 2147483646)

For the exeption of tle and beamForming, these arguments are the same as I described before and will be passed to the node constructor. beamForming describes wether a satellite can focus and only transmit to one gs at a time (true) or wether it will transmit to all gs in its footprint (false). The argument tle is a string which can either not be passed or can contain a tle of 2 or 3 lines.

If the tle is not passed on, the satellite's position can be updated using the static create_constellation method, which given a list of satellites can be turned into a constellation. This method should only work for relatively circular constellations (i.e. LEO & GEO), but will not work for other orbits such as sun-syncronous oribts and also Pi constellations such as Iridium (take a look in the comments for more info.):

    create_constellation(listOfSats: 'List[Satellite]', numberOfPlanes: int, numberOfSatellitesPerPlane: int, inclination: float, altitude: float, referenceTime: 'Time'

One hypothetical example of this method would be if a user wanted to simulate a full satellite constellation that hasn't been launched yet. For example, SpaceX's starlink constellation is planned to be built of 5 shells, with the first one having 72 different oribital planes, with each plane having 22 satellites, an elevation of 550 km, and a inclination of 53 degrees. Accordingly, the user can create a list of satellite objects and then call the method which will evenly spread out all of the satellites along their planes at the given start time. Here is an example:

    ##note: this is hypothetical untested code.
    starlink = []
    startTime = Time().from_str("2022-06-15 12:00:00")  ##this is a utils.Time object, more info is below
    for i in range(0, 1584):
        starlink.append( Satellite(str(i), i) )
    Satellite.create_constellation(starlink, 72, 22, 53, 550*1000, startTime)

Back to example.py, our example code decorates all the satellite object with the IotSatellite decorator which allows the satellite to receive an incoming packet, hold that packet, and then eventually transmit it.

    satellites = [IoTSatellite(i) for  i  in  satellites]

After we populated the list of satellites, we start to setup the necessary variables for the simulation. we create the startTime and endTime of the simulation:

    startTime = Time().from_str("2022-07-15 12:00:00")
    endTime = Time().from_str("2022-07-15 13:00:00")

These are both utils.Time objects. This class is a wrapper from the datetime.datetime object to make things cleaner throughout the simulation. Some things to note about this object:

    #if you try
    time1 = startTime
    #time1 and startTime are the same object, and any changes will happen to both of them
    #to have time1 be a new object of the time of startTime, run
    time1 = startTime.copy()
    ##to add or subtract seconds to the time object, try
    sec = 10
    time1.add_seconds(sec) ##adds 10 seconds and updates the time1 object
    time1.add_seconds(-sec) ##subtracts 10 seconds and updates the time1 object

After this information is setup, we create the simulator object, the simulator has the constructor of:

    Simulator(timeStep: float, startTime: Time, endTime: Time, satList: 'List[Satellite]', gsList: 'List[Station]', recreated: bool = False)

As described earlier, the timeStep is in seconds, the startTime and endTime are Time objects. satList and gsList are lists of satellites and ground stations (they should just be references to the original objects). You can probably ignore the recreated variable - I will describe this later when I talk about recreated.py.

In the simulator, there is a method called calculate_topologys which calculates all of the topology objects for the timeframes. This can be saved using the save_topology method and the load_topology method, in order to run the simulation multiple times without have to recompute these. Within these methods, the topology consturctor is called:

    Topology(time: Time, satList: 'List[Satellite]', groundList: 'List[Station]')

When you create this object, it will create two important attributes: availabilityMap which is in the form of Dict[Satellite][Station] = bool. The bool describes wether or not the objects can see each other. This is calculated using a series of equations to find the angle in the sky a satellite is from a gs. The minimum angle for a gs to see a satellite is defined in const.py's MINIMUM_VISIBLE_ANGLE.

The other variable is called possibleLinks where Dict[Satellite][Station] = Link. This method will take the availability map and calculate the expected datarate if the two would transmit to each other. For this variable, if you try to access a satellite and a station that can't see each other, you should get a KeyError.

These link objects are created using the create*link method from links.py. This method was based on the SNR model of TinyGS satellites and \_it should be changed based on the satellite constellation you are trying to simulate.*

Back in simulator.py, each one of these topology objects will be stored into a dictionary of time to topology maps. After the constructor is run, the user can run simulator.run(), which will run the actual simulation. At each timestep, the simulator runs each object's load_data method. Per the comments in node.py, the load_data method should determine the rate to generate data and add it to the dataQueue. Pay attention that when you create a data object, the size is treated as an integer, so if a timestep is lower than the time to generate a data, you will either need to keep track of how much is created in each timestep or find some way to split the data generation. An example of these two seperate methods can be found in both imageSatellite.py and iotDevice.py.

Similarly, after the data is generated, the simulator will call the object's load_packet_buffer method which will convert the dataQueue over to packets. Examples of these methods being implemented can be found in iotDevice.py, imageSatellite.py, receiveGS.py, and iotSatellite.py. Also, take a look in the node.py comments for this method.

At each timestep, the simulator will create a new routing object, which will decide which devices should transmit to which. The constructor simply takes a topology object:

    Routing(top: 'Topology')

This object will have a variable called bestLinks which is a Dict[Satellite][Station] = Link where accessing any undefined links will cause a KeyError. This attribute is created by the routing class's schedule_best_links method. This method is dependent on const.py's RoutingMechanism variable. As of now, we have implemented 4 different scheduling algorithms:

- assign_by_datarate_and_available_memory - This is a greedy algorithm which will prioritize which will prioritize objects which have the strongest datalink and the most memory to transmit
- transmit_with_random_delays- This is an algorithm which will have each gs transmit a packet and waits a random amount of time in the beacon interval before sending another
- use_all_links - This algorithm will have every satellite transmit with every gs. This isn't realistic or feasible at all, but sometimes useful for testing or some other purpose.
- transmission_probability_function - implements the TPF equation given in: https://ieeexplore.ieee.org/document/9685152/

Within each of these routing mechanisms, each object is told a time period to transmit based on:

    link.assign_transmission(self, startTime: 'int', duration: 'float', channel: int, node: 'Node')

where startTime is the offset in the beacon interval (i.e. if the main timeStep is 60 seconds and the startTime is 30 seconds, it will start transmitting 30 seconds in), duration is the length of time it should transmit for, channel is the receiving channel of the receiving device, and node is the device that is transmitting. The startTime and duration are accurate to the const.MICRO_TIMESTEP variable. So for instance, if the duration is actually 4.321 seconds long and the MICRO_TIMESTEP is 1 second, the duration will be treated as 5 seconds.

This method call will be passed on to transmission.py where all of the sending of packets happens. Once a packet is received, the receiving object's receive_packet method (this is one of the three I talked about eariler) will be run with the new packet. Take a look in any of the decorator classes for examples on this.

Once the objects transmit all of the information, they will reload their data, and then repeat for the timestep. Once the simulation is over, you can take a look at the log, which you can see processed in the loggingCode/ folder. Take a look in that folders README for more info.

## Recreate.py

Recreate.py highlights some additional functionality that we added to make running the simulator more easy to use. There are two main cases that this logic can be used for:

1.  When we are running a large simulation and want to save memory.
2.  When a user wants to use a larger timeStep for a while, then a more fine-grained simulation

In the code, what you can see is that at each interval, we are using the simulator classes method:

    save_objects(self, fileName: str)

This method will pickle all of the simulator objects and the logging information so the simulator can be rebuilt later in the file.

Then, the simulator can be loaded from:

    open_stored_simulator(fileName: str, timeStepNew: float = None, startTimeNew: Time = None, endTimeNew: Time = None)

This will load the simulator stored in the file. If you choose to leave any of these variables as none, it will use the original variables stored in the previous simulation and return a new Simulator object. This will be the exact same simulator object as before but instead, it will turn self.recreated on.
