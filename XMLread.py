import xml.etree.ElementTree as ET

# SELECT LANES FOR TRAFFIC INTERSECTION
node1_north_ids = ["315703119#0_0", "315703119#0_1", "315703119#0_2"]
node1_south_ids = ["1005107471#0_0", "1005107471#0_1"]
# node1_east_ids = 
# node1_west_ids = 
queueing_length_total = {}
queueing_time_total = {}

tree = ET.parse('Data\mainofframp-queue.xml')
root = tree.getroot()

## GET INDIVIDUAL QUEUE ATTRIBUTRES FROM SELECTED LANES AS WELL AS TOTAL QUEUE LENGTH AND TIME AT EACH TIMESTEP
# Initialize queueing lengths for each timestep to an empty list
for data in root.findall('data'):
    timestep = data.get('timestep')
    queueing_length_total[timestep] = {}
    queueing_time_total[timestep] = {}

# CONCATENATING LANES FOR NS TRAFFIC FOR NODE 1
node1NS_ids = node1_north_ids + node1_south_ids

# print(node1_ids)

# Calculate queueing lengths and times for North and South lanes
for data in root.findall('data'):
    timestep = data.get('timestep')
    for lane in data.findall('./lanes/lane'):
        lane_id = lane.get('id')
        # if lane_id in node1_north_ids:
        if lane_id in node1NS_ids:                                            ## USING MULTIPLE DIRECTIONS OF LANES
            queueing_length = float(lane.get('queueing_length'))
            queueing_time = float(lane.get('queueing_time'))
            if lane_id not in queueing_length_total[timestep]:
                queueing_length_total[timestep][lane_id] = []
            if lane_id not in queueing_time_total[timestep]:
                queueing_time_total[timestep][lane_id] = []
            queueing_length_total[timestep][lane_id].append(queueing_length)
            queueing_time_total[timestep][lane_id].append(queueing_time)

# Print individual queue length and time for each lane, and combined queue length and time for the node
for timestep, queue_lengths in queueing_length_total.items():
    print(f"Timestep: {timestep}")
    total_length = 0
    total_time = 0
    for lane_id in node1NS_ids:
        lengths = queueing_length_total[timestep].get(lane_id, [0])
        times = queueing_time_total[timestep].get(lane_id, [0])
        print(f"Lane ID: {lane_id}, Queue Lengths: {lengths}, Queue Times: {times}")
        total_length += sum(lengths)
        total_time += sum(times)
    print(f"Total Queueing Length for Timestep {timestep}: {total_length}m")
    print(f"Total Queueing Time for Timestep {timestep}: {total_time}s \n")
