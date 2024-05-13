# DDQN Traffic Management System Instructions

# Install python version 3.9.13 or later:
https://www.python.org/downloads/

# Install SUMO:
https://sumo.dlr.de/docs/Installing/index.html

# Set SUMO_HOME and PATH variables:
https://www.youtube.com/watch?v=fIHOQNhvOu4&t=294s

The above video shows the installation process as well as setting the correct path and home variables.

# Install requirements:

pip install -r requirements.txt

# Creating network file from OSM
Go to https://www.openstreetmap.org/ and select the region you want to use.
Export as osm file, rename to whatever you want
In folder with osm map, click the folder path and type 'cmd". Press enter.
Enter the command: netconvert --osm-files osmFile.osm --output-file networkFileName.net.xml --geometry.remove --roundabouts.guess --ramps.guess --junctions.join --tls.guess-signals --tls.remove-simple --tls.join

# Creating a network file in Netedit
Open Netedit software by pressing windows key and typing 'Netedit'.
Create a new network file.
Build the network.
Save network file as networkFileName.net.xml

# Creating random trips:
Open the cmd window by clicking the folder path location of the net.xml file in the file explorer, and entering 'cmd' and pressing enter.
Enter the command: python "path-to-randomtrips.py" -n networkFileName.net.xml -e endTimeInSeconds -o networkFileName.trips.xml
A new .trips.xml file should be created in the folder.
Enter the command: duarouter -n networkFileName.net.xml --route-files networkFileName.trips.xml -o networkFileName.rou.xml --ignore-errors
A new .rou.xml file should be created in the folder.

# Creating config file:
In the same folder as your .net.xml file, create a new .txt document. Rename it to networkFileName.sumocfg
Paste the following configuration data:
<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on enter-date-here by Eclipse SUMO sumo Version 1.19.0
-->

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="networkFileName.net.xml" synonymes="n net" type="FILE" help="Load road network description from FILE"/>
        <route-files value="networkFileName.rou.xml" synonymes="r routes" type="FILE" help="Load routes descriptions from FILE(s)"/>
    </input>

    <output>
        <queue-output value="" type="FILE" help="Save the vehicle queues at the junctions (experimental)"/>
    </output>

    <time>
        <begin value="0" synonymes="b" type="TIME" help="Defines the begin time in seconds; The simulation starts at this time"/>
        <end value="endTimeInSeconds" synonymes="e" type="TIME" help="Defines the end time in seconds; The simulation ends at this time"/>
    </time>

    <report>
        <verbose value="true" synonymes="v" type="BOOL" help="Switches to verbose output"/>
        <no-step-log value="true" type="BOOL" help="Disable console output of current simulation step"/>
    </report>

</configuration>

Fill in the net-file and route-file values with the corresponding file names.
Set the end time value to the corresponding end time.
Save file.
Open SUMO-gui by pressing windows key and typing 'sumo-gui' and pressing enter.
Open the configuration file.
Play simulation to verify it works.

# Editing SUMO light orientation:
Go into net.xml file and locate the "linkIndex" values of the desired light.
Re-arrange index to account for action space choices.
