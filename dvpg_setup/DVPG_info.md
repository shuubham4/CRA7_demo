# Accessing data through InfluxDB:
1. Login to your CARDS DVPG account.
2. Go to the code folder
3. You may have to install InfluxDB (sudo apt-get install python3-influxdb)
4. Run python3 InfluxDBQuery.py
5. You can change the topic information in the code to see the data status of different towers and sensors.

**Or,**

Run directly the following code:
- influx -precision rfc3339 -username guest -password MSA-InfluxDB-guest -database msatestbed -host 192.168.250.32
- You can query for data from different towers using the following command:
-- SELECT value FROM <measurement_name> where source_id = 'tower_0'
-- Change the  <measurement_name> with any of the following:
