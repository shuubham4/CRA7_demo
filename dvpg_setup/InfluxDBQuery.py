# -*- coding: utf-8 -*-

from influxdb import InfluxDBClient
import requests
import json

hostname = "192.168.250.32"
port = 8086
username = "guest"
password = "MSA-InfluxDB-guest"
databasename = "msatestbed"


client = InfluxDBClient(host=hostname,
                        port=8086, username=username,
                        password=password,
                        database=databasename
                        )

#query1String = "select * from ambient_temp where source_id='tower_1' " \
#    "order by time desc limit 10"
#query1String = "select * from internal_temperature order by time desc limit 10"
query1String = "select * from movement_detection order by time desc limit 10"

results = client.query(query1String)

#    "SELECT * FROM \"ambient_temp\" " +
#    "where source_id = 'tower_0' order by time desc")
#    "where source_id =~ /tower*/ "+
#    "group by source_id order by time desc limit 10")

print(results)

i = 0
for point in results.get_points():
    i += 1
    print(i," = ", point)
