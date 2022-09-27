# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 08:03:29 2022

@author: tgregory
"""
broker="192.168.250.51"
port=1883

import paho.mqtt.client as paho
import sys
import json
import os
import time

debuggingstring = '{"time":"2022-01-28T07:30:39-07:00","location":{"lat":32.59022433,"lon":-106.7659365,"alt":{"value":1000.0,"measurement":"cm"}},"tower":"tower_2","sensorType":"temperature","reading":[{"name":"temperature","value":-3.495483,"measurement":"celsius"}]}'

def on_connect(client, userdata,result,dat):
    if (dat==0):
        print ("Connected to "+broker+":"+str(port))
    else:
        print("CONNECTION REFUSED ("+broker+":"+str(port)+")")
        
def processTemperatureReading(reading):
    for r in reading:
        readingname = r.get('name')
        readingvalue = r.get('value')
        readingunits = r.get('measurement')
        if (readingname == 'temperature'):
            if (readingvalue):
                if (readingunits):
                    if (readingunits=='celsius'):
                        readingvalue=(9/5*readingvalue)+32
                print (readingname+": "+"{:.1f}".format(readingvalue) + " deg F")
        
def on_message(client, userdata, message):
    jsonresults = json.loads(message.payload.decode("utf-8"))
    # The following line will print the entire JSON payload.
    print("message received  ",str(message.payload.decode("utf-8")),
       "topic",message.topic,"retained ",message.retain)
#    try:
    sensortype = jsonresults.get('sensorType')
    reading = jsonresults.get('reading')
    if (sensortype):
        if (sensortype == 'temperature'):
            if reading:
                processTemperatureReading(reading)
                    
        
def on_disconnect(client,userdata,result):
    print("Lost connection to MQTT Server")
#    print("userdata="+str(userdata))
#    print("result="+str(result))

def on_subscribe(client,userdata, mid, qos):
    print("on_subscribe")
    print(userdata)
    print(mid)
    
    
     
client1 = paho.Client("control1")
client1.on_connect = on_connect
client1.on_disconnect = on_disconnect
client1.on_subscribe = on_subscribe
client1.on_message = on_message


print ("Connecting to: "+broker+":"+str(port))
client1.connect(broker,port)
while (not client1.is_connected()):
    print("Waiting for connection to MQTT Server")
    client1.loop()
    time.sleep(1)
    if (client1.is_connected()):
        print("CONNECTED!!!")

topic="dvpg/msa/tower/2/height/200/#"
subscriberesult = client1.subscribe(topic,1)
print("subscriberesult="+str(subscriberesult))

client1.loop_start()
counter=0
try:
    while (client1.is_connected()):
        if (not client1.is_connected()):
            print("Disconnected!")
        time.sleep(1)
        if (counter % 10) == 0:
            print("**** Press CTRL-C to exit! ****")
            counter = 0
        counter += 1            

except KeyboardInterrupt:
    print("Exiting main loop.")

client1.loop_stop()  
print("Removing Subscription to topic: "+topic)
client1.unsubscribe(topic)
print("Disconnecting from MQTT Server at "+broker+":"+str(port))
client1.disconnect()

