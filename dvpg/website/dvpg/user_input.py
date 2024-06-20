#!/usr/bin/env python3
from flask import Flask, render_template, request
import rospy
from std_msgs.msg import String

app = Flask(__name__)

# Initialize ROS node
rospy.init_node('web_input_node', anonymous=True)
pub = rospy.Publisher('web_input', String, queue_size=10)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        # Publish to ROS topic
        pub.publish(input_text)
        return render_template('index.html', input_text=input_text)
    return render_template('index.html', input_text=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')