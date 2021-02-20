#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

from cv_bridge import CvBridge
from cv_bridge.boost.cv_bridge_boost import getCvType

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class Human_Check_Node:
    def __init__(self):
        # Define and parse input arguments
        self.min_conf_threshold = 0.5
        self.imW, self.imH = 1920, 1080
        # Path to .tflite file, which contains the model that is used for object detection
        self.PATH_TO_CKPT = '/home/ubuntu/catkin_ws/tflite1/models/coco_model/detect.tflite'

        # Path to label map file
        self.PATH_TO_LABELS = '/home/ubuntu/catkin_ws/tflite1/models/coco_model/labelmap.txt'

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.interpreter = tflite.Interpreter(model_path=self.PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

            # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        self.cv_bridge = CvBridge()
        # create a publisher object to send data
        self.human_check_publisher = rospy.Publisher("humancheck", Bool, queue_size=10)

        # TODO fill in the TOPIC_NAME and MESSAGE_TYPE
        rospy.Subscriber("/camera/color/image_raw", Image, self.human_check)

    def human_check(self, data):

        # skip messages which older then 1 sec
        msg_nsecs = data.header.stamp.nsecs
        now = rospy.get_rostime()
        if (msg_nsecs + 20000000 < now.nsecs):
            return
        human = False
        try:
            image = self.cv_bridge.imgmsg_to_cv2(data,'bgr8')
        except:
            rospy.logwarn('Image could not be transformed skipping image')
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.height, self.width))
        input_data = np.expand_dims(image_resized, axis=0)

            # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                if object_name == 'person': #if label is person stop looking and label it as a human
                    human = True
                    ymin = int(max(1,(boxes[i][0] * self.imH)))
                    xmin = int(max(1,(boxes[i][1] * self.imW)))
                    ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                    xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                    x = str(xmin+((xmax-xmin)/2))
                    y = str(ymin+((ymax-ymin)/2))
                    info ='Human found at: ('+ x +','+ y+')'
                    rospy.loginfo(info)
        self.human_check_publisher.publish(human)
        if human == False:
            rospy.loginfo('Seems like noone is arround...')
if __name__ == "__main__":
    rospy.init_node("Human_Check_Node")
    node = Human_Check_Node()
    rospy.spin()
