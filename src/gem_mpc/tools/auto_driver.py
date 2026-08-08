#!/usr/bin/env python3
import rospy
import math
import numpy as np
from ackermann_msgs.msg import AckermannDrive

def run_driver():
    rospy.init_node('auto_driver', anonymous=True)
    pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=10)
    rate = rospy.Rate(10) # 10Hz control loop

    start_time = rospy.get_time()
    
    while not rospy.is_shutdown():
        t = rospy.get_time() - start_time
        
        msg = AckermannDrive()
        # --- CONTROL LOGIC ---
        # 1. Vary Speed: Ramp up to max (5.5 m/s approx 20km/h), then slow down
        # We use a slow sine wave for speed: oscillates between 1 m/s and 4 m/s
        target_speed = 2.5 + 1.5 * math.sin(0.1 * t)
        
        # 2. Vary Steering: Faster sine wave to weave left and right
        # Max steering is usually around 0.6 radians
        target_steer = 0.5 * math.sin(0.5 * t)
        
        msg.speed = target_speed
        msg.steering_angle = target_steer
        
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        run_driver()
    except rospy.ROSInterruptException:
        pass
