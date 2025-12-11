#!/usr/bin/env python3
import rospy
import pandas as pd
import math
from gazebo_msgs.msg import ModelStates
from ackermann_msgs.msg import AckermannDrive
from tf.transformations import euler_from_quaternion

class DataRecorder:
    def __init__(self):
        rospy.init_node('data_recorder', anonymous=True)
        
        # Configuration
        self.robot_name = "gem" # CHECK THIS: Might be "polaris" or "gem"
        self.data = []
        
        # State variables
        self.current_cmd_speed = 0.0
        self.current_cmd_steer = 0.0
        
        # Subscribers
        rospy.Subscriber("/gem/ackermann_cmd", AckermannDrive, self.cmd_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        
        print("Recorder started. Driving data will be saved on shutdown...")

    def cmd_callback(self, msg):
        # Store the latest command
        self.current_cmd_speed = msg.speed
        self.current_cmd_steer = msg.steering_angle

    def state_callback(self, msg):
        try:
            # Find the index of our robot
            idx = msg.name.index(self.robot_name)
        except ValueError:
            return

        # Extract Position
        p = msg.pose[idx].position
        q = msg.pose[idx].orientation
        
        # Convert Quaternion to Yaw
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Extract Velocity (Local Frame usually preferred, but Global is fine for raw data)
        v = msg.twist[idx].linear
        w = msg.twist[idx].angular
        
        # Calculate scalar speed (approximate)
        speed_actual = math.sqrt(v.x**2 + v.y**2)

        # Record Data Row
        # Time, Input_Speed, Input_Steer, X, Y, Yaw, V_actual, Yaw_Rate
        row = {
            'time': rospy.get_time(),
            'cmd_speed': self.current_cmd_speed,
            'cmd_steer': self.current_cmd_steer,
            'x': p.x,
            'y': p.y,
            'yaw': yaw,
            'v_actual': speed_actual,
            'yaw_rate': w.z
        }
        self.data.append(row)

    def save_data(self):
        df = pd.DataFrame(self.data)
        df.to_csv('gem_data.csv', index=False)
        print(f"Data saved to gem_data.csv with {len(df)} rows.")

if __name__ == '__main__':
    recorder = DataRecorder()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        recorder.save_data()