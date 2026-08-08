#!/usr/bin/env python3
import argparse
import rospy
import pandas as pd
import math
import time
from pathlib import Path
from gazebo_msgs.msg import ModelStates
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion

from gem_mpc import paths

class DataRecorder:
    def __init__(self, out_path: Path = None):
        rospy.init_node('data_recorder', anonymous=True)

        # default to a timestamped log inside data/ so runs never overwrite
        self.out_path = Path(out_path) if out_path else (
            paths.DATA_DIR / f"gem_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )

        # Configuration
        self.robot_name = "gem" 
        self.data = []
        
        # State variables
        self.current_cmd_speed = 0.0
        self.current_cmd_steer = 0.0
        self.steer_actual = math.nan
        self.steer_rate = math.nan
        self._left_idx = None
        self._right_idx = None
        
        # Subscribers
        rospy.Subscriber("/gem/ackermann_cmd", AckermannDrive, self.cmd_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/gem/joint_states", JointState, self.joint_state_callback)
        
        print(f"Recorder started. Data will be saved to {self.out_path} on shutdown...")

    def cmd_callback(self, msg):
        # Store the latest command
        self.current_cmd_speed = msg.speed
        self.current_cmd_steer = msg.steering_angle

    def joint_state_callback(self, msg: JointState):
        # Lazily cache the joint indices once we see them
        try:
            if self._left_idx is None:
                self._left_idx = msg.name.index("left_steering_hinge_joint")
            if self._right_idx is None:
                self._right_idx = msg.name.index("right_steering_hinge_joint")
        except ValueError:
            # Joint names not present in this message
            print("Joint names not present in this message")
            return

        left_theta = msg.position[self._left_idx] if len(msg.position) > self._left_idx else 0.0
        right_theta = msg.position[self._right_idx] if len(msg.position) > self._right_idx else 0.0
        left_rate = msg.velocity[self._left_idx] if len(msg.velocity) > self._left_idx else 0.0
        right_rate = msg.velocity[self._right_idx] if len(msg.velocity) > self._right_idx else 0.0

        # Use the average of left/right steering joints as the actual steering angle
        self.steer_actual = 0.5 * (left_theta + right_theta)
        self.steer_rate = 0.5 * (left_rate + right_rate)

    def state_callback(self, msg):
        try:
            # Find the index of our robot
            idx = msg.name.index(self.robot_name)
        except ValueError:
            print("Robot name not present in this message")
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
            'steer_actual': self.steer_actual,
            'steer_rate': self.steer_rate,
            'x': p.x,
            'y': p.y,
            'yaw': yaw,
            'v_actual': speed_actual,
            'yaw_rate': w.z
        }
        self.data.append(row)

    def save_data(self):
        df = pd.DataFrame(self.data)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_path, index=False)
        print(f"Data saved to {self.out_path} with {len(df)} rows.")


def main():
    parser = argparse.ArgumentParser(description="Record GEM driving data to CSV.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output CSV path (default: data/gem_data_<timestamp>.csv)",
    )
    # myargv strips the remapping arguments roslaunch appends
    args = parser.parse_args(rospy.myargv()[1:])

    recorder = DataRecorder(out_path=args.out)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        recorder.save_data()


if __name__ == '__main__':
    main()
