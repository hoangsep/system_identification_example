import rospy
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import JointState
import time

class JointChecker:
    def __init__(self):
        rospy.init_node('check_joints')
        self.pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        rospy.Subscriber('/gem/joint_states', JointState, self.callback)
        self.l_idx = None
        self.r_idx = None
        self.latest_msg = None

    def callback(self, msg):
        self.latest_msg = msg
        if self.l_idx is None:
            try:
                self.l_idx = msg.name.index("left_steering_hinge_joint")
                self.r_idx = msg.name.index("right_steering_hinge_joint")
            except: pass

    def run(self):
        # Command +0.5 rad (Left Turn)
        print("Commanding +0.5 rad (Left Turn)...")
        msg = AckermannDrive()
        msg.steering_angle = 0.5
        msg.speed = 0.0
        
        for _ in range(20):
            self.pub.publish(msg)
            rospy.sleep(0.1)
            
        if self.latest_msg and self.l_idx is not None:
            l = self.latest_msg.position[self.l_idx]
            r = self.latest_msg.position[self.r_idx]
            print(f"Angle Cmd: 0.5")
            print(f"Left Joint: {l:.4f}")
            print(f"Right Joint: {r:.4f}")
            
            if l > 0 and r < 0:
                print("⚠️ DIAGNOSIS: Right wheel joint is INVERTED relative to Left wheel!")
            elif l > 0 and r > 0:
                print("✅ DIAGNOSIS: Both wheels rotate in the same direction (Correct for model).")
            else:
                print("❓ DIAGNOSIS: Weird values.")
        else:
            print("No JointState received!")

if __name__ == '__main__':
    try:
        JointChecker().run()
    except rospy.ROSInterruptException: pass
