#!/usr/bin/env python3
import rospy
import sys, select, termios, tty
from ackermann_msgs.msg import AckermannDrive

# --- CONFIGURATION ---
MAX_SPEED = 5.5   # Max speed command
MAX_STEER = 0.61   # Max steering angle (approx 35 degrees)
SPEED_STEP = 0.5   # Speed change per tap
STEER_STEP = 0.1   # Steering change per loop while holding key
CENTERING_RATE = 0.7 # Strength of auto-centering (0.0 to 1.0). Lower = faster centering.

msg = """
---------------------------
Auto-Centering Teleop
---------------------------
   W / S  : Increase / Decrease Speed (Sticky)
   A / D  : Turn Left / Right (Auto-Return)
   SPACE  : Emergency Stop
   CTRL-C : Quit

Current Speed: %.2f | Steer: %.2f
"""


def print_status(speed, steer, prefix=""):
    label = f"{prefix} " if prefix else ""
    print(f"{label}Speed: {speed:.2f} | Steer: {steer:.2f}")

def getKey(settings, timeout=0.1):
    tty.setraw(sys.stdin.fileno())
    # select waits for input for 'timeout' seconds
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('gem_teleop_key')
    
    pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
    
    target_speed = 0.0
    target_steer = 0.0

    try:
        print(msg % (target_speed, target_steer))
        while not rospy.is_shutdown():
            # Wait briefly for key press
            key = getKey(settings, timeout=0.1)
            
            # --- THROTTLE LOGIC (Sticky) ---
            if key == 'w':
                target_speed = min(MAX_SPEED, target_speed + SPEED_STEP)
                print_status(target_speed, target_steer, "Speed UP")
            elif key == 's':
                target_speed = max(-MAX_SPEED, target_speed - SPEED_STEP)
                print_status(target_speed, target_steer, "Speed DOWN")
            elif key == ' ':
                target_speed = 0.0
                target_steer = 0.0
                print_status(target_speed, target_steer, "!!! STOP !!!")
            elif key == '\x03': # Ctrl+C
                break

            # --- STEERING LOGIC (Auto-Centering) ---
            if key == 'a':
                # Increase left steering
                target_steer = min(MAX_STEER, target_steer + STEER_STEP)
                print_status(target_speed, target_steer, "Steer LEFT")
            elif key == 'd':
                # Increase right steering
                target_steer = max(-MAX_STEER, target_steer - STEER_STEP)
                print_status(target_speed, target_steer, "Steer RIGHT")
            else:
                # If no steering key is pressed (or if 'w'/'s' was pressed),
                # Decay steering towards 0
                target_steer = target_steer * CENTERING_RATE
                
                # Snap to 0 if very small
                if abs(target_steer) < 0.01:
                    target_steer = 0.0

            # --- PUBLISH ---
            ack = AckermannDrive()
            ack.speed = target_speed
            ack.steering_angle = target_steer
            
            pub.publish(ack)

            # Optional: Limit print rate so it doesn't flicker too much
            # print(f"\rSpd: {target_speed:.2f} | Str: {target_steer:.2f}", end="")

    except Exception as e:
        print(e)

    finally:
        # Stop the car on exit
        ack = AckermannDrive()
        ack.speed = 0; ack.steering_angle = 0
        pub.publish(ack)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
