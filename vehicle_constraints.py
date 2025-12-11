# Basic actuator limits for the GEM vehicle.
# Keep these values in sync with both the training pipeline and MPC controller.

# Absolute longitudinal acceleration limit (applied symmetrically for accel/brake), m/s^2
MAX_ACCEL = 1.5

# Steering angle rate limit, rad/s
MAX_STEER_RATE = 1.5
