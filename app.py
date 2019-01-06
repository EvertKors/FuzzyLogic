import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import sys

DEGREES = int(sys.argv[1])

x_qual = np.arange(-90, 90, 1)
#motor = np.arange(0, 11, 1)

# Memberships for the input (line position)
left = fuzz.trimf(x_qual, [-90, -90, 0])
straight = fuzz.trimf(x_qual, [-90, 0, 90])
right = fuzz.trimf(x_qual, [0, 90, 90])

# Memberships for the output (direction)
robot_left = fuzz.trimf(x_qual, [-90, -90, 0])
robot_straight = fuzz.trimf(x_qual, [-90, 0, 90])
robot_right = fuzz.trimf(x_qual, [0, 90, 90])

# Membership for the output (speed)
speed = fuzz.pimf(x_qual, -90, -20, 20, 90)

# Prepare plots
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

# Plot the input
ax0.plot(x_qual, left, 'b', linewidth=1.5, label='left')
ax0.plot(x_qual, straight, 'g', linewidth=1.5, label='straight')
ax0.plot(x_qual, right, 'r', linewidth=1.5, label='right')
ax0.set_title('Input: Path position')
ax0.set_xlabel('Position of path (in degrees relative to current position)')
ax0.legend()

# Plot the direction output
ax1.plot(x_qual, robot_left, 'b', linewidth=1.5, label='left')
ax1.plot(x_qual, robot_straight, 'g', linewidth=1.5, label='straight')
ax1.plot(x_qual, robot_right, 'r', linewidth=1.5, label='right')
ax1.set_title('Output: Robot direction')
ax1.set_xlabel('The direction the robot should navigate to.')
ax1.legend()

# Plot the speed output
ax2.plot(x_qual, speed, 'b', linewidth=1.5, label='speed')
ax2.set_title('Output: Motor speed')
ax2.set_xlabel('The speed of the motor')
ax2.legend()


for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# Set the layout
plt.tight_layout()

#
# Membership plots
#

# Define the memberships
q_left = fuzz.interp_membership(x_qual, left, DEGREES)
q_straight = fuzz.interp_membership(x_qual, straight, DEGREES)
q_right = fuzz.interp_membership(x_qual, right, DEGREES)

# Activation methods
activation_l = np.fmin(q_left, robot_left)
activation_m = np.fmin(q_straight, robot_straight)
activation_r = np.fmin(q_right, robot_right)
x0 = np.zeros_like(x_qual)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

# Fill the visualisation
ax0.fill_between(x_qual, x0, activation_l, facecolor='b', alpha=0.7)
ax0.plot(x_qual, robot_left, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_qual, x0, activation_m, facecolor='g', alpha=0.7)
ax0.plot(x_qual, robot_straight, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_qual, x0, activation_r, facecolor='r', alpha=0.7)
ax0.plot(x_qual, robot_right, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity for Directon')

# Turn off top/right axes
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.get_xaxis().tick_bottom()
ax0.get_yaxis().tick_left()


#
# Output speed
#
plt.tight_layout()

# Speed relative to degree of membership of straightness
q_motor = fuzz.interp_membership(x_qual, straight, DEGREES)
activation = np.fmin(q_motor, speed)  # removed entirely to 0
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.fill_between(x_qual, x0, activation, facecolor='b', alpha=0.7)
ax0.plot(x_qual, speed, 'b', linewidth=0.5, linestyle='--', )
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.get_xaxis().tick_bottom()
ax0.get_yaxis().tick_left()

plt.tight_layout()

plt.show()

