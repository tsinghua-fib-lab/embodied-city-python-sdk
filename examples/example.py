from embodiedcity.client import DroneClient, ImageType, CameraID
import numpy as np

base_url = "https://embodied-city.fiblab.net"
drone_id = "x" # The drone ID taken over, such as "0", "1"
token = "xxxxxxxx"

# Initializing the drone
client = DroneClient(base_url, drone_id, token)

# Get the current pose {[x, y, z], [pitch, roll, yaw]}
pos = client.get_current_state()

# Rise 10 meters
client.move_vertical(10)

# Descend 10 meters
client.move_vertical(-10)

# Move 10 meters to the left
client.move_horizontal(10)

# Move forward 10 meters
client.move_back_forth(10)

# Rotates counterclockwise pi/2
client.move_by_yaw(np.pi / 2)

# Obtain the RGB image of the front camera
img = client.get_image(0, 0)

# Drone forced to move to designated location
client.set_vehicle_pose(6.50150258e+03, -4.19969414e+03, -1.31595741e+00,  0.00000000e+00,
        0.00000000e+00,  1.80000000e+02)