from embodiedcity.client import DroneClient, ImageType, CameraID
import numpy as np
import asyncio

async def main():
    # Initializing the drone
    base_url = "https://embodied-city.fiblab.net"
    drone_id = ""  # The drone ID taken over, such as "0", "1"
    token = ""
    client = DroneClient(base_url, drone_id, token)

    pois = client.query_pois(6825, -3795, 10, "")
    print(pois)

    # Get the current pose {[x, y, z], [pitch, roll, yaw]}
    pos = await client.aget_current_state()

    # Rise 10 meters
    await client.amove_vertical(10)

    # Descend 10 meters
    await client.amove_vertical(-10)

    # Move 10 meters to the left
    await client.amove_horizontal(10)

    # Move forward 10 meters
    await client.amove_back_forth(10)

    # Rotates counterclockwise pi/2
    await client.amove_by_yaw(np.pi / 2)

    # Obtain the RGB image of the front camera
    img = await client.aget_image(ImageType.Scene, CameraID.FrontCenter)

    # Drone forced to move to designated location
    await client.aset_vehicle_pose(
        6.50150258e03,
        -4.19969414e03,
        -1.31595741e00,
        0.00000000e00,
        0.00000000e00,
        1.80000000e02,
    )

if __name__ == "__main__":
    asyncio.run(main())