try:
    import requests
except ImportError:
    print("Please install the requests package using `pip install requests`")
    exit(1)

__all__ = ["ImageType", "CameraID", "DroneClient"]

class ImageType:
    Scene = 0
    """Scene (RGB)"""
    DepthPlanar = 1
    """DepthPlanar (Depth)"""
    Segmentation = 2

class CameraID:
    front_center = 0
    """front_center"""
    front_right = 1
    """front_right"""
    front_left = 2
    """front_left"""
    bottom_center = 3
    """bottom_center"""
    back_center = 4
    """back_center"""

class DroneClient:
    def __init__(self, base_url: str, drone_id: str, token: str):
        """
        Args:
        - base_url: The base URL of the server
        - drone_id: The ID of the drone
        - token: The token to authenticate requests with the server
        """
        self._base_url = base_url.rstrip("/")
        self._drone_id = drone_id
        self._token = token

    def _make_request(self, action: str, *args):
        url = f"{self._base_url}/api/call-function"
        res = requests.post(url, json={
            "droneId": self._drone_id,
            "action": action,
            "args": args,
            "token": self._token
        })
        if res.status_code != 200:
            raise Exception(f"Failed to make request: {res.text}")
        # TODO: 图像要单独处理
        return res.json()
    
    def move_back_forth(self, distance: float):
        """
        Move the drone back and forth by a certain distance

        Args:
        - distance: The distance to move the drone by (unit: meters). distance > 0 means moving forward, distance < 0 means moving backward.

        Returns:
        - None
        """
        return self._make_request("move_back_forth", distance)
    
    def move_horizontal(self, distance: float):
        """
        Move the drone left and right by a certain distance

        Args:
        - distance: The distance to move the drone by (unit: meters). distance > 0 means moving left, distance < 0 means moving right.

        Returns:
        - None
        """
        return self._make_request("move_horizontal", distance)
    
    def move_vertical(self, distance: float):
        """
        Move the drone up and down by a certain distance

        Args:
        - distance: The distance to move the drone by (unit: meters). distance > 0 means moving up, distance < 0 means moving down.

        Returns:
        - None
        """
        return self._make_request("move_vertical", distance)
    
    def move_by_yaw(self, yaw: float):
        """
        Rotate the drone by a certain angle

        Args:
        - yaw: The angle to rotate the drone by (unit: radians). Positive values mean rotating counterclockwise, negative values mean rotating clockwise.

        Returns:
        - None
        """
        return self._make_request("move_by_yaw", yaw)
    
    def get_image(self, image_type: ImageType, camera_id: CameraID):
        """
        Get an image from the drone

        Args:
        - image_type: The type of image to get
        - camera_id: The ID of the camera to get the image from

        Returns:
        - numpy.array: The image
        """
        return self._make_request("get_image", int(image_type), int(camera_id))
    
    def get_current_state(self):
        """
        Get the current state of the drone

        Returns:
        - [x, y, z]: The position of the drone
        - [pitch, roll, yaw]: The orientation of the drone
        """

        return self._make_request("get_current_state")
    
    def move_to_position(self, x: float, y: float, z: float):
        """
        Move the drone to a certain position

        Args:
        - x: The x-coordinate of the target position
        - y: The y-coordinate of the target position
        - z: The z-coordinate of the target position

        Returns:
        - None
        """
        return self._make_request("move_to_position", x, y, z)
    
    def set_vehicle_pose(self, x: float, y: float, z: float, pitch: float, roll: float, yaw: float):
        """
        Set the pose of the drone.
        Attention: This function will teleport the drone to the target position.

        Args:
        - x: The x-coordinate of the target position
        - y: The y-coordinate of the target position
        - z: The z-coordinate of the target position
        - pitch: The pitch of the drone
        - roll: The roll of the drone
        - yaw: The yaw of the drone

        Returns:
        - None
        """
        return self._make_request("set_vehicle_pose", x, y, z, pitch, roll, yaw)