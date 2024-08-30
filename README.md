# embodied-city-python-sdk

A Simple Python SDK to interact with the Embodied City API.

## Installation

```bash
pip install embodiedcity
```

## Usage

#### Acquire ID and token

Before you can use the SDK, you need to acquire a drone and obtain its token.
You can get one by signing up at [Embodied City](https://embodied-city.fiblab.net/).

In the website, you should go to the "Console" page, choose an available drone, and click on the "Acquire" button.
After that, you will see a token with the drone ID.

> ATTENTION: The token is a secret key that should not be shared with anyone.

> ATTENTION: The token will expire after a certain period of time if you do not use it. (the time constrain will be notified in the website)

#### Initialize the client

```python
from embodiedcity import DroneClient, ImageType, CameraID

base_url = "https://embodied-city.fiblab.net"
drone_id = "xxx"
token = "xxxxxxxx"
client = DroneClient(base_url, drone_id, token)
```

#### Move the drone

```python
# Move the drone forward by 1 meter
client.move_back_forth(1)
```

#### Take a picture
    
```python
# Take a picture of the scene
image = client.take_picture(ImageType.Scene, CameraID.FrontCenter)
```

#### Get the depth image
    
```python
# Take a picture of the depth
image = client.take_picture(ImageType.DepthPlanar, CameraID.FrontCenter)
```

## Release the drone

After you finish using the drone, you should release it to make it available for others.

You can do this by clicking on the "Release" button in the "Console" page.
