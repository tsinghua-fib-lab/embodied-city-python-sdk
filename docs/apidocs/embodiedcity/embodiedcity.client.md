# {py:mod}`embodiedcity.client`

```{py:module} embodiedcity.client
```

```{autodoc2-docstring} embodiedcity.client
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImageType <embodiedcity.client.ImageType>`
  - ```{autodoc2-docstring} embodiedcity.client.ImageType
    :summary:
    ```
* - {py:obj}`CameraID <embodiedcity.client.CameraID>`
  - ```{autodoc2-docstring} embodiedcity.client.CameraID
    :summary:
    ```
* - {py:obj}`POI <embodiedcity.client.POI>`
  - ```{autodoc2-docstring} embodiedcity.client.POI
    :summary:
    ```
* - {py:obj}`AffineTransformer <embodiedcity.client.AffineTransformer>`
  - ```{autodoc2-docstring} embodiedcity.client.AffineTransformer
    :summary:
    ```
* - {py:obj}`DroneClient <embodiedcity.client.DroneClient>`
  - ```{autodoc2-docstring} embodiedcity.client.DroneClient
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <embodiedcity.client.__all__>`
  - ```{autodoc2-docstring} embodiedcity.client.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: embodiedcity.client.__all__
:value: >
   ['ImageType', 'CameraID', 'POI', 'DroneClient']

```{autodoc2-docstring} embodiedcity.client.__all__
```

````

`````{py:class} ImageType
:canonical: embodiedcity.client.ImageType

```{autodoc2-docstring} embodiedcity.client.ImageType
```

````{py:attribute} Scene
:canonical: embodiedcity.client.ImageType.Scene
:value: >
   0

```{autodoc2-docstring} embodiedcity.client.ImageType.Scene
```

````

````{py:attribute} DepthPlanar
:canonical: embodiedcity.client.ImageType.DepthPlanar
:value: >
   1

```{autodoc2-docstring} embodiedcity.client.ImageType.DepthPlanar
```

````

````{py:attribute} Segmentation
:canonical: embodiedcity.client.ImageType.Segmentation
:value: >
   2

```{autodoc2-docstring} embodiedcity.client.ImageType.Segmentation
```

````

`````

`````{py:class} CameraID
:canonical: embodiedcity.client.CameraID

```{autodoc2-docstring} embodiedcity.client.CameraID
```

````{py:attribute} FrontCenter
:canonical: embodiedcity.client.CameraID.FrontCenter
:value: >
   0

```{autodoc2-docstring} embodiedcity.client.CameraID.FrontCenter
```

````

````{py:attribute} FrontRight
:canonical: embodiedcity.client.CameraID.FrontRight
:value: >
   1

```{autodoc2-docstring} embodiedcity.client.CameraID.FrontRight
```

````

````{py:attribute} FrontLeft
:canonical: embodiedcity.client.CameraID.FrontLeft
:value: >
   2

```{autodoc2-docstring} embodiedcity.client.CameraID.FrontLeft
```

````

````{py:attribute} BottomCenter
:canonical: embodiedcity.client.CameraID.BottomCenter
:value: >
   3

```{autodoc2-docstring} embodiedcity.client.CameraID.BottomCenter
```

````

````{py:attribute} BackCenter
:canonical: embodiedcity.client.CameraID.BackCenter
:value: >
   4

```{autodoc2-docstring} embodiedcity.client.CameraID.BackCenter
```

````

`````

`````{py:class} POI
:canonical: embodiedcity.client.POI

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} embodiedcity.client.POI
```

````{py:attribute} id
:canonical: embodiedcity.client.POI.id
:type: int
:value: >
   None

```{autodoc2-docstring} embodiedcity.client.POI.id
```

````

````{py:attribute} x
:canonical: embodiedcity.client.POI.x
:type: float
:value: >
   None

```{autodoc2-docstring} embodiedcity.client.POI.x
```

````

````{py:attribute} y
:canonical: embodiedcity.client.POI.y
:type: float
:value: >
   None

```{autodoc2-docstring} embodiedcity.client.POI.y
```

````

````{py:attribute} name
:canonical: embodiedcity.client.POI.name
:type: str
:value: >
   None

```{autodoc2-docstring} embodiedcity.client.POI.name
```

````

````{py:attribute} category
:canonical: embodiedcity.client.POI.category
:type: str
:value: >
   None

```{autodoc2-docstring} embodiedcity.client.POI.category
```

````

`````

`````{py:class} AffineTransformer()
:canonical: embodiedcity.client.AffineTransformer

```{autodoc2-docstring} embodiedcity.client.AffineTransformer
```

```{rubric} Initialization
```

```{autodoc2-docstring} embodiedcity.client.AffineTransformer.__init__
```

````{py:method} compute_affine_transform(src_points, dst_points)
:canonical: embodiedcity.client.AffineTransformer.compute_affine_transform

```{autodoc2-docstring} embodiedcity.client.AffineTransformer.compute_affine_transform
```

````

````{py:method} apply_affine_transform(points, transform)
:canonical: embodiedcity.client.AffineTransformer.apply_affine_transform

```{autodoc2-docstring} embodiedcity.client.AffineTransformer.apply_affine_transform
```

````

````{py:method} local_to_latlon(x, y)
:canonical: embodiedcity.client.AffineTransformer.local_to_latlon

```{autodoc2-docstring} embodiedcity.client.AffineTransformer.local_to_latlon
```

````

````{py:method} latlon_to_local(lat, lon)
:canonical: embodiedcity.client.AffineTransformer.latlon_to_local

```{autodoc2-docstring} embodiedcity.client.AffineTransformer.latlon_to_local
```

````

`````

`````{py:class} DroneClient(base_url: str, drone_id: str, token: str)
:canonical: embodiedcity.client.DroneClient

```{autodoc2-docstring} embodiedcity.client.DroneClient
```

```{rubric} Initialization
```

```{autodoc2-docstring} embodiedcity.client.DroneClient.__init__
```

````{py:method} _make_request(action: str, *args)
:canonical: embodiedcity.client.DroneClient._make_request

```{autodoc2-docstring} embodiedcity.client.DroneClient._make_request
```

````

````{py:method} _amake_request(action: str, *args)
:canonical: embodiedcity.client.DroneClient._amake_request
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient._amake_request
```

````

````{py:method} _prepare_pois()
:canonical: embodiedcity.client.DroneClient._prepare_pois

```{autodoc2-docstring} embodiedcity.client.DroneClient._prepare_pois
```

````

````{py:method} move_back_forth(distance: float)
:canonical: embodiedcity.client.DroneClient.move_back_forth

```{autodoc2-docstring} embodiedcity.client.DroneClient.move_back_forth
```

````

````{py:method} amove_back_forth(distance: float)
:canonical: embodiedcity.client.DroneClient.amove_back_forth
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient.amove_back_forth
```

````

````{py:method} move_horizontal(distance: float)
:canonical: embodiedcity.client.DroneClient.move_horizontal

```{autodoc2-docstring} embodiedcity.client.DroneClient.move_horizontal
```

````

````{py:method} amove_horizontal(distance: float)
:canonical: embodiedcity.client.DroneClient.amove_horizontal
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient.amove_horizontal
```

````

````{py:method} move_vertical(distance: float)
:canonical: embodiedcity.client.DroneClient.move_vertical

```{autodoc2-docstring} embodiedcity.client.DroneClient.move_vertical
```

````

````{py:method} amove_vertical(distance: float)
:canonical: embodiedcity.client.DroneClient.amove_vertical
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient.amove_vertical
```

````

````{py:method} move_by_yaw(yaw: float)
:canonical: embodiedcity.client.DroneClient.move_by_yaw

```{autodoc2-docstring} embodiedcity.client.DroneClient.move_by_yaw
```

````

````{py:method} amove_by_yaw(yaw: float)
:canonical: embodiedcity.client.DroneClient.amove_by_yaw
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient.amove_by_yaw
```

````

````{py:method} get_image(image_type: int, camera_id: int)
:canonical: embodiedcity.client.DroneClient.get_image

```{autodoc2-docstring} embodiedcity.client.DroneClient.get_image
```

````

````{py:method} aget_image(image_type: int, camera_id: int)
:canonical: embodiedcity.client.DroneClient.aget_image
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient.aget_image
```

````

````{py:method} get_current_state()
:canonical: embodiedcity.client.DroneClient.get_current_state

```{autodoc2-docstring} embodiedcity.client.DroneClient.get_current_state
```

````

````{py:method} aget_current_state()
:canonical: embodiedcity.client.DroneClient.aget_current_state
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient.aget_current_state
```

````

````{py:method} move_to_position(x: float, y: float, z: float)
:canonical: embodiedcity.client.DroneClient.move_to_position

```{autodoc2-docstring} embodiedcity.client.DroneClient.move_to_position
```

````

````{py:method} amove_to_position(x: float, y: float, z: float)
:canonical: embodiedcity.client.DroneClient.amove_to_position
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient.amove_to_position
```

````

````{py:method} set_vehicle_pose(x: float, y: float, z: float, pitch: float, roll: float, yaw: float)
:canonical: embodiedcity.client.DroneClient.set_vehicle_pose

```{autodoc2-docstring} embodiedcity.client.DroneClient.set_vehicle_pose
```

````

````{py:method} aset_vehicle_pose(x: float, y: float, z: float, pitch: float, roll: float, yaw: float)
:canonical: embodiedcity.client.DroneClient.aset_vehicle_pose
:async:

```{autodoc2-docstring} embodiedcity.client.DroneClient.aset_vehicle_pose
```

````

````{py:method} query_pois(x: float, y: float, radius: float, category_prefix: str = '', limit: typing.Optional[int] = None) -> typing.List[typing.Tuple[embodiedcity.client.POI, float]]
:canonical: embodiedcity.client.DroneClient.query_pois

```{autodoc2-docstring} embodiedcity.client.DroneClient.query_pois
```

````

`````
