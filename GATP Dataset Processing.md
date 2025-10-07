<div align="center">

# GATP Dataset Processing (Step-by-Step)

</div>

GATP (Graph-Aware Task Planning Dataset) is a multimodal task planning dataset based on the enhanced ALFRED benchmark. By introducing scene graphs and task graphs, it is specifically designed to capture detailed object-action relationships and spatial structure, providing multi-dimensional information support for complex task planning. This article details how to gradually construct scene graphs and task graphs by leveraging ALFRED's JSON file information and AI2THOR's environmental object information.

## 1. Download the original ALFRED Benchmark


Download Trajectory JSONs and Resnet feats from the following link: [ALFRED Github](https://github.com/askforalfred/alfred)(The complete dataset is approximately 49GB)


## 2. ALFRED JSON Processing

The json file structure of the original dataset downloaded from ALFRED is as follows

```tree
train/
├── look_at_obj_in_light-AlarmClock-None-DeskLamp-301/
│   ├── trial_T20190907_174127_043461
│       ├──traj_data.json
│   ├── trial_T20190907_174142_375532
│       ├──traj_data.json
│   └── trial_T20190907_174157_650656
│       ├──traj_data.json
├── look_at_obj_in_light-AlarmClock-None-DeskLamp-302/
│   ├── trial_T20190908_192756_298295
│       ├──traj_data.json
│   ├── trial_T20190907_174142_375532
│       ├──traj_data.json
│   └── trial_T20190908_192820_480968
│       ├──traj_data.json
└── ...
```

What we need to do is recursively iterate through all JSON files in the source directory, extract the floor_plan, task_desc, high_descs, task_type, objectId, and actions, and save them to a JSON file with the same name as the floor_plan file in the source directory. Find all FloorPlan*.json files, merge them into a single JSON file based on the floor_plan name, save it to the target directory (/data), and delete the source file.

So, you can run:

```bash
python ALFRED_JSON_Process.py
```
The output file structure should be as follows:

```tree
data/
├── FloorPlan01/
│   └── FloorPlan01.json
├── FloorPlan02/
│   └── FloorPlan02.json
└── ...
```


## 3. FLoorPlan Task Processing

Split the task entries in the FloorPlan*.json file in the ALFRED dataset into pairs of task_desc and high_descs, convert each field to string format, and clean the objectid to retain only the alphabetical part.

We take FloorPlan1.json as an example to illustrate the processing process and input and output file structure.

The input file is FloorPlan1.json, and its file structure is as follows

```tree
[
    {
        "floor_plan": "FloorPlan1",
        "task_desc": [
            "Move a tea pot from the stove to a shelf.",
            "Put the tea kettle on the shelf",
            "placed tea pot on shelf "
        ],
        "high_descs": [
            [
                "Turn right and walk to the stove.",
                "Pick up the tea pot on the left side of the stove.",
                "Turn left and walk towards the shelves on the right.",
                "Place the tea pot on the middle shelf, to the left of the glass container."
            ],
            [
                "Walk into the kitchen and approach the stove",
                "Pick up the tea kettle from the left side of the stove",
                "Turn left and walk over to the shelf left of the kitchen counter",
                "Put the tea kettle on the shelf left of the clear glass vase"
            ],
            [
                "Turn to your right go straight then turn to left ",
                "Grab tea pot off of the stove",
                "Turn all the way around ",
                "Then turn around again walk to shelf and place tea pot on shelf "
            ]
        ],
        "task_type": "pick_and_place_simple",
        "objectid": [
            "Kettle|-00.04|+00.95|-02.58",
            "Kettle|-00.04|+00.95|-02.58",
            "Kettle|-00.04|+00.95|-02.58",
            "Kettle|-00.04|+00.95|-02.58"
        ],
        "actions": [
            "LookDown",
            "MoveAhead",
            "RotateRight",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "RotateLeft",
            "MoveAhead",
            "MoveAhead",
            "PickupObject",
            "RotateLeft",
            "RotateLeft",
            "MoveAhead",
            "RotateRight",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "RotateRight",
            "PutObject"
        ]
    },
    ...
]
```

Then, you can run:

```bash
python FloorPlan_Task_Process.py
```

The output file is FloorPlan1_Final.json, and its file structure is as follows


```tree
[
    {
        "task_desc": "Move a tea pot from the stove to a shelf.",
        "high_descs": "Turn right and walk to the stove. Pick up the tea pot on the left side of the stove. Turn left and walk towards the shelves on the right. Place the tea pot on the middle shelf, to the left of the glass container.",
        "actions": "LookDown MoveAhead RotateRight MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead RotateLeft MoveAhead MoveAhead PickupObject RotateLeft RotateLeft MoveAhead RotateRight MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead RotateRight PutObject",
        "objectid": "Kettle"
    },
    ...
]
```



## 4. Get Floorplan1_info.json from AI2THOR Simulator

In the AI2THOR simulation environment, we use 'event.metadata' to obtain detailed information about each object in each environment to build the scene graph dataset for the next step.

Taking FloorPlan1 as an example, the structure of the output item information should be as follows:

```tree
{
    "StoveBurner": [
        {
            "position": { "x": -0.0361, "y": 0.9151, "z": -2.3722 },
            "rotation": { "x": 0.0, "y": 0.0, "z": 0.0 },
            "cameraHorizon": 0.0,
            "visible": false,
            "receptacle": true,
            "toggleable": false,
            "isToggled": false,
            "breakable": false,
            "isBroken": false,
            "canFillWithLiquid": false,
            "isFilledWithLiquid": false,
            "dirtyable": false,
            "isDirty": false,
            "canBeUsedUp": false,
            "isUsedUp": false,
            "cookable": false,
            "isCooked": false,
            "ObjectTemperature": "RoomTemp",
            "canChangeTempToHot": true,
            "canChangeTempToCold": false,
            "sliceable": false,
            "isSliced": false,
            "openable": false,
            "isOpen": false,
            "pickupable": false,
            "isPickedUp": false,
            "mass": 0.0,
            "salientMaterials": null,
            "receptacleObjectIds": [],
            "distance": 3.50728321,
            "objectId": "StoveBurner|-00.04|+00.92|-02.37",
            "parentReceptacle": null,
            "parentReceptacles": null,
            "currentTime": 0.0,
            "isMoving": false,
            "objectBounds": null
        }
    ],
    "Drawer": [
        {
            "position": { "x": 0.95, "y": 0.8328, "z": -2.197 },
            "rotation": { "x": 0.0, "y": 0.0, "z": 0.0 },
            "cameraHorizon": 0.0,
            "visible": false,
            "receptacle": true,
            "toggleable": false,
            "isToggled": false,
            "breakable": false,
            "isBroken": false,
            "canFillWithLiquid": false,
            "isFilledWithLiquid": false,
            "dirtyable": false,
            "isDirty": false,
            "canBeUsedUp": false,
            "isUsedUp": false,
            "cookable": false,
            "isCooked": false,
            "ObjectTemperature": "RoomTemp",
            "canChangeTempToHot": false,
            "canChangeTempToCold": false,
            "sliceable": false,
            "isSliced": false,
            "openable": true,
            "isOpen": false,
            "pickupable": false,
            "isPickedUp": false,
            "mass": 0.0,
            "salientMaterials": null,
            "receptacleObjectIds": [],
            "distance": 3.74539185,
            "objectId": "Drawer|+00.95|+00.83|-02.20",
            "parentReceptacle": null,
            "parentReceptacles": null,
            "currentTime": 0.0,
            "isMoving": false,
            "objectBounds": null
        }
    ]
    ...
}
```


## 5. Create Scene Graph Dataset

We construct a scene graph dataset based on the scene information extracted from each scene

So, you can run:

```bash
python FloorPlan_SceneGraph_Build.py
```

The final constructed scene graph dataset is shown below:

```tree
{
    "nodes": [
        {
            "id": "Knife"
        },
        {
            "id": "Potato"
        },
        {
            "id": "Table"
        },
        {
            "id": "Microwave"
        },
        ...
    ],
    "links": [
        {
            "source": "Knife",
            "target": "Potato",
            "relationship": "Slice"
        },
        {
            "source": "Potato",
            "target": "Table",
            "relationship": "on"
        },
        {
            "source": "Knife",
            "target": "Table",
            "relationship": "on"
        },
        {
            "source": "Potato",
            "target": "Microwave",
            "relationship": "heat"
        },
        ...
    ]
}
```


## 6. Create Task Graph Dataset

Based on the constructed Scene Graph dataset and the task information of each scene, we build a task graph dataset for each scene.

So, you can run:

```bash
python FloorPlan_TaskGraph_Build.py
```

The structure of the task data in each input scene is as follows:

```tree

{
        "task_desc": "Move a tea pot from the stove to a shelf.",
        "high_descs": "Turn right and walk to the stove. Pick up the tea pot on the left side of the stove. Turn left and walk towards the shelves on the right. Place the tea pot on the middle shelf, to the left of the glass container.",
        "actions": "LookDown MoveAhead RotateRight MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead RotateLeft MoveAhead MoveAhead PickupObject RotateLeft RotateLeft MoveAhead RotateRight MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead RotateRight PutObject",
        "objectid": "Kettle"
    },
    {
        "task_desc": "Put the tea kettle on the shelf",
        "high_descs": "Walk into the kitchen and approach the stove Pick up the tea kettle from the left side of the stove Turn left and walk over to the shelf left of the kitchen counter Put the tea kettle on the shelf left of the clear glass vase",
        "actions": "LookDown MoveAhead RotateRight MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead RotateLeft MoveAhead MoveAhead PickupObject RotateLeft RotateLeft MoveAhead RotateRight MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead RotateRight PutObject",
        "objectid": "Kettle"
    },
    ...
}
```

The output task graph data is embedded in each of the above task data

```tree

{
        "task_desc": "Place a slice of cooked potato onto the counter.",
        "high_descs": "Turn right, move to the table. Pick up the knife from the table. Slice the potato on the table. Turn left, move to the counter left of the bread. Put the knife on the counter near the soap. Turn left, move to the table. Pick up a slice of potato from the table. Turn left, move to the counter. Put the potato slice into the microwave, cook it, pick it back up. Turn right, move to the counter left of the bread. Put the cooked potato slice on the counter."
        "actions": "LookDown MoveAhead RotateRight MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead RotateLeft MoveAhead MoveAhead PickupObject RotateLeft RotateLeft MoveAhead RotateRight MoveAhead MoveAhead MoveAhead MoveAhead MoveAhead RotateRight PutObject",
        "objectid": "table, knife, potato, counter, bread, soap, microwave",
        "nodes_info": [
            {
                "id": "table"
            },
            {
                "id": "knife"
            },
            {
                "id": "potato"
            },
            {
                "id": "counter"
            },
            {
                "id": "bread"
            },
            {
                "id": "soap"
            },
            {
                "id": "microwave"
            }
        ],
        "links_info": [
            {
                "source": "knife",
                "target": "potato",
                "relationship": "slice"
            },
            {
                "source": "table",
                "target": "potato",
                "relationship": "on"
            },
            {
                "source": "microwave",
                "target": "potato",
                "relationship": "heat"
            },
            {
                "source": "microwave",
                "target": "counter",
                "relationship": "on"
            },

            {
                "source": "counter",
                "target": "bread",
                "relationship": "left"
            },
            {
                "source": "soap",
                "target": "counter",
                "relationship": "near"
            }
        ]
    },
    ...
}
```

So far, we have successfully expanded the ALFRED dataset and successfully generated a new dataset GATP. This new dataset essentially adds scene graph and task graph data on the basis of ALFRED.

