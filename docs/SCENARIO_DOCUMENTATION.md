# MDriveBench Scenario System Documentation

This document provides comprehensive documentation for the scenario definition and execution system in MDriveBench (CoLMDriver). It covers how scenarios are defined, all available configuration parameters, and how they integrate into the CARLA simulation pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Scenario Components](#scenario-components)
4. [Route Definition (XML)](#route-definition-xml)
5. [Scenario Definition (JSON)](#scenario-definition-json)
6. [Custom Actors System](#custom-actors-system)
7. [Actor Manifest](#actor-manifest)
8. [Weather Configuration](#weather-configuration)
9. [Pipeline Integration](#pipeline-integration)
10. [Complete Examples](#complete-examples)
11. [Advanced Usage](#advanced-usage)

---

## Overview

The MDriveBench scenario system consists of three main components:

1. **Routes**: Define waypoint trajectories for ego vehicles (XML format)
2. **Scenarios**: Define dynamic events/challenges along routes (JSON format)
3. **Custom Actors**: Define additional NPC vehicles, pedestrians, and static objects (XML + JSON manifest)

These components work together to create complete driving scenarios in CARLA.

---

## Directory Structure

```
simulation/leaderboard/data/
├── scenarios/                    # Scenario trigger definitions
│   ├── no_scenarios.json        # Empty scenario file (free driving)
│   └── town05_all_scenarios.json
├── CustomRoutes/                 # Custom route definitions
│   ├── A/                       # Route collection A
│   │   ├── actors_manifest.json # Actor definitions for routes
│   │   ├── actors_behavior.json # Triggered actor behaviors (optional)
│   │   ├── town05_custom_ego_vehicle_0.xml
│   │   ├── town05_custom_ego_vehicle_1.xml
│   │   └── actors/
│   │       └── static/
│   │           └── town05_custom_Truck_1_static.xml
│   ├── B/
│   └── ...
├── CustomRoutes_codriving/       # Model-specific routes
├── CustomRoutes_tcp/
├── Interdrive/                   # InterDrive benchmark routes
└── training_routes/              # Training data routes
```

---

## Scenario Components

### 1. Routes (Ego Vehicle Paths)

Routes define the trajectory that ego vehicles should follow. They are specified in XML format.

### 2. Scenario Triggers

Scenarios are challenge events (e.g., vehicle cut-in, pedestrian crossing) that occur at specific locations along routes. They are defined in JSON format.

### 3. Custom Actors

Additional vehicles, pedestrians, or static objects placed in the world, defined via XML files and referenced by a JSON manifest.

---

## Route Definition (XML)

### Basic Structure

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="247" town="Town05" role="ego">
    <waypoint x="-155.802" y="91.388" z="0.0" yaw="0.071620" pitch="360.0" roll="0.0" />
    <waypoint x="-143.802" y="91.402" z="0.0" yaw="0.067980" />
    <waypoint x="-133.688" y="91.414" z="0.0" yaw="64.160572" />
    <!-- More waypoints -->
  </route>
</routes>
```

### Route Attributes

#### `<route>` Element

| Attribute | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `id` | string | **Yes** | Unique route identifier | `"247"` |
| `town` | string | **Yes** | CARLA town/map name | `"Town05"` |
| `role` | string | **Yes** | Actor role type | `"ego"`, `"npc"`, `"static"` |
| `model` | string | No | Vehicle blueprint ID (for non-ego) | `"vehicle.tesla.model3"` |
| `speed` | float | No | Target speed in m/s | `8.0` |
| `length` | float | No | Vehicle length (for static objects) | `"12"` |
| `width` | float | No | Vehicle width (for static objects) | `"3"` |

#### `<waypoint>` Element

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `x` | float | **Yes** | - | X coordinate in meters |
| `y` | float | **Yes** | - | Y coordinate in meters |
| `z` | float | **Yes** | - | Z coordinate (height) in meters |
| `yaw` | float | **Yes** | - | Yaw rotation in degrees (0-360) |
| `pitch` | float | No | `0.0` | Pitch rotation in degrees |
| `roll` | float | No | `0.0` | Roll rotation in degrees |
| `connection` | string | No | - | Road option hint (for route configs) |

### Weather in Routes (Optional)

Routes can optionally define weather conditions:

```xml
<route id="1" town="Town01">
  <weather 
    cloudiness="50.0" 
    precipitation="30.0" 
    precipitation_deposits="20.0"
    wind_intensity="10.0"
    sun_azimuth_angle="180.0"
    sun_altitude_angle="45.0"
    fog_density="10.0"
    fog_distance="50.0"
    fog_falloff="1.0"
    wetness="40.0" 
  />
  <waypoint ... />
</route>
```

#### Weather Attributes

| Attribute | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `cloudiness` | float | 0-100 | 30 | Cloud coverage percentage |
| `precipitation` | float | 0-100 | 0 | Rain intensity |
| `precipitation_deposits` | float | 0-100 | 0 | Puddle accumulation |
| `wind_intensity` | float | 0-100 | 0.35 | Wind strength |
| `sun_azimuth_angle` | float | 0-360 | 0 | Sun horizontal position |
| `sun_altitude_angle` | float | -90 to 90 | 70 | Sun vertical position (negative = night) |
| `fog_density` | float | 0-100 | 0 | Fog thickness |
| `fog_distance` | float | 0+ | 0 | Fog start distance in meters |
| `fog_falloff` | float | 0+ | 1.0 | Fog density falloff |
| `wetness` | float | 0-100 | 0 | Road wetness |

### Preset Weather

Alternatively, use predefined weather conditions:

```xml
<route id="1" town="Town01" weather="5">
  <!-- weather="5" corresponds to WetNoon -->
</route>
```

**Preset Weather IDs:**
- `1`: ClearNoon
- `2`: ClearSunset
- `3`: CloudyNoon
- `4`: CloudySunset
- `5`: WetNoon
- `6`: WetSunset
- `7`: MidRainyNoon
- `8`: MidRainSunset
- `9`: WetCloudyNoon
- `10`: WetCloudySunset
- `11`: HardRainNoon
- `12`: HardRainSunset
- `13`: SoftRainNoon
- `14`: SoftRainSunset

---

## Scenario Definition (JSON)

Scenarios define challenging events that can be triggered at specific locations along routes.

### Basic Structure

```json
{
  "available_scenarios": [
    {
      "Town05": [
        {
          "available_event_configurations": [
            {
              "transform": {
                "x": 151.37,
                "y": -26.18,
                "z": 0.0,
                "yaw": 88.0,
                "pitch": 0.0
              },
              "other_actors": {
                "front": [
                  {
                    "x": 155.0,
                    "y": -26.0,
                    "z": 0.5,
                    "yaw": 90.0,
                    "model": "vehicle.tesla.model3",
                    "speed": 5.0
                  }
                ],
                "left": [],
                "right": []
              }
            }
          ],
          "scenario_type": "Scenario4"
        }
      ]
    }
  ]
}
```

### JSON Structure Breakdown

#### Root Level

```json
{
  "available_scenarios": [...]  // Array of town-scenario mappings
}
```

#### Town Level

Each entry maps a town to its scenarios:

```json
{
  "Town05": [
    { scenario_1 },
    { scenario_2 }
  ]
}
```

#### Scenario Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scenario_type` | string | **Yes** | Type of scenario challenge |
| `available_event_configurations` | array | **Yes** | Possible trigger locations |

#### Event Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `transform` | object | **Yes** | Scenario trigger position |
| `other_actors` | object | No | NPC actors for this scenario |

#### Transform Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x` | float | **Yes** | X coordinate |
| `y` | float | **Yes** | Y coordinate |
| `z` | float | **Yes** | Z coordinate |
| `yaw` | float | **Yes** | Yaw angle (degrees) |
| `pitch` | float | No | Pitch angle |

#### Other Actors Object

```json
{
  "other_actors": {
    "front": [actor_list],
    "left": [actor_list],
    "right": [actor_list]
  }
}
```

Each actor in the list:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `x` | float | **Yes** | - | X position |
| `y` | float | **Yes** | - | Y position |
| `z` | float | **Yes** | - | Z position |
| `yaw` | float | **Yes** | - | Yaw angle |
| `model` | string | No | `"vehicle.*"` | CARLA blueprint ID |
| `speed` | float | No | `8.0` | Target speed in m/s |

### Scenario Types

The system supports these predefined scenario types:

| Scenario | Class | Description |
|----------|-------|-------------|
| `Scenario1` | `ControlLoss` | Ego vehicle loses control |
| `Scenario2` | `FollowLeadingVehicle` | Follow a slow leading vehicle |
| `Scenario3` | `DynamicObjectCrossing` | Object crosses ego's path |
| `Scenario4` | `VehicleTurningRoute` | Vehicle turns at intersection |
| `Scenario5` | `OtherLeadingVehicle` | Another vehicle leads |
| `Scenario6` | `ManeuverOppositeDirection` | Oncoming traffic scenario |
| `Scenario7` | `SignalJunctionCrossingRoute` | Signalized intersection crossing |
| `Scenario8` | `SignalJunctionCrossingRoute` | Signalized intersection (variant) |
| `Scenario9` | `SignalJunctionCrossingRoute` | Signalized intersection (variant) |
| `Scenario10` | `NoSignalJunctionCrossingRoute` | Unsignalized intersection |

### Empty Scenarios

For free driving without scenario challenges:

```json
{
  "available_scenarios": [
    {
      
    }
  ]
}
```

Or use `data/scenarios/no_scenarios.json`.

---

## Custom Actors System

Custom actors allow you to place specific vehicles, pedestrians, or static objects along routes.

### Actor XML Files

#### Ego Vehicle

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="247" town="Town05" role="ego">
    <waypoint x="-155.802" y="91.388" z="0.0" yaw="0.071620" />
    <waypoint x="-143.802" y="91.402" z="0.0" yaw="0.067980" />
    <!-- More waypoints -->
  </route>
</routes>
```

#### NPC Vehicle

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="247" town="Town05" role="npc" 
         model="vehicle.tesla.model3" speed="8.0">
    <waypoint x="-150.0" y="90.0" z="0.0" yaw="0.0" />
    <waypoint x="-140.0" y="90.0" z="0.0" yaw="0.0" />
  </route>
</routes>
```

#### Static Object

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="247" town="Town05" role="static" 
         model="vehicle.carlamotors.carlacola" 
         length="12" width="3">
    <waypoint x="-130.928" y="89.262" z="0.0" yaw="0.0" />
  </route>
</routes>
```

#### Pedestrian

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="247" town="Town05" role="pedestrian" 
         model="walker.pedestrian.0001" speed="1.5">
    <waypoint x="-145.0" y="95.0" z="0.0" yaw="90.0" />
    <waypoint x="-145.0" y="85.0" z="0.0" yaw="90.0" />
  </route>
</routes>
```

### Actor Roles and Defaults

| Role | Default Model | Default Speed (m/s) | Description |
|------|---------------|---------------------|-------------|
| `ego` | `vehicle.lincoln.mkz2017` | - | Player-controlled vehicle |
| `npc` | `vehicle.tesla.model3` | 8.0 | AI-driven vehicle |
| `pedestrian` | `walker.pedestrian.0001` | 1.5 | Walking pedestrian |
| `bicycle` | `vehicle.diamondback.century` | 4.0 | Bicycle |
| `bike` | `vehicle.diamondback.century` | 4.0 | Motorcycle/bike |
| `static` | `static.prop.trafficcone01` | 0.0 | Static obstacle |
| `static_prop` | `static.prop.trafficcone01` | 0.0 | Static prop |

---

## Actor Manifest

The `actors_manifest.json` file connects actor XML files to specific routes.

### Manifest Structure

```json
{
  "ego": [
    {
      "file": "town05_custom_ego_vehicle_0.xml",
      "route_id": "247",
      "town": "Town05",
      "name": "town05_custom_ego_vehicle_0",
      "kind": "ego",
      "model": "vehicle.lincoln.mkz2017"
    }
  ],
  "npc": [
    {
      "file": "actors/npc/town05_custom_npc_1.xml",
      "route_id": "247",
      "town": "Town05",
      "name": "town05_custom_npc_1",
      "kind": "npc",
      "model": "vehicle.tesla.model3",
      "speed": 10.0,
      "avoid_collision": false
    }
  ],
  "static": [
    {
      "file": "actors/static/town05_custom_Truck_1_static.xml",
      "route_id": "247",
      "town": "Town05",
      "name": "town05_custom_Truck_1_static",
      "kind": "static",
      "model": "vehicle.carlamotors.carlacola",
      "length": "12",
      "width": "3"
    }
  ],
  "pedestrian": [
    {
      "file": "actors/pedestrians/pedestrian_1.xml",
      "route_id": "247",
      "town": "Town05",
      "name": "pedestrian_crossing",
      "kind": "pedestrian",
      "model": "walker.pedestrian.0007",
      "speed": 2.0
    }
  ]
}
```

### Manifest Entry Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | **Yes** | Relative path to actor XML file |
| `route_id` | string | **Yes** | Route ID this actor belongs to |
| `town` | string | **Yes** | Town/map name |
| `name` | string | **Yes** | Unique actor identifier |
| `kind` | string | **Yes** | Actor type (`ego`, `npc`, `static`, `pedestrian`) |
| `model` | string | No | CARLA blueprint override |
| `speed` | float | No | Speed override (m/s) |
| `avoid_collision` | boolean | No | Whether NPC avoids collisions |
| `length` | string | No | Vehicle length (for static) |
| `width` | string | No | Vehicle width (for static) |

---

## Actor Behaviors (Optional)

Use `actors_behavior.json` to define trigger-based behaviors for custom actors.

### Behavior File Structure

```json
{
  "behaviors": [
    {
      "route_id": "247",
      "actor_name": "pedestrian_crossing",
      "trigger": {
        "type": "distance_to_vehicle",
        "vehicle": "Vehicle 1",
        "distance_m": 8.0,
        "evidence": "when Vehicle 1 gets close"
      },
      "action": {
        "type": "start_motion",
        "direction": null,
        "target_vehicle": null
      }
    }
  ]
}
```

### Trigger Types

- `distance_to_vehicle`: Trigger when the actor is within `distance_m` of an ego vehicle.

### Action Types

- `start_motion`: Begin following the actor's waypoint plan (e.g., pedestrian starts crossing).
- `hard_brake`: NPC vehicle brakes to a stop.
- `lane_change`: NPC vehicle changes lane.
  - Use `direction`: `"left"` or `"right"` when specified.
  - Use `target_vehicle`: `"Vehicle 1"` to request a lane change into that ego vehicle's lane (best effort).

---

## Weather Configuration

Weather can be configured in three ways:

### 1. In Route XML (Per-Route)

```xml
<route id="1" town="Town01">
  <weather cloudiness="50.0" precipitation="30.0" />
  <waypoint ... />
</route>
```

### 2. Preset Weather ID (In Route XML)

```xml
<route id="1" town="Town01" weather="11">
  <!-- HardRainNoon -->
</route>
```

### 3. Default Weather

If no weather is specified, default is:
```python
carla.WeatherParameters(sun_altitude_angle=70, cloudiness=30)
```

---

## Pipeline Integration

### How Scenarios Flow Through the System

```
1. Shell Script (eval_driving.sh)
   ↓
   Sets environment variables:
   - ROUTES_DIR: Directory containing routes
   - SCENARIOS: Path to scenario JSON file
   - CUSTOM_ACTOR_MANIFEST: Path to actors_manifest.json
   
2. Leaderboard Evaluator (leaderboard_evaluator.py)
   ↓
   - Loads route XML files
   - Parses scenario JSON
   - Initializes CARLA world
   
3. Route Parser (route_parser.py)
   ↓
   - parse_routes_file(): Reads route XML
   - parse_annotations_file(): Reads scenario JSON
   - _build_custom_actor_configs(): Loads custom actors
   
4. Route Scenario (route_scenario.py)
   ↓
   - Interpolates waypoints
   - Matches scenarios to route positions
   - Instantiates scenario classes
   
5. Scenario Manager (scenario_manager.py)
   ↓
   - Manages scenario lifecycle
   - Triggers scenarios at correct positions
   
6. CARLA Execution
   ↓
   - Spawns actors
   - Executes behaviors
   - Evaluates criteria
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ROUTES_DIR` | Directory with route XML files | `simulation/leaderboard/data/CustomRoutes/A` |
| `ROUTES` | Specific route file path | `data/CustomRoutes/A/town05_custom_ego_vehicle_0.xml` |
| `SCENARIOS` | Path to scenario JSON | `data/scenarios/town05_all_scenarios.json` |
| `CUSTOM_ACTOR_MANIFEST` | Path to actor manifest | `data/CustomRoutes/A/actors_manifest.json` |
| `CUSTOM_ACTOR_BEHAVIORS` | Path to actor behavior file | `data/CustomRoutes/A/actors_behavior.json` |
| `CUSTOM_ACTOR_DEFAULT_SPEED` | Default actor speed (m/s) | `8.0` |

### Running a Scenario

```bash
export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=simulation/leaderboard
export SCENARIO_RUNNER_ROOT=simulation/scenario_runner

# Set scenario paths
export ROUTES_DIR=${LEADERBOARD_ROOT}/data/CustomRoutes/A
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town05_all_scenarios.json
export CUSTOM_ACTOR_MANIFEST=${ROUTES_DIR}/actors_manifest.json
export CUSTOM_ACTOR_BEHAVIORS=${ROUTES_DIR}/actors_behavior.json

# Run evaluation
python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --routes_dir=${ROUTES_DIR} \
  --scenarios=${SCENARIOS} \
  --agent=simulation/leaderboard/team_code/your_agent.py \
  --agent-config=config.yaml \
  --port=2000
```

---

## Complete Examples

### Example 1: Simple Route with Weather

**File:** `custom_route.xml`

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="100" town="Town05">
    <weather cloudiness="80.0" precipitation="50.0" wetness="70.0" />
    <waypoint x="-100.0" y="50.0" z="0.0" yaw="90.0" />
    <waypoint x="-100.0" y="100.0" z="0.0" yaw="90.0" />
    <waypoint x="-100.0" y="150.0" z="0.0" yaw="90.0" />
  </route>
</routes>
```

### Example 2: Route with NPC Interactions

**Manifest:** `actors_manifest.json`

```json
{
  "ego": [
    {
      "file": "ego_route.xml",
      "route_id": "200",
      "town": "Town05",
      "name": "ego_vehicle"
    }
  ],
  "npc": [
    {
      "file": "actors/npc_vehicle.xml",
      "route_id": "200",
      "town": "Town05",
      "name": "leading_vehicle",
      "model": "vehicle.audi.a2",
      "speed": 5.0
    }
  ]
}
```

**NPC File:** `actors/npc_vehicle.xml`

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="200" town="Town05" role="npc">
    <waypoint x="-100.0" y="60.0" z="0.0" yaw="90.0" />
    <waypoint x="-100.0" y="120.0" z="0.0" yaw="90.0" />
  </route>
</routes>
```

### Example 3: Static Obstacle Scenario

**Scenario JSON:**

```json
{
  "available_scenarios": [
    {
      "Town05": [
        {
          "available_event_configurations": [
            {
              "transform": {
                "x": -100.0,
                "y": 80.0,
                "z": 0.0,
                "yaw": 90.0
              }
            }
          ],
          "scenario_type": "Scenario3"
        }
      ]
    }
  ]
}
```

**Static Actor XML:**

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="200" town="Town05" role="static" 
         model="static.prop.container">
    <waypoint x="-98.0" y="80.0" z="0.0" yaw="0.0" />
  </route>
</routes>
```

### Example 4: Multi-Agent Scenario

For multi-agent scenarios, create separate ego vehicle routes:

**actors_manifest.json:**

```json
{
  "ego": [
    {
      "file": "ego_vehicle_0.xml",
      "route_id": "300",
      "town": "Town05",
      "name": "ego_0",
      "model": "vehicle.lincoln.mkz2017"
    },
    {
      "file": "ego_vehicle_1.xml",
      "route_id": "300",
      "town": "Town05",
      "name": "ego_1",
      "model": "vehicle.tesla.model3"
    }
  ]
}
```

---

## Advanced Usage

### Custom Scenario Parameters

Some scenarios support additional parameters via YAML configuration files.

**File:** `scenario_parameter_custom.yaml`

```yaml
scenario_configs:
  Scenario3:
    proportion: 0.3
    dynamic_object_type: "vehicle"
  Scenario4:
    proportion: 0.4
    turn_direction: "left"
  Scenario7:
    proportion: 0.3
    traffic_light_color: "red"
```

Set via environment variable:

```bash
export SCENARIOS_PARAMETER=leaderboard/scenarios/scenario_parameter_custom.yaml
```

### Trigger Distance

Control when scenarios activate relative to ego position:

```bash
python leaderboard_evaluator.py \
  --trigger_distance 30.0  # Activate scenarios 30m ahead
```

### Scenario Filtering

The system automatically:
- Matches scenario triggers to route waypoints (within 2m threshold)
- Filters scenarios by route direction (e.g., only left-turn scenarios on left turns)
- Prevents duplicate scenarios at same location

### Debug Mode

Enable detailed scenario logging:

```bash
python leaderboard_evaluator.py \
  --debug 2  # Verbose debugging
```

---

## CARLA Blueprint Reference

### Common Vehicle Models

- `vehicle.lincoln.mkz2017` (default ego)
- `vehicle.tesla.model3`
- `vehicle.audi.a2`
- `vehicle.bmw.grandtourer`
- `vehicle.chevrolet.impala`
- `vehicle.dodge.charger_police`
- `vehicle.carlamotors.carlacola` (truck)
- `vehicle.volkswagen.t2`

### Walker Models

- `walker.pedestrian.0001` - `walker.pedestrian.0014`

### Static Props

- `static.prop.trafficcone01`
- `static.prop.container`
- `static.prop.streetbarrier`

Use CARLA documentation for complete blueprint list.

---

## Troubleshooting

### Common Issues

**1. Scenario Not Triggering**

- Check trigger position matches route waypoints (within 2m)
- Verify scenario type is appropriate for route geometry
- Confirm town name matches exactly

**2. Custom Actors Not Spawning**

- Verify `CUSTOM_ACTOR_MANIFEST` environment variable is set
- Check file paths in manifest are relative to manifest location
- Ensure route_id matches between manifest and route file

**3. Weather Not Applied**

- Weather in route XML takes precedence over defaults
- Check XML syntax (no typos in attribute names)
- Weather values must be within valid ranges

**4. Actor Collision at Spawn**

- Increase spacing between waypoints
- Check z-coordinate is at ground level (typically 0.0-0.5)
- Use `avoid_collision` flag for NPCs

---

## Best Practices

1. **Waypoint Spacing**: Keep waypoints 10-20m apart for smooth trajectories
2. **Z-Coordinates**: Always verify ground height (usually 0.0 in CARLA)
3. **Scenario Density**: Don't place too many scenarios too close together
4. **Testing**: Start with `no_scenarios.json` to test routes before adding scenarios
5. **Naming**: Use descriptive, consistent names for routes and actors
6. **Version Control**: Keep scenario definitions in version control
7. **Documentation**: Comment complex multi-actor scenarios

---

## Additional Resources

- **CARLA Documentation**: https://carla.readthedocs.io/
- **Blueprint Library**: Check CARLA's blueprint library in-game
- **Scenario Runner**: `simulation/scenario_runner/` for scenario class implementations
- **Example Routes**: `simulation/leaderboard/data/Interdrive/` for reference

---

## Summary

The MDriveBench scenario system provides flexible control over:
- **Routes**: Ego vehicle trajectories with waypoints
- **Scenarios**: Dynamic challenges triggered at specific locations
- **Actors**: Custom NPCs, pedestrians, and static objects
- **Weather**: Environmental conditions
- **Integration**: Seamless connection to CARLA simulation

By combining these components, you can create complex, realistic driving scenarios for autonomous vehicle evaluation and training.
