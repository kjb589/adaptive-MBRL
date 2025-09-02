# import the MAVS simulation loader
from mavs_spav_simulation import MavsSpavSimulation
# Import the additional autonomy modules
import autonomy
import astar
import rl_planner
# Import some python functions we'll need
from math import sqrt
from train_planner import train_step
import time
import numpy as np
import pickle

# Create a buffer to store (state, action, next_state) tuples
dataset = []

# Create and occupancy grid and resize it
grid = autonomy.OccupancyGrid()
grid.resize(400,400)
# Set the resolution (in meters) of the grid
grid.info.resolution = 1.0
# Set the origin of the grid (lower left corner)
grid.set_origin(-200.0,-200.0)
# Define the goal point on our map and convert it to a grid index
goal_points = [
    [62.75, 7.5],  # original goal
    [62.75, -35],
    [10.0, -35],
    [-50.0, -35],
    [-50.0, 0.0]
]

# Create the simulation
sim = MavsSpavSimulation()

# Start the main simulation loop
dt = 1.0/30.0 # time step, seconds
prev_state = None
prev_action = None
for goal_point in goal_points:
    goal_index = grid.coordinate_to_index(goal_point[0], goal_point[1])
    dist_to_goal = 1000.0
    n=0

    while dist_to_goal > 4.0:
        tw0 = time.time()

        # Get current vehicle state (z): position, heading, speed
        position = sim.veh.GetPosition()
        orientation = sim.veh.GetOrientation()
        heading = sim.veh.GetHeading()
        speed = sim.veh.GetSpeed()
        curr_state = np.array([position[0], position[1], heading, speed], dtype=np.float32)

        # Update controller to get action
        sim.controller.SetCurrentState(position[0], position[1], speed, heading)
        dc = sim.controller.GetDrivingCommand(dt)

        # Get action vector (a): throttle, steering, braking
        action = np.array([dc.throttle, dc.steering, dc.braking], dtype=np.float32)

        # Update vehicle state
        sim.veh.Update(sim.env, dc.throttle, dc.steering, dc.braking, dt)

        # Compute next_state (for supervised learning) as the state after this update
        next_position = sim.veh.GetPosition()
        next_heading = sim.veh.GetHeading()
        next_speed = sim.veh.GetSpeed()
        next_state = np.array([next_position[0], next_position[1], next_heading, next_speed], dtype=np.float32)

        # Save (z, a, z_next) tuple
        dataset.append((curr_state, action, next_state))

        # Environment updates
        sim.env.AdvanceTime(dt)
        dist_to_goal = sqrt(pow(next_position[0]-goal_point[0],2)+pow(next_position[1]-goal_point[1],2))

        # Path planning + lidar logic
        if n % 10 == 0 and n > 0:
            # Update and display the drive camera
            # which is for visualization purposes only.
            sim.drive_cam.SetPose(position, orientation)
            sim.drive_cam.Update(sim.env, dt)
            sim.drive_cam.Display()

            # Update and display the lidar, which will be used
            # by the A* algorithm
            sim.lidar.SetPose(position, orientation)
            sim.lidar.Update(sim.env, dt)
            sim.lidar.Display()

            # Get lidar point cloud registered to world coordinates
            registered_points = sim.lidar.GetPoints()

            # Downsample LiDAR points
            if len(registered_points) > 2000:
                registered_points = registered_points[::10]  # Downsample by factor of 10

            # add the points to the grid
            grid.add_registered_points(registered_points)
            # determine the grid index of the current vehicle location
            current_grid_index = grid.coordinate_to_index(position[0], position[1])
            # calculate a path through the occupancy grid using A*
            path = astar.astar(grid.data, (current_grid_index[0], current_grid_index[1]), (goal_index[0], goal_index[1]))
            # if the path is valid, set it as the new path for the controller
            if path:
                # first convert it back to ENU coordinates
                path_enu = grid.index_path_to_coordinates(path)
                # Update the controller path
                sim.controller.SetDesiredPath(path_enu)

        n += 1
        print(n)
        # timeout condition
        if n > 2000:
            print("Episode timeout, skipping to next goal")
            break

        wall_dt = time.time() - tw0
        if wall_dt < dt:
            time.sleep(dt - wall_dt)

# After simulation ends: save dataset to disk
with open("dynamics_data.pkl", "wb") as f:
    pickle.dump(dataset, f)

print(f"Saved {len(dataset)} samples to dynamics_data.pkl.")
