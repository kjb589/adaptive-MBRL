import sys
import math
#sys.path.append(r'C:/Users/cgoodin/Desktop/vm_shared/shared_repos/mavs/src/mavs_python')
sys.path.append(r'C:\Users\mahfu\mavs\src\mavs_python')
import mavs_interface as mavs
import mavs_python_paths
mavs_data_path = mavs_python_paths.mavs_data_path

class TrafficLight(object):
    def __init__(self, yellow_pos, direction, vert_offset):
        self.current_state = 'green'
        green_pos = [yellow_pos[0],yellow_pos[1],yellow_pos[2]-vert_offset]
        red_pos = [yellow_pos[0],yellow_pos[1],yellow_pos[2]+vert_offset]
        self.light_positions = {'green':green_pos,'yellow':yellow_pos,'red':red_pos}
        self.light_direction = direction
        self.light_id = -1 # this is an invalid ID, must be set when you create the light
    def AddToScene(self, loc_env):
        self.light_id = loc_env.AddSpotLight([255,255,255],self.light_positions[self.current_state],self.light_direction,35) 
    def CycleState(self,loc_env):
        if self.current_state=='green':
            self.current_state='yellow'
        elif self.current_state=='yellow':
            self.current_state='red'
        elif self.current_state=='red':
            self.current_state = 'green'
        loc_env.MoveLight(self.light_id, self.light_positions[self.current_state],self.light_direction)

class FourWayIntersection(object):
    def __init__(self, intersection_offset, positions, directions, offsets):
        """
        Lights must be listed going clockwise around the intersection
        such that lights 0 and 2 are paired and 1 and 3 are paired
        """
        self.lights = []
        for i in range(4):
            pos = [positions[i][0]+intersection_offset[0], positions[i][1]+intersection_offset[1], 
                   positions[i][2]+intersection_offset[2]]
            self.lights.append(TrafficLight(pos,directions[i],offsets[i]))
        for i in range(4):
            if (i%2==0):
                self.lights[i].current_state = 'green'
            else:
                self.lights[i].current_state = 'red'
        self.elapsed_time = 0.0
        self.cycle_time = 60.0
        self.yellow_time = 5.0
        self.position = intersection_offset
        self.time_to_change = 0.0
    def GetDistanceToIntersection(self, p):
        dx = p[0] - self.position[0]
        dy = p[1] - self.position[1]
        d = math.sqrt(dx*dx + dy*dy)
        return d
    def GetClosestLightState(self,p):
        closest = -1
        d = 1.0E9
        for i in range(4):
            dx = p[0] - self.lights[i].light_positions['yellow'][0]
            dy = p[1] - self.lights[i].light_positions['yellow'][1]
            d_i = math.sqrt(dx*dx + dy*dy)
            if d_i<d:
                d = d_i
                closest = i
        change_to = 'green'
        if (self.lights[closest].current_state=='green'):
            change_to = 'yellow'
        elif (self.lights[closest].current_state=='yellow'):
            change_to = 'red'
        else:
            change_to = 'green'
        stat_msg = "Light is "+self.lights[closest].current_state+" with "+str(self.time_to_change)+" seconds until change to "+change_to
        return stat_msg
    def AddToScene(self,loc_env):
        for i in range(4):
            self.lights[i].AddToScene(loc_env)

    def Update(self,loc_env, dt):
        self.elapsed_time = self.elapsed_time + dt
        cycle = False
        for i in range(4):
            if (self.elapsed_time>=(self.cycle_time-self.yellow_time) and self.lights[i].current_state=='green'):
                self.lights[i].CycleState(loc_env)
            else:
                self.time_to_change = (self.cycle_time-self.yellow_time)-self.elapsed_time
            if (self.elapsed_time>=self.cycle_time):
                self.lights[i].CycleState(loc_env)
                cycle = True
            else:
                self.time_to_change = self.cycle_time - self.elapsed_time
        if cycle:
            self.elapsed_time = 0.0

if __name__=='__main__':
    scene = mavs.MavsEmbreeScene()
    env = mavs.MavsEnvironment()
    mavs_scenefile = "/scenes/four_way_scene.json"
    scene.Load(mavs_data_path+mavs_scenefile)

    env.SetScene(scene)

    
    light_positions = [[-6.9,-2.13,6.17],[2.13,-6.9, 6.17],[6.9,2.13,6.17],[-2.13,6.9, 6.17]]
    light_directions = [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]
    light_offsets = [0.23,0.23,0.23,0.23]

    center_intersection_offset = [0.0, 0.0, 0.0]
    center_four_way = FourWayIntersection(center_intersection_offset, light_positions, light_directions, light_offsets)
    center_four_way.AddToScene(env)
    center_four_way.cycle_time = 10.0

    north_intersection_offset = [0.0, 50.0, 0.0]
    north_four_way = FourWayIntersection(north_intersection_offset, light_positions, light_directions, light_offsets)
    north_four_way.AddToScene(env)
    north_four_way.cycle_time = 10.0

    south_intersection_offset = [0.0, -50.0, 0.0]
    south_four_way = FourWayIntersection(south_intersection_offset, light_positions, light_directions, light_offsets)
    south_four_way.AddToScene(env)
    south_four_way.cycle_time = 10.0

    cam = mavs.MavsCamera()
    cam.Initialize(384,384,0.0035,0.0035,0.0035)
    cam.SetGammaAndGain(0.75,1.0)
    cam.SetPose([0, 0,2], [1.0, 0.0, 0.0, 0.0])

    cam.FreePose()

    dt = 1.0/10.0
    while (True):

        # Get the current camera position
        position, orientation = cam.GetPose()
        cam.SetPose(position,orientation)
        cam.Update(env,dt)
        cam.Display()
    
        center_four_way.Update(env, dt)
        north_four_way.Update(env, dt)
        south_four_way.Update(env, dt)
