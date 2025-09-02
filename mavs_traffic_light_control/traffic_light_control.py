import time
import sys
import math
import keyboard
sys.path.append(r'C:/Users/cgoodin/Desktop/vm_shared/shared_repos/mavs/src/mavs_python')
import mavs_interface as mavs
import mavs_python_paths
mavs_data_path = mavs_python_paths.mavs_data_path
from Intersection import FourWayIntersection

scene = mavs.MavsEmbreeScene()
env = mavs.MavsEnvironment()
mavs_scenefile = "/scenes/four_way_scene.json"
scene.Load(mavs_data_path+mavs_scenefile)

env.SetScene(scene)

# Set environment properties
env.SetTime(12) # 0-23
env.SetFog(1.0) # 0.0-100.0
env.SetSnow(0.0) # 0-25
env.SetTurbidity(9.0) # 2-10
env.SetAlbedo(0.4) # 0-1
env.SetCloudCover(0.0) # 0-1
env.SetRainRate(0.0) # 0-25
env.SetWind( [2.5, 1.0] ) # Horizontal windspeed in m/s

# create the intersection and add it to the environment
intersection_offset = [0.0, 0.0, 0.0]
light_positions = [[-6.9,-2.13,6.17],[2.13,-6.9, 6.17],[6.9,2.13,6.17],[-2.13,6.9, 6.17]]
light_directions = [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]
light_offsets = [0.23,0.23,0.23,0.23]

four_way = FourWayIntersection(intersection_offset, light_positions, light_directions, light_offsets)
four_way.AddToScene(env)
four_way.cycle_time = 10.0
four_way.yellow_time = 3.0

cam = mavs.MavsCamera()
cam.Initialize(384,384,0.0035,0.0035,0.0035)
cam.SetGammaAndGain(0.75,1.0)
cam.SetPose([0, 0,2], [1.0, 0.0, 0.0, 0.0])
cam.FreePose()

drive_cam = mavs.MavsCamera()
drive_cam.Initialize(256,256,0.0035,0.0035,0.0035)
drive_cam.SetGammaAndGain(0.75,1.0)
drive_cam.SetOffset([-7.5, 0,2], [1.0, 0.0, 0.0, 0.0])
#drive_cam.FreePose()

sf = 2
hd_cam = mavs.MavsCamera()
hd_cam.Initialize(int(1920/sf),int(1080/sf),(1920.0/1080.0)*0.0035, 0.0035,0.0035)
hd_cam.SetGammaAndGain(0.75,2.0)
hd_cam.RenderShadows(True)
hd_cam.SetAntiAliasingFactor(5)

veh = mavs.MavsRp3d()
veh_file = 'forester_2017_rp3d_tires.json'
veh.Load(mavs_data_path+'/vehicles/rp3d_vehicles/' + veh_file)
veh.SetInitialPosition(-50.0, 50.0, 0.0) # in global ENU
veh.SetInitialHeading(0.0) # in radians

controller = mavs.MavsVehicleController()
controller.SetDesiredPath([
[	-50	,	50	],
[	-48	,	50	],
[	0	,	50	],
[	48	,	50	],
[	50	,	50	],
[	50	,	48	],
[	50	,	0	],
[	50	,	-48	],
[	50	,	-50	],
[	48	,	-50	],
[	0	,	-50	],
[	-48	,	-50	],
[	-50	,	-50	],
[	-50	,	-48	],
[	-50	,	0	],
[	-50	,	48	]])
controller.SetDesiredSpeed(5.0) # m/s 
controller.SetSteeringScale(12.0)
controller.SetWheelbase(3.3) # meters
controller.SetMaxSteerAngle(0.855) # radians
controller.TurnOnLooping()

frame_num = 0
dt = 1.0/100.0
saving = False
while (True):
    # tw0 is for timing purposes used later
    tw0 = time.time()

    #controller.SetCurrentState(veh.GetPosition()[0],veh.GetPosition()[1],
    #                               veh.GetSpeed(),veh.GetHeading())
    #dc = controller.GetDrivingCommand(dt)
    dc = drive_cam.GetDrivingCommand()
    # Update the vehicle
    veh.Update(env, dc.throttle, dc.steering, dc.braking, dt)

    # Get the current camera position
    position, orientation = cam.GetPose()

    if keyboard.is_pressed('p'):
        # get the current pose
        print(position,orientation)
        sys.stdout.flush()

    if (frame_num%10==0):
        #cam.SetPose(position,orientation)
        #cam.Update(env,10*dt)
        #cam.Display()
        drive_cam.SetPose(veh.GetPosition(),veh.GetOrientation())
        drive_cam.Update(env,10*dt)
        drive_cam.Display()
    
    four_way.Update(env, dt)
    d = four_way.GetDistanceToIntersection(veh.GetPosition())
    if (d<10.0):
        print("Intersection state = ",four_way.GetClosestLightState(veh.GetPosition()))
        sys.stdout.flush()
    
    if keyboard.is_pressed('l'):
        saving = not saving
        if saving:
            print('Saving frames...')
            sys.stdout.flush()
        else:
            print('Stopping save frames.')
            sys.stdout.flush()
    
    if saving:
        cam.SaveCameraImage('frame_'+str(frame_num).zfill(4)+'.bmp')

    if keyboard.is_pressed('r'):
        print("Rendering hi-res frame...")
        sys.stdout.flush()
        hd_cam.SetPose(position,orientation)
        hd_cam.Update(env,dt)
        hd_cam.SaveCameraImage('image_'+str(frame_num).zfill(4)+'.bmp')
        print("Rendered frame ",frame_num," in ",(time.time()-tw0)," seconds.")
        sys.stdout.flush()
        
    if keyboard.is_pressed('f'):
        print("Frame rate = ",1.0/(time.time()-tw0))
        sys.stdout.flush()

    frame_num = frame_num + 1