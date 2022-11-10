"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12


LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
print("in tmodify")
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)
# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

# range = robot.getDevice('range-finder')
# range.enable(timestep)
# camera = robot.getDevice('camera')
# camera.enable(timestep)
# camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis
map = None
##### ^^^ [End] Do Not Modify ^^^ #####
print("out of do no tmodify")
##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
mode = 'manual' # Part 1.1: manual mode
mode = 'planner'
mode = 'autonomous'




###################
#
# Planner
#
###################
if mode == 'planner':
    print("here")
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    # start_w =(4.66,8.43) #initial position # (Pose_X, Pose_Z) in meters
    # end_w = (10.0, 7.0) # (Pose_X, Pose_Z) in meters
    start_w =(-8.43, -4.66) #initial position # (Pose_X, Pose_Z) in meters
    end_w = (-7, -10.0) # (Pose_X, Pose_Z) in meters
    
    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start =  ( -int(start_w[0]*30), 360 + int(start_w[1]*30))
    end =  ( -int(end_w[0]*30), 360 + int(end_w[1]*30))
    print(start)

    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    
    # https://github.com/ryancollingwood/arcade-rabbit-herder/blob/master/pathfinding/astar.py        
       
    class Node:
        """
        A node class for A* Pathfinding
        """
    
        def __init__(self, parent=None, position=None):
            self.parent = parent
            self.position = position
    
            self.g = 0
            self.h = 0
            self.f = 0
    
        def __eq__(self, other):
            return self.position == other.position
    
    
    def return_path(current_node):
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Return reversed path
    
    
    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
    
        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0
    
        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)
        
        # what squares do we search (diagonal movement not allowed)
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
       
        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node
            current_index = 0
            current_node = open_list[current_index]#think your node is stuck at 0 so cahnged.
            
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
    
            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)
    
            # Found the goal
            if current_node == end_node:
                return return_path(current_node)

            # Generate children
            children = []
            
            for new_position in adjacent_squares:  # Adjacent squares
    
                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
    
                # Make sure within range
                within_range_criteria = [
                    node_position[0] > (len(map) - 1),
                    node_position[0] < 0,
                    node_position[1] > (len(map[len(map) - 1]) - 1),
                    node_position[1] < 0,
                ]
                
                if any(within_range_criteria):
                    continue
    
                # Make sure walkable terrain
                if map[node_position[0]][node_position[1]] != 0:
                    continue
    
                # Create new node
                new_node = Node(current_node, node_position)
    
                # Append
                children.append(new_node)
    
            # Loop through children
            for child in children:
                
                # Child is on the closed list
                if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                    continue
    
                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + \
                          ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h
    
                # Child is already in the open list
                if len([open_node for open_node in open_list if child == open_node and child.g > open_node.g]) > 0:
                    continue
    
                # Add the child to the open list
                open_list.append(child)

        # no list found
        print("no lists found")
        return open_list

    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load("map.npy")
    # plt.imshow(np.fliplr(map))
    # plt.show()
    

    # Part 2.2: Compute an approximation of the “configuration space”
    mapMask = np.zeros((360,360)) 
    for x in range(0,len(map[0]), 1):
        for y in range(0,len(map[1]), 1):
            if map[x][y] == 1:
                for row in range(y-7,y+7):
                    for col in range(x-7,x+7):
                        if row in range(0, 360):
                            if col in range(0,360):
                                mapMask[row][col] = 1
    map = mapMask
    map = np.rot90(map)
    map = np.flipud(map)
    #map = np.fliplr(map)
    plt.imshow(map)
    plt.show()
    
    # Part 2.3 continuation: Call path_planner
    if map[start[0]][start[1]] != 0:
        print("starting on an obstacle silly")
        
    if map[end[0]][end[1]] !=0:
        print("ending on obstical")

    path = path_planner(map, start, end)
    print(path)
    # display.setColor(int(0xFF0000))
    # for i in range (1,len(path)):
        
        # display.drawPixel(path[i-1],path[i])
    waypoints = []
    for node in path:
        x = -(node[0]/30)
        y = -((360 - node[1])/30)
        waypoints.append((x,y))
    print (waypoints)
    for OBJECT in path:
         mapMask[OBJECT[0]][OBJECT[1]] = 255
    plt.imshow(mapMask)
    plt.show()
    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    np.save("path.npy", waypoints)
######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.empty((360,360)) 
waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    
    waypoints = [] # Replace with code to load your path
    waypoints = np.load("path.npy")  
    waypointIndex = 1
    goal= abs(waypoints[1])
    path = waypoints
    state = 0
state = 0 # use this to iterate through your path


while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_y = -gps.getValues()[1]
    pose_x = -gps.getValues()[0]

    n = compass.getValues()
    rad = ((math.atan2(n[0], -n[2])))#-1.5708)
    pose_theta = rad
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = -math.cos(alpha)*rho + 0.202
        ry = math.sin(alpha)*rho -0.004


        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(pose_theta)*rx - math.sin(pose_theta)*ry + pose_x
        wy =  +(math.sin(pose_theta)*rx + math.cos(pose_theta)*ry) + pose_y


    
        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))

        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            testerx, testery = int(wx*30),int(360-int(wy*30))
            if testerx >= 360:
                testerx = 359
            if testerx < 0:
                testerx = 0
            if testery >= 360:
                testery=359
            if testery < 0:
                testery = 0
                
            g = map[testerx][testery]
            if g>1:
                g=1
            color = (g*256**2+g*256+g)*255
            display.setColor(int(color))
            display.drawPixel(int(wx*30),360-int(wy*30))
            map[testerx][testery]+=.005
            
            # You will eventually REPLACE the following 3 lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            # display.setColor(0xFFFFFF)
            # display.drawPixel(int(wx*30),360-int(wy*30))
            # map[int(wx*30)][360-int(wy*30)]=1 

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    
    #print(pose_x,pose_y,pose_theta)
    display.drawPixel(int(pose_x*30),360-int(pose_y*30))



    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            # for i in range(0, 360)
                # for j in range(0, 360)
                    # if map[i][j] < .5:
                        # map[i][j] = 0
            
            mappingMap = map > .5
            mappingMap = mappingMap * 1
            print(mappingMap)
            
            np.save("map.npy",mappingMap)
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else: # not manual mode
     # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
            # STEP 2.1: Calculate error with respect to current and goal position
        #position error
        PossError = math.sqrt((goal[0] - pose_x )**2 + ( goal[1] - pose_y)**2)
        print(pose_x)
        print(pose_y)
        print(PossError)
        print(goal)
        #bearing error
        bearError = (math.atan2(goal[1]-pose_y,goal[0]-pose_x)) - (pose_theta + math.pi)
        print(bearError)
        # PossError = math.sqrt((goal[0] - pose_x)**2 + (goal[1] - pose_y)**2)
        # #bearing error
        # bearError = math.atan2((goal[1] - pose_y), (goal[0] - pose_x)) - pose_theta
        if bearError < -3.1415: 
            bearError += 6.283 
        if bearError > 6.283:
            bearError -= 6.283
        # Heading error:
        gain =.25
        
        if(abs(PossError)<gain):
            waypointIndex = waypointIndex + 1
            print(waypointIndex)
            if(waypointIndex >= len(waypoints)):
                robot_parts[MOTOR_LEFT].setVelocity(0)
                robot_parts[MOTOR_RIGHT].setVelocity(0)
                exit(0)
            else:
                goal = abs(waypoints[waypointIndex])
            
        
        pass
        
        # ##STEP 2.2: Feedback Controller
    
        dX = PossError
        dTheta = 70 * bearError
      
        pass
        
        
        # ##STEP 1: Inverse Kinematics Equations (vL and vR as a function dX and dTheta)
        # ##Note that vL and vR in code is phi_l and phi_r on the slides/lecture
        vL = (2  * dX - dTheta * AXLE_LENGTH) / 2
        vR = (2 * dX + dTheta * AXLE_LENGTH) / 2 
        
        pass
        
        # ##STEP 2.3: Proportional velocities
        max_val = max(abs(vL), abs(vR), MAX_SPEED)
        vL += max_val
        vR += max_val
        maxVar = max(vL, vR)
        
        vL = (vL / maxVar - 0.5) * 1
        vR = (vR / maxVar - 0.5) * 1
    
    
       
        
        pass
    
        # ##STEP 2.4: Clamp wheel speeds
        if abs(vL) > MAX_SPEED:
            
            vL =  MAX_SPEED
        if abs(vR) > MAX_SPEED:
            
            vR =  MAX_SPEED
            
        pass
    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    #pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    #pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    #pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)