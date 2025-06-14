import pybullet as p
import pybullet_data
import time

import numpy as np
import matplotlib.pyplot as plt

## TODO:
# Collision avoidance
# Performance meassure

# Object
class Object():
    def __init__(self, x, y):
        # Define cube dimensions (half extents)
        cube_half_extents = [0.1, 0.1, 0.1]  # Cube of size 2*[0]m x 2*[1]m x 2*[2]m
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_half_extents)
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=cube_half_extents, rgbaColor=[0, 1, 0, 1]) 

        cube_start_pos = [x, y, cube_half_extents[2]]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.createMultiBody(baseMass=10,
                                    baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id,
                                    basePosition=cube_start_pos,
                                    baseOrientation=cube_start_orientation)
        p.changeDynamics(
            bodyUniqueId=cube_id,
            linkIndex=-1,
            lateralFriction=5.0
        )
        self.id = cube_id
        self.state = 'free'

# Agent
class Agent():
    def __init__(self, x, y):
        # Define cube dimensions (half extents)
        cube_half_extents = [0.2, 0.2, 0.2]  # Cube of size 2*[0]m x 2*[1]m x 2*[2]m
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_half_extents)
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=cube_half_extents, rgbaColor=[0, 0, 1, 1]) 

        cube_start_pos = [x, y, cube_half_extents[2]]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.createMultiBody(baseMass=1,
                                    baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id,
                                    basePosition=cube_start_pos,
                                    baseOrientation=cube_start_orientation
        )
        self.id = cube_id
        self.v = np.array([0,0,0])
        self.agent_pos = np.array(p.getBasePositionAndOrientation(self.id)[0])
        self.max_speed = 10
        self.state = 'searching'
        self.search_radius = 5
        self.target = None
        self.grab_constrain = None

    def agentMotion(self, v=[0,0,0], w=0):
        p.resetBasePositionAndOrientation(
            self.id,
            [self.agent_pos[0], self.agent_pos[1], 0.2],
            [0, 0, 0, 1]
        )
        p.resetBaseVelocity(
            self.id,
            linearVelocity=v,     
            angularVelocity=[0, 0, w]
        )

    def agentBehaviour(self, agents, objects):
        self.agent_pos = np.array(p.getBasePositionAndOrientation(self.id)[0])

        if self.state == 'searching':
            k = self.search_radius
            for i in range(len(objects)):
                obj_pos = np.array(p.getBasePositionAndOrientation(objects[i].id)[0])
                if np.linalg.norm(obj_pos-self.agent_pos) <= k:
                    self.target = objects[i]
                    k = np.linalg.norm(obj_pos-self.agent_pos)
            if k != self.search_radius:
                self.state = 'grabbing'
            
            # Biased random walk
            bias = self.agent_pos/(np.linalg.norm(self.agent_pos)+0.1)
            if np.linalg.norm(self.agent_pos) >= home_base_radius*2:
                bias *= -0.1

            if np.random.rand(1) > 0.99:
                x = np.random.rand()*np.random.choice([-self.max_speed, self.max_speed])
                y = np.random.rand()*np.random.choice([-self.max_speed, self.max_speed])
                self.v = np.array([x,y,0])
                self.v += bias*3
                self.v *= self.max_speed/(np.linalg.norm(self.v)+0.01)
                self.agentMotion(self.v)
            else:
                self.agentMotion(self.v+bias*10)


        elif self.state == 'grabbing':
            if self.target.state == 'free':
                target_pos = np.array(p.getBasePositionAndOrientation(self.target.id)[0])
                dist = (target_pos - self.agent_pos)
                self.v = dist/np.linalg.norm(dist)*self.max_speed
                if np.linalg.norm(dist) <= 0.4:
                    self.target.state = 'grabbed'
                    self.agentMotion()

                    p.resetBasePositionAndOrientation(
                        self.target.id,
                        [self.agent_pos[0], self.agent_pos[1], 0.5],
                        [0, 0, 0, 1]  # identity quaternion = no rotation
                    )
                    # Grab the object
                    parent_pos, parent_orn = p.getBasePositionAndOrientation(self.id)
                    child_pos, child_orn   = p.getBasePositionAndOrientation(self.target.id)

                    # Transform from parent world frame to parent local frame
                    parent_inv_pos, parent_inv_orn = p.invertTransform(parent_pos, parent_orn)

                    # Compute child pose relative to parent
                    rel_pos, rel_orn = p.multiplyTransforms(
                        parent_inv_pos, parent_inv_orn,
                        child_pos, child_orn
                    )
                    self.grab_constrain = p.createConstraint(
                        parentBodyUniqueId=self.id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=self.target.id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=rel_pos,
                        childFramePosition=[0, 0, 0],  # attach at child center
                        parentFrameOrientation=rel_orn,
                        childFrameOrientation=[0, 0, 0, 1]
                    )

                    self.state = 'homing'
                    
                else:
                    self.agentMotion(self.v)
            else:
                self.state = 'searching'

        elif self.state == 'homing':
            target_pos = np.array(p.getBasePositionAndOrientation(self.target.id)[0])
            dist = (np.array([0,0,0]) - target_pos)
            self.v = dist/np.linalg.norm(dist)*self.max_speed

            # Check if on home base
            if np.linalg.norm(dist) <= home_base_radius*0.75:
                self.agentMotion()
                p.removeConstraint(self.grab_constrain)
                self.target.state = 'collected'
                self.state = 'searching'
            else:
                self.agentMotion(self.v)
        
setups = [1,3,5,10]
results = []
for n in range(len(setups)):
    agents = []
    objects = []
    collected = []

    num_agents = setups[n]
    num_objects = 100
    home_base_radius = 5
    world_size = 15

    total_time = 20
    time_step = 0.005

    GUI = True

    # Setup world
    if GUI:
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(
            cameraDistance=20.0,
            cameraYaw=0,                
            cameraPitch=-89.99,
            cameraTargetPosition=[0, 0, 0] 
        )
    else:
        physicsClient = p.connect(p.DIRECT)

    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")

    ## Spawn agents
    for i in range(num_agents):
        # Create the agents at random points inside the home base
        r = (np.random.rand()*home_base_radius) 
        angle = np.random.rand()*2*np.pi
        x = r*np.cos(angle)
        y = r*np.sin(angle)

        agents.append(Agent(x,y))

    ## Spawn objects
    for i in range(num_objects):
        # Create the cubes at random points around the home base
        r = (np.random.rand()*world_size + home_base_radius) 
        angle = np.random.rand()*2*np.pi
        x = r*np.cos(angle)
        y = r*np.sin(angle)
        
        objects.append(Object(x,y))

    steps = int(total_time/time_step)
    for i in range(steps):
        for agent in agents:
            agent.agentBehaviour(agents, objects)

        for c in collected:
            if np.linalg.norm(p.getBasePositionAndOrientation(c.id)[0]) > home_base_radius:
                c.state = 'free'
                objects.append(collected.pop(collected.index(c)))
        for o in objects:
            if np.linalg.norm(p.getBasePositionAndOrientation(o.id)[0]) <= home_base_radius:
                o.state = 'collected'
                collected.append(objects.pop(objects.index(o)))

        p.stepSimulation()
        time.sleep(time_step)

    percent_collected = len(collected)/num_objects*100
    np.round(percent_collected,2)
    print('World size:', str(world_size)+'m x '+str(world_size)+'m', '| Home base radius:', str(home_base_radius)+'m')
    print('Angents:', num_agents, '| Objects:', num_objects)
    print('Time: ', str(total_time) + 's')
    print('Objects collected:', str(percent_collected)+'%')
    results.append(percent_collected)
    p.disconnect()

plt.plot(results)
plt.show()