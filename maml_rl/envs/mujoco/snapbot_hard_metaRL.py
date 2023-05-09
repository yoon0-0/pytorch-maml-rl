import math,os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET
import random

# Convert quaternion to Euler angle 
def quaternion_to_euler_angle(w, x, y, z):
    """
        w, x, y, z -> R, P, Y
    """
    y_sqr = y*y

    t_0 = +2.0 * (w*x + y*z)
    t_1 = +1.0 - 2.0 * (x*x + y_sqr)
    X = math.degrees(math.atan2(t_0, t_1))
	
    t_2 = +2.0 * (w*y - z*x)
    t_2 = +1.0 if t_2 > +1.0 else t_2
    t_2 = -1.0 if t_2 < -1.0 else t_2
    Y = math.degrees(math.asin(t_2))
	
    t_3 = +2.0 * (w*z + x*y)
    t_4 = +1.0 - 2.0 * (y_sqr + z*z)
    Z = math.degrees(math.atan2(t_3, t_4))
	
    return X, Y, Z

# Snapbot (6 legs) Environment
class Snapbot6EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 6 legs',
                leg_idx     = 123456,
                # xml_path    = 'mujoco_random_env/xml/snapbot_6/robot_6_',
                rand_mass_box = [1, 4],
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200,
                task        = {},
                test        = False,
                n_malfun_leg  = 1,
                index       = 0
                ):
        """
            Initialize
        """
        self.VERBOSE    = VERBOSE
        self.name       = name
        self.leg_idx    = str(leg_idx)
        # self.xml_path   = os.path.abspath(xml_path+'{}.xml'.format(self.leg_idx))
        self.frame_skip = frame_skip
        self.condition  = condition
        self.ctrl_coef  = ctrl_coef
        self.head_coef  = head_coef
        self.rand_mass  = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40,43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40,43,40,43,40,43,40])
        self.rand_mass_box = rand_mass_box
        self._task = task
        self._n_malfun_leg = n_malfun_leg
        self._index = index
        self.test = test

        xml_path   = os.path.dirname(os.path.abspath(__file__))+"/xml/snapbot_6/robot_6_123456.xml"
        self.xml_path = xml_path
        self.set_malfun_leg()
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(6legs) Environment")   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] head_coef:[{}]".format(self.ctrl_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:],self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[6:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        heading_diff  = heading_after - heading_before
        x_diff        = x_pos_after - x_pos_before
        y_diff        = y_pos_after - y_pos_before
        # Set reward
        reward_forward = (x_pos_after-x_pos_before) / self.dt
        reward_survive = 1
        cost_ctrl      = self.ctrl_coef * np.square(a).sum()
        cost_heading   = self.head_coef * (heading_diff**2+y_diff**2)
            
        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = min(reward_forward, 1) + reward_survive - cost_ctrl - cost_heading
        self.info = dict({'reward_forward': reward_forward, 'cost_ctrl': cost_ctrl})
        
        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info

    def _get_obs(self):
        self.index = np.array(self.get_max_leg()).reshape(1)
        # self.sim.data.qvel.flat = 6(x,y,z,r,p,y) + 2(camera) + 4(다리 당 joint)*6(다리 개수) = 32
        # self.sim.data.qpos.flat = 7(x,y,z,a,b,c,d) + 2(camera) + 4(다리 당 joint)*6(다리 개수) = 33
        new_qpos = self.sim.data.qpos
        new_qvel = self.sim.data.qvel
        # zero masking TODO
        # if self.malfun_leg_idxs:
        #     for idx in self.malfun_leg_idxs:
        #         new_qpos[9:][4*idx:4*idx+4] = 0
        #         new_qvel[8:][4*idx:4*idx+4] = 0
        #         self.prev_state[9:][4*idx:4*idx+4] = 0
        #         self.prev_state[31+8:][4*idx:4*idx+4] = 0
        #         self.prev_torque[2*idx:2*idx+2] = 0

        # print("new_qpos", new_qpos)
        # print("new_qvel", new_qvel)
        # print("self.prev_state", self.prev_state)
        # print("self.prev_torque", self.prev_torque)
        return np.concatenate([
            new_qpos.flat[2:], # 33-2 = 31
            new_qvel.flat, # 32
            self.prev_state, # 31+32 = 63
            self.prev_torque, # 6*2 = 12
            self.index
        ])

    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22],q[25],q[26],q[29],q[30]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg

    def get_seonsor_data(self):
        l1 = self.sim.data.get_sensor('touchsensor_1')
        l2 = self.sim.data.get_sensor('touchsensor_2')
        l3 = self.sim.data.get_sensor('touchsensor_3')
        l4 = self.sim.data.get_sensor('touchsensor_4')
        l5 = self.sim.data.get_sensor('touchsensor_5')
        l6 = self.sim.data.get_sensor('touchsensor_6')
        ls = [l1,l2,l3,l4,l5,l6]
        return ls

    def get_max_leg(self):  
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 8.0 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3): # follow the robot torso
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx]
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame
    
    def sample_tasks(self, num_tasks):
        # return self.sample_tasks_leg_weight(num_tasks)
        return self.sample_tasks_box_weight(num_tasks)

    def sample_tasks_box_weight(self, num_tasks):
        low_bound      = self.rand_mass_box[0]
        high_bound     = self.rand_mass_box[1]
        box_weights = self.np_random.uniform(low_bound, high_bound, size=(num_tasks,))
        tasks = [{'box_weight': box_weight} for box_weight in box_weights]
        return tasks

    def sample_tasks_leg_weight(self, num_tasks):
        low_bound      = self.rand_mass_leg[0]
        high_bound     = self.rand_mass_leg[1]
        leg_weights = self.np_random.uniform(low_bound, high_bound, size=(num_tasks,))
        tasks = [{'leg_weight': leg_weight} for leg_weight in leg_weights]
        return tasks
    
    def set_malfun_leg(self):
        # self.ctrl_leg = random.sample(range(6), 6-self._n_malfun_leg)
        self.malfun_leg_idxs = random.sample(range(6), self._n_malfun_leg)

    def reset_task(self, task):
        self._task = task
        if self.malfun_leg_idxs:
            self.set_malfun_leg()
        # self.set_box_weight(task.get('box_weight', 0))
        # self.set_leg_weight(task.get('leg_weight', 0))

# Snapbot (5 legs) Environment
class Snapbot5EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 5legs',
                leg_idx     = 12345,
                # xml_path    = 'mujoco_random_env/xml/snapbot_5/robot_5_',
                rand_mass_box = [1, 4],
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200,
                n_malfun_leg= 1,
                test        = True,
                task        = {},
                index       = 0
                ):
        """
            Initialize
        """
        self.VERBOSE     = VERBOSE
        self.name        = name
        self.leg_idx     = str(leg_idx)
        self.xml_path    = os.path.dirname(os.path.abspath(__file__))+"/xml/snapbot_5/robot_5_12345.xml"
        self.frame_skip  = frame_skip
        self.condition   = condition
        self.ctrl_coef   = ctrl_coef
        self.head_coef   = head_coef
        self.rand_mass   = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40,43,40,43,40])
        self._task = task
        self._n_malfun_leg = n_malfun_leg
        self._index = index
        self.test = test
        
        self.set_malfun_leg()
        # Open xml
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(5legs: {}) Environment".format(self.leg_idx))   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] head_coef:[{}]".format(self.ctrl_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:],self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[5:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        x_diff        = x_pos_after - x_pos_before
        y_diff        = y_pos_after - y_pos_before
        heading_diff  = heading_after - heading_before
        # Set reward
        reward_forward = (x_pos_after-x_pos_before) / self.dt
        reward_survive = 1
        cost_ctrl      = self.ctrl_coef * np.square(a).sum()
        cost_heading   = self.head_coef * (heading_diff**2+y_diff**2)
            
        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = min(reward_forward, 1) + reward_survive - cost_ctrl - cost_heading
        self.info = dict({'reward_forward': reward_forward, 'cost_ctrl': cost_ctrl})
        
        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info

    def _get_obs(self):
        # TODO: 계산 다시 하기
        self.index = np.array(self.get_max_leg()).reshape(1)
        # self.sim.data.qvel.flat = 6(x,y,z,r,p,y) + 2(camera) + 4(다리 당 joint)*5(다리 개수) = 28
        # self.sim.data.qpos.flat = 7(x,y,z,a,b,c,d) + 2(camera) + 4(다리 당 joint)*5(다리 개수) = 29
        new_qpos = self.sim.data.qpos
        new_qvel = self.sim.data.qvel

        # print("new_qpos", new_qpos)
        # print("new_qvel", new_qvel)
        # print("self.prev_state", self.prev_state)
        # print("self.prev_torque", self.prev_torque)

        return np.concatenate([
            new_qpos.flat[2:], # 29-2 = 27
            new_qvel.flat, # 28
            self.prev_state, # 31+32 = 55
            self.prev_torque, # 5*2 = 10
            self.index
        ])

    def set_malfun_leg(self):
        # self.ctrl_leg = random.sample(range(6), 6-self._n_malfun_leg)
        if not self.test:
            self.malfun_leg_idxs = random.sample(range(5), self._n_malfun_leg)
        else:
            self.malfun_leg_idxs = [5]

    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22],q[25],q[26]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg

    def get_seonsor_data(self):
        l1 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[0]))
        l2 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[1]))
        l3 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[2]))
        l4 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[3]))
        l5 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[4]))
        ls = [l1,l2,l3,l4,l5]
        return ls

    def get_max_leg(self):  
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 8.0 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3): # follow the robot torso
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx]
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame

    def sample_tasks(self, num_tasks):
        # return self.sample_tasks_leg_weight(num_tasks)
        # return self.sample_tasks_box_weight(num_tasks)
        return
    
    def reset_task(self, task):
        self._task = task
        self.set_malfun_leg()

# Snapbot (4 legs) Environment
class Snapbot4EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 4 legs',
                leg_idx     = 1245,
                # xml_path    = '/snapbot_4/robot_4_',
                rand_mass_box = [1, 4],
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                task        = {},
                index       = 0,
                render_res  = 200,
                n_malfun_leg= 2,
                test        = True
                ):
        """
            Initialize
        """
        self.VERBOSE    = VERBOSE
        self.name       = name
        self.leg_idx    = str(leg_idx)
        # self.xml_path   = os.path.abspath(xml_path+'{}.xml'.format(self.leg_idx))
        self.rand_mass_box = rand_mass_box
        self.frame_skip = frame_skip
        self.condition  = condition
        self.ctrl_coef  = ctrl_coef
        self.head_coef  = head_coef
        self.rand_mass  = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40,43,40])
        self._task = task
        self._index = index
        self.test = test
        self._n_malfun_leg = n_malfun_leg
        self.set_malfun_leg()
        xml_path   = os.path.dirname(os.path.abspath(__file__))+"/xml/snapbot_4/robot_4_1245.xml"
        self.xml_path = xml_path

        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(4legs) Environment")   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] head_coef:[{}]".format(self.ctrl_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:], self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[4:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        x_diff        = x_pos_after - x_pos_before
        y_diff        = y_pos_after - y_pos_before
        heading_diff  = heading_after - heading_before
        # Set reward
        reward_forward = (x_pos_after-x_pos_before) / self.dt
        reward_survive = 1
        cost_ctrl      = self.ctrl_coef * np.square(a).sum()
        cost_heading   = self.head_coef * (heading_diff**2+y_diff**2)
            
        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = min(reward_forward, 1) + reward_survive - cost_ctrl - cost_heading
        self.info = dict({'reward_forward': reward_forward, 'cost_ctrl': cost_ctrl})
        
        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info

    
    def _get_obs(self):
        """
            Get observation
        """
        self.index = np.array(self.get_max_leg()).reshape(1)
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.prev_state,
            self.prev_torque,
            self.index
        ])
    
    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg
    
    def get_seonsor_data(self):
        """
            Get sensor data from touchsensors
        """
        l1 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[0]))
        l2 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[1]))
        l3 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[2]))
        l4 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[3]))
        ls = [l1, l2, l3, l4]
        return ls

    def get_max_leg(self):
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 8.0 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3): # follow the robot torso
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx]
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame
        
    def sample_tasks(self, num_tasks):
        # return self.sample_tasks_leg_weight(num_tasks)
        return self.sample_tasks_box_weight(num_tasks)

    def sample_tasks_box_weight(self, num_tasks):
        low_bound      = self.rand_mass_box[0]
        high_bound     = self.rand_mass_box[1]
        box_weights = self.np_random.uniform(low_bound, high_bound, size=(num_tasks,))
        tasks = [{'box_weight': box_weight} for box_weight in box_weights]
        return tasks

    def sample_tasks_leg_weight(self, num_tasks):
        low_bound      = self.rand_mass_leg[0]
        high_bound     = self.rand_mass_leg[1]
        leg_weights = self.np_random.uniform(low_bound, high_bound, size=(num_tasks,))
        tasks = [{'leg_weight': leg_weight} for leg_weight in leg_weights]
        return tasks
    
    def set_malfun_leg(self):
        # self.ctrl_leg = random.sample(range(6), 6-self._n_malfun_leg)
        if not self.test:
            self.malfun_leg_idxs = random.sample(range(4), self._n_malfun_leg)
        else:
            self.malfun_leg_idxs = [2, 5] # 3, 6th

    def reset_task(self, task):
        self._task = task
        self.set_malfun_leg()

# SnapbotEnvClass
class Snapbot3EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 3 legs',
                leg_idx     = 245,
                xml_path    = 'mujoco_random_env/xml/snapbot_3/robot_3_',
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200,
                test        = True
                ):
        """
            Initialize
        """
        self.VERBOSE    = VERBOSE
        self.name       = name
        self.leg_idx    = str(leg_idx)
        self.xml_path   = os.path.abspath(xml_path+'{}.xml'.format(self.leg_idx))
        self.frame_skip = frame_skip
        self.condition  = condition
        self.ctrl_coef  = ctrl_coef
        self.head_coef  = head_coef
        self.rand_mass  = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40])
        self.set_malfun_leg()
        self.test       = test
        # Open xml
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(3legs) Environment")   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] head_coef:[{}]".format(self.ctrl_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:], self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[3:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        x_diff        = x_pos_after - x_pos_before
        y_diff        = y_pos_after - y_pos_before
        heading_diff  = heading_after - heading_before
        # Set reward
        reward_forward = (x_pos_after-x_pos_before) / self.dt
        reward_survive = 1
        cost_ctrl      = self.ctrl_coef * np.square(a).sum()
        cost_heading   = self.head_coef * (heading_diff**2+y_diff**2)
            
        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = min(reward_forward, 1) + reward_survive - cost_ctrl - cost_heading
        self.info = dict({'reward_forward': reward_forward, 'cost_ctrl': cost_ctrl})

        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info
    
    def _get_obs(self):
        """
            Get observation
        """
        self.index = np.array(self.get_max_leg()).reshape(1)
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.prev_state,
            self.prev_torque,
            self.index
        ])

    def sample_tasks(self, num_tasks):
        return

    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg
    
    def get_seonsor_data(self):
        """
            Get sensor data from touchsensors
        """
        l1 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[0]))
        l2 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[1]))
        l3 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[2]))
        ls = [l1, l2, l3]
        return ls

    def get_max_leg(self):
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 3.3 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3):
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx] # follow the robot torso
            # self.viewer.cam.lookat[d_idx] = 0 # fix at zero
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame

    def set_malfun_leg(self):
        self.malfun_leg_idxs = [1, 3, 6]

    def reset_task(self, task):
        self._task = task
        # self.set_box_weight(task.get('box_weight', 0))
        # self.set_leg_weight(task.get('leg_weight', 0))

if __name__ == "__main__":
    env = Snapbot3EnvClass(leg_idx=245, render_mode=None)
    # env = Snapbot3EnvClass(render_mode=None)
    for i in range(1000):
        env.render()
        print(env.get_joint_pos_deg())
        env.step(np.random.standard_normal(6)*1)

