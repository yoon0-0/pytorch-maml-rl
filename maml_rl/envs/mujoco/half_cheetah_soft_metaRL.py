
import math, os, torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET

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

class HalfCheetahSoftMetaRL(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Half cheetah with random box and leg',
                xml_path    = 'mujoco_random_env/xml/half_cheetah_box_leg.xml',
                frame_skip  = 5,
                rand_mass_box = [1, 4],
                rand_mass_leg = [1, 4],
                rand_fric   = [0.1, 1.0],
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200,
                task        = {},
                index       = 0
                ):
        """
            Initialize
        """
        self.VERBOSE    = VERBOSE
        self.name       = name
        self.xml_path   = os.path.abspath(xml_path)
        self.frame_skip = frame_skip
        self.rand_mass_box = rand_mass_box
        self.rand_mass_leg = rand_mass_leg
        self.rand_fric   = rand_fric
        self.render_mode = render_mode
        self.render_w = render_w
        self.render_h = render_h
        self.render_res = render_res
        self.k_p = 0.01
        self.k_i = 0.05
        self.k_d = 0.0001
        self.joint_pos_deg_min = np.array([-20,-20,-20,-20,-20,-20])
        self.joint_pos_deg_max = np.array([20,20,20,20,20,20])
        self._index = index
        self._task = task

        # self.reset_random()
        self.set_box_weight(task.get('box_weight', 0))
        # self.set_leg_weight(task.get('leg_weight', 0))

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:], self.sim.data.qvel.flat])
        self.prev_torque  = a
        
        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]

        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        # self.r    = (x_pos_after - x_pos_before) / self.dt
        self.r = min((x_pos_after-x_pos_before)/self.dt, 1) - 0.1*np.square(a).sum() + 1
        self.info = dict()

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
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.prev_state,
            self.prev_torque,
        ])
        
    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def reset_random(self):
        # self.set_random_box_weight()
        # self.set_random_leg_weight()
        # self.set_random_fric()
        
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

        # Observation and action dimensions 
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("HalfCheetah(2legs) with random box and leg weights")   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}]".format(self.odim, self.adim, self.dt))
            print("box_mass:[{}] leg_mass: [{}] fric:[{}]".format(self.get_box_weight(), self.get_leg_weight(), self.get_fric()))

        # Timing
        self.hz = int(1/self.dt)
        # Viewer setup
        if self.render_mode is not None:
            self.viewer_custom_setup(
                render_mode = self.render_mode,
                render_w    = self.render_w,
                render_h    = self.render_h,
                render_res  = self.render_res
            )

    def set_random_box_weight(self):
        low_bound      = self.rand_mass_box[0]
        high_bound     = self.rand_mass_box[1]
        mass_amplitude = high_bound - low_bound
        box_weight = np.round(np.random.uniform(low_bound, high_bound), 2)
        box_rgb    = np.round(abs((box_weight-low_bound)/mass_amplitude - 1), 3)
        target_xml = open(self.xml_path, 'rt', encoding='UTF8')
        tree = ET.parse(target_xml)
        root = tree.getroot()
        target_tag = root[7][2][8][2][2]
        target_tag.attrib["mass"] = "{}".format(box_weight)
        target_tag.attrib["rgba"] = "{} {} {} 1".format(box_rgb, box_rgb, box_rgb)
        tree.write(self.xml_path)

    def set_box_weight(self, box_weight, TEST=False):
        if not TEST:
            xml_path   = os.path.dirname(os.path.abspath(__file__))+"/xml/halfcheetah/half_cheetah_box_"+str(self._index)+".xml"
        else:
            print('TEST MODE')
            xml_path   = os.path.dirname(os.path.abspath(__file__))+"/xml/halfcheetah/half_cheetah_box_test.xml"
        self.xml_path = xml_path
        low_bound      = self.rand_mass_box[0]
        high_bound     = self.rand_mass_box[1]
        mass_amplitude = high_bound - low_bound
        if mass_amplitude != 0:
            box_rgb    = np.round(abs((box_weight-low_bound)/mass_amplitude - 1), 3)
        else:
            box_rgb    = np.round(abs((box_weight-low_bound)/1 - 1), 3)
        target_xml = open(self.xml_path, 'rt', encoding='UTF8')
        tree = ET.parse(target_xml)
        root = tree.getroot()
        target_tag = root[7][2][8][2][2]
        target_tag.attrib["mass"] = "{}".format(box_weight)
        target_tag.attrib["rgba"] = "{} {} {} 1".format(box_rgb, box_rgb, box_rgb)
        tree.write(self.xml_path)

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

    def set_random_leg_weight(self):
        low_bound      = self.rand_mass_leg[0]/2
        high_bound     = self.rand_mass_leg[1]/2
        mass_amplitude = high_bound - low_bound
        leg_weight = np.round(np.random.uniform(low_bound, high_bound), 2)
        leg_rgb    = np.round(abs((leg_weight-low_bound)/mass_amplitude - 1), 3)
        target_xml = open(self.xml_path, 'rt', encoding='UTF8')
        tree = ET.parse(target_xml)
        root = tree.getroot()
        target_tag_1 = root[7][2][6][2][1]
        target_tag_2 = root[7][2][6][2][2][1]
        target_list  = [target_tag_1, target_tag_2]
        for i in target_list:
            i.attrib["mass"] = "{}".format(leg_weight)
            i.attrib["rgba"] = "{} {} {} 1".format(leg_rgb, leg_rgb, leg_rgb)
        tree.write(self.xml_path)

    def set_leg_weight(self, leg_weight, TEST=False):
        if not TEST:
            xml_path   = os.path.dirname(os.path.abspath(__file__))+"/xml/halfcheetah/half_cheetah_"+str(self._index)+".xml"
        else:
            print('TEST MODE')
            xml_path   = os.path.dirname(os.path.abspath(__file__))+"/xml/halfcheetah/half_cheetah_test.xml"
        self.xml_path = xml_path
        low_bound      = self.rand_mass_leg[0]/2
        high_bound     = self.rand_mass_leg[1]/2
        mass_amplitude = high_bound - low_bound
        if mass_amplitude != 0:
            leg_rgb    = np.round(abs((leg_weight/2-low_bound)/mass_amplitude - 1), 3)
        else:
            leg_rgb    = np.round(abs((leg_weight/2-low_bound)/1 - 1), 3)
        target_xml = open(self.xml_path, 'rt', encoding='UTF8')
        tree = ET.parse(target_xml)
        root = tree.getroot()
        target_tag_1 = root[5][2][6][2][1]
        target_tag_2 = root[5][2][6][2][2][1]
        target_list  = [target_tag_1, target_tag_2]
        for i in target_list:
            i.attrib["mass"] = "{}".format(leg_weight/2)
            i.attrib["rgba"] = "{} {} {} 1".format(leg_rgb, leg_rgb, leg_rgb)
        tree.write(self.xml_path)

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

    def set_random_fric(self):
        low_bound  = self.rand_fric[0]
        high_bound = self.rand_fric[1]
        friction   = np.round(np.random.uniform(low_bound, high_bound), 2)
        target_xml = open(self.xml_path, 'rt', encoding='UTF8')
        tree = ET.parse(target_xml)
        root = tree.getroot()
        target_tag = root[1][1]
        target_tag.attrib["friction"] = "{} 0.1 0.1".format(friction)
        tree.write(self.xml_path)

    def set_fric(self, fric):
        friction = np.round(fric, 2)        
        target_xml = open(self.xml_path, 'rt', encoding='UTF8')
        tree = ET.parse(target_xml)
        root = tree.getroot()
        target_tag = root[1][1]
        target_tag.attrib["friction"] = "{} 0.1 0.1".format(friction)
        tree.write(self.xml_path)

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
        
    def get_box_weight(self):
        """
            Get leg weight in [Kg]
        """
        xml    = open(self.xml_path, 'rt', encoding='UTF8')
        tree   = ET.parse(xml)
        root   = tree.getroot()
        target = root[7][2][8][2][2]
        mass   = np.round(float(target.attrib["mass"]), 2)
        return mass

    def get_leg_weight(self):
        """
            Get leg weight in [Kg]
        """
        xml    = open(self.xml_path, 'rt', encoding='UTF8')
        tree   = ET.parse(xml)
        root   = tree.getroot()
        target = root[7][2][6][2][1]
        mass   = np.round(float(target.attrib["mass"])*2, 2)
        return mass

    def get_fric(self):
        xml  = open(self.xml_path, 'rt', encoding='UTF8')
        tree = ET.parse(xml)
        root = tree.getroot()
        target   = root[1][1]
        friction = target.attrib["friction"]
        return friction

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[3],q[4],q[5],q[6],q[7],q[8]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg
    
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

    # CHANGE LEG / BOX
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

    # CHANGE LEG / BOX
    def reset_task(self, task):
        self._task = task
        self.set_box_weight(task.get('box_weight', 0))
        # self.set_leg_weight(task.get('leg_weight', 0))

if __name__ == "__main__":
    env = HalfCheetahRandomEnvClassMixVersion(render_mode=None)
    for i in range(1000):
        # env.render()
        env.step(np.random.standard_normal(6))
        print(env.get_heading())