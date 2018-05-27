from dnc.envs.base import KMeansEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

import os.path as osp

class PickerEnv(KMeansEnv, Serializable):
    """
    Picking a block, where the block position is randomized over a square region
    
    goal_args is of form ('noisy', center_of_box, half-width of box)
    
    """
    FILE = osp.join(osp.abspath(osp.dirname(__file__)), 'assets/picker.xml')

    def __init__(self, goal_args=('noisy', (.6,.2), .1), frame_skip=5, *args, **kwargs):
        
        self.goal_args = goal_args
        
        super(PickerEnv, self).__init__(frame_skip=frame_skip, *args, **kwargs)
        Serializable.__init__(self, goal_args, frame_skip, *args, **kwargs)

    def get_current_obs(self):
        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.

        return np.concatenate([
            self.model.data.qpos.flat[:],
            self.model.data.qvel.flat[:],
            finger_com,
        ]).reshape(-1)

    def step(self,action):
        self.model.data.ctrl = action
        
        reward = 0
        timesInHand = 0

        for _ in range(self.frame_skip):
            self.model.step()
            step_reward = self.reward()
            timesInHand += step_reward > 0
            reward += step_reward

        done = reward == 0 and self.numClose > 0 # Stop it if the block is flinged

        ob = self.get_current_obs()

        new_com = self.model.data.com_subtree[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

        return Step(ob, float(reward), done,timeInHand=timesInHand)

    def reward(self):
        obj_position = self.get_body_com("object")

        if obj_position[2] < 0.08:
            return 0

        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.
        
        vec_1 = obj_position - finger_com
        dist_1 = np.linalg.norm(vec_1)

        if dist_1 < .1 and obj_position[0] > .2:
            self.numClose += 1
            return obj_position[2]
        else:
            return 0

    def sample_position(self,goal_type,center=(0.6,0.2),noise=0):
        if goal_type == 'fixed':
            return [center[0],center[1],.03]
        elif goal_type == 'noisy':
            x,y = center
            return [x+(np.random.rand()-0.5)*2*noise,y+(np.random.rand()-0.5)*2*noise,.03]
        else:
            raise NotImplementedError()
            
    def retrieve_centers(self,full_states):
        return full_states[:,9:12]

    def propose_original(self):
        return self.sample_position(*self.goal_args)

    @overrides
    def reset(self):
        qpos = self.init_qpos.copy().reshape(-1)
        qvel = self.init_qvel.copy().reshape(-1) + np.random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)

        qpos[1] = -1

        self.position = self.propose() # Proposal
        qpos[9:12] = self.position
        qvel[9:12] = 0

        self.set_state(qpos.reshape(-1), qvel)

        self.numClose = 0

        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = +0.0
        self.viewer.cam.elevation = -40

    @overrides
    def log_diagnostics(self, paths, prefix=''):

        timeOffGround = np.array([
            np.sum(path['env_infos']['timeInHand'])*.01
        for path in paths])

        timeInAir = timeOffGround[timeOffGround.nonzero()]

        if len(timeInAir) == 0:
            timeInAir = [0]

        avgPct = lambda x: round(np.mean(x) * 100, 2)

        logger.record_tabular(prefix+'PctPicked', avgPct(timeOffGround > .3))
        logger.record_tabular(prefix+'PctReceivedReward', avgPct(timeOffGround > 0))
        
        logger.record_tabular(prefix+'AverageTimeInAir',np.mean(timeOffGround))
        logger.record_tabular(prefix+'MaxTimeInAir',np.max(timeOffGround ))