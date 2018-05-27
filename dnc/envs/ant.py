import numpy as np

from dnc.envs.base import KMeansEnv
from rllab.envs.base import Step

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger

import os.path as osp


class AntEnv(KMeansEnv, Serializable):

    FILE = osp.join(osp.abspath(osp.dirname(__file__)), 'assets/ant.xml')

    def __init__(self, angle_range=(0,2*np.pi), frame_skip=2, *args, **kwargs):
        self.goal = np.array([0,0])
        self.angle_range = angle_range
  
        super(AntEnv, self).__init__(frame_skip=frame_skip, *args, **kwargs)
        Serializable.__init__(self, angle_range, frame_skip, *args, **kwargs)
        
    def get_current_obs(self):
        current_position = self.get_body_com("torso")
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.goal,
           current_position,
           current_position[:2]-self.goal
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        
        # Base Reward
        compos = self.get_body_com("torso")
        dist = np.linalg.norm((compos[:2] - self.goal)) / 5
        forward_reward = 1 - dist
        
        # Control and Contact Costs
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))
        )
        
        reward = forward_reward - ctrl_cost - contact_cost
        
        state = self._state
        notdone = all([
            np.isfinite(state).all(),
            not self.touching('torso_geom','floor'),
            state[2] >= 0.2,
            state[2] <= 1.0,
        ])

        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done, distance=dist, task=self.goal)

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        
        qpos = self.init_qpos.copy().reshape(-1)
        qvel = self.init_qvel.copy().reshape(-1) + np.random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)

        qvel[9:12] = 0

        self.goal = self.propose()
        qpos[-7:-5] = self.goal
        
        self.set_state(qpos.reshape(-1), qvel)
        
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 20.0
        self.viewer.cam.azimuth = +90.0
        self.viewer.cam.elevation = -20

    def retrieve_centers(self,full_states):
        return full_states[:,15:17]

    def propose_original(self):
        angle = self.angle_range[0] + (np.random.rand()*(self.angle_range[1]-self.angle_range[0]))
        magnitude = 5
        
        return np.array([
            magnitude * np.cos(angle),
            magnitude * np.sin(angle)
        ])

    @overrides
    def log_diagnostics(self, paths,prefix=''):
        min_distances = np.array([
            np.min(path["env_infos"]['distance'])
            for path in paths
        ])

        final_distances = np.array([
            path["env_infos"]['distance'][-1]
            for path in paths
        ])
        avgPct = lambda x: round(np.mean(x)*100,2)

        logger.record_tabular(prefix+'AverageMinDistanceToGoal', np.mean(min_distances))
        logger.record_tabular(prefix+'MinMinDistanceToGoal', np.min(min_distances))

        logger.record_tabular(prefix+'AverageFinalDistanceToGoal', np.mean(final_distances))
        logger.record_tabular(prefix+'MinFinalDistanceToGoal', np.min(final_distances))

        logger.record_tabular(prefix+'PctInGoal', avgPct(progsFinal < .2))
