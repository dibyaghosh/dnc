import rllab.envs.mujoco.mujoco_env as mujoco_env
from rllab.core.serializable import Serializable
from sklearn.cluster import KMeans

import numpy as np

class MujocoEnv(mujoco_env.MujocoEnv):
    def __init__(self, frame_skip=1, *args, **kwargs):
        self.bd_index = None
        super().__init__(*args, **kwargs)
        self.frame_skip = frame_skip
        self.geom_names_to_indices = {name:index for index,name in enumerate(self.model.geom_names)}

    
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()

    def get_body_com(self, body_name):
        # Speeds up getting body positions

        if self.bd_index is None:
            self.bd_index = {name:index for index,name in enumerate(self.model.body_names)}
        
        idx = self.bd_index[body_name]
        return self.model.data.com_subtree[idx]

    def touching(self, geom1_name, geom2_name):
        idx1 = self.geom_names_to_indices[geom1_name]
        idx2 = self.geom_names_to_indices[geom2_name]
        for c in self.model.data.contact:
            if (c.geom1 == idx1 and c.geom2 == idx2) or (c.geom1 == idx2 and c.geom2 == idx1):
                return True
        return False

    def touching_group(self, geom1_name, geom2_names):
        idx1 = self.geom_names_to_indices[geom1_name]
        idx2s = set([self.geom_names_to_indices[geom2_name] for geom2_name in geom2_names])

        for c in self.model.data.contact:
            if (c.geom1 == idx1 and c.geom2 in idx2s) or (c.geom1 in idx2s and c.geom2 == idx1):
                return True
        return False
    
    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def get_viewer(self):
        if self.viewer is None:
            viewer = super().get_viewer()
            self.viewer_setup()
            return viewer
        else:
            return self.viewer


class KMeansEnv(MujocoEnv):
    def __init__(self,kmeans_args=None,*args,**kwargs):
        if kmeans_args is None:
            self.kmeans = False
        else:
            self.kmeans = True
            self.kmeans_centers = kmeans_args['centers']
            self.kmeans_index = kmeans_args['index']

        super(KMeansEnv, self).__init__(*args, **kwargs)

    def propose_original(self):
        raise NotImplementedError()
    
    def propose_kmeans(self):
        while True:
            proposal = self.propose_original()
            distances = np.linalg.norm(self.kmeans_centers-proposal,axis=1)
            if np.argmin(distances) == self.kmeans_index:
                return proposal
    
    def propose(self):
        if self.kmeans:
            return self.propose_kmeans()
        else:
            return self.propose_original()
    
    def create_partitions(self,n=10000,k=3):
        X = np.array([self.reset() for i in range(n)])
        kmeans = KMeans(n_clusters=k).fit(X)
        return self.retrieve_centers(kmeans.cluster_centers_)
    
    def retrieve_centers(self,full_states):
        raise NotImplementedError()
    
    def get_param_values(self):
        if self.kmeans:
            return dict(kmeans=True, centers=self.kmeans_centers, index=self.kmeans_index)
        else:
            return dict(kmeans=False)
        
    def set_param_values(self, params):
        self.kmeans = params['kmeans']
        if self.kmeans:
            self.kmeans_centers = params['centers']
            self.kmeans_index = params['index']


def create_env_partitions(env, k=4):
    
    assert isinstance(env, KMeansEnv)
    cluster_centers = env.create_partitions(k=k)

    envs = [env.clone(env) for i in range(k)]
    for i,local_env in enumerate(envs):
        local_env.kmeans = True
        local_env.kmeans_centers = cluster_centers
        local_env.kmeans_index = i
    
    return envs
