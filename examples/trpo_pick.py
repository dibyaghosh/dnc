# Environment Imports
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
import dnc.envs as dnc_envs

# Algo Imports

from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Experiment Imports

from rllab.misc.instrument import stub, run_experiment_lite

def run_task(args,*_):
    
    env = TfEnv(normalize(dnc_envs.create_stochastic('pick'))) # Cannot be solved easily by TRPO

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        min_std=1e-2,
        hidden_sizes=(150, 100, 50),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        force_batch_sampler=True,
        max_path_length=50,
        discount=1,
        step_size=0.02,
    )
    
    algo.train()


run_experiment_lite(
    run_task,
    log_dir='data/trpo/pick',
    n_parallel=12,
    snapshot_mode="last",
    seed=1,
    variant=dict(),
    use_cloudpickle=True
)
