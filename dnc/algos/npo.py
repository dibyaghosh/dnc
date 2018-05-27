# Base imports
import numpy as np
import tensorflow as tf
from dnc.algos.batch_polopt import BatchPolopt

# Optimizers
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

# Utilities
from rllab.misc.ext import sliced_fun
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import ext

# Logging
import rllab.misc.logger as logger


# Convenience Function
def default(variable, defaultValue):
    return variable if variable is not None else defaultValue


class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer_class=None,
            optimizer_args=None,
            step_size=0.01,
            penalty=0.0,
            **kwargs
    ):

        self.optimizer_class = default(optimizer_class, PenaltyLbfgsOptimizer)
        self.optimizer_args = default(optimizer_args, dict())

        self.penalty = penalty
        self.constrain_together = penalty > 0

        self.step_size = step_size

        self.metrics = []
        super(NPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):

        ###############################
        #
        # Variable Definitions
        #
        ###############################

        all_task_dist_info_vars = []
        all_obs_vars = []

        for i, policy in enumerate(self.local_policies):

            task_obs_var = self.env_partitions[i].observation_space.new_tensor_variable('obs%d' % i, extra_dims=1)
            task_dist_info_vars = []

            for j, other_policy in enumerate(self.local_policies):

                state_info_vars = dict()  # Not handling recurrent policies
                dist_info_vars = other_policy.dist_info_sym(task_obs_var, state_info_vars)
                task_dist_info_vars.append(dist_info_vars)

            all_obs_vars.append(task_obs_var)
            all_task_dist_info_vars.append(task_dist_info_vars)

        obs_var = self.env.observation_space.new_tensor_variable('obs', extra_dims=1)
        action_var = self.env.action_space.new_tensor_variable('action', extra_dims=1)
        advantage_var = tensor_utils.new_tensor('advantage', ndim=1, dtype=tf.float32)

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % k)
            for k, shape in self.policy.distribution.dist_info_specs
        }

        old_dist_info_vars_list = [old_dist_info_vars[k] for k in self.policy.distribution.dist_info_keys]

        input_list = [obs_var, action_var, advantage_var] + old_dist_info_vars_list + all_obs_vars

        ###############################
        #
        # Local Policy Optimization
        #
        ###############################

        self.optimizers = []
        self.metrics = []

        for n, policy in enumerate(self.local_policies):

            state_info_vars = dict()
            dist_info_vars = policy.dist_info_sym(obs_var, state_info_vars)
            dist = policy.distribution

            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
            lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
            surr_loss = - tf.reduce_mean(lr * advantage_var)

            if self.constrain_together:
                additional_loss = Metrics.kl_on_others(n, dist, all_task_dist_info_vars)
            else:
                additional_loss = tf.constant(0.0)

            local_loss = surr_loss + self.penalty * additional_loss

            kl_metric = tensor_utils.compile_function(inputs=input_list, outputs=additional_loss, log_name="KLPenalty%d" % n)
            self.metrics.append(kl_metric)

            mean_kl_constraint = tf.reduce_mean(kl)

            optimizer = self.optimizer_class(**self.optimizer_args)
            optimizer.update_opt(
                loss=local_loss,
                target=policy,
                leq_constraint=(mean_kl_constraint, self.step_size),
                inputs=input_list,
                constraint_name="mean_kl_%d" % n,
            )
            self.optimizers.append(optimizer)

        ###############################
        #
        # Global Policy Optimization
        #
        ###############################

        # Behaviour Cloning Loss

        state_info_vars = dict()
        center_dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        behaviour_cloning_loss = tf.losses.mean_squared_error(action_var, center_dist_info_vars['mean'])
        self.center_optimizer = FirstOrderOptimizer(max_epochs=1, verbose=True, batch_size=1000)
        self.center_optimizer.update_opt(behaviour_cloning_loss, self.policy, [obs_var, action_var])

        # TRPO Loss

        kl = dist.kl_sym(old_dist_info_vars, center_dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, center_dist_info_vars)
        center_trpo_loss = - tf.reduce_mean(lr * advantage_var)
        mean_kl_constraint = tf.reduce_mean(kl)

        optimizer = self.optimizer_class(**self.optimizer_args)
        optimizer.update_opt(
            loss=center_trpo_loss,
            target=self.policy,
            leq_constraint=(mean_kl_constraint, self.step_size),
            inputs=[obs_var, action_var, advantage_var] + old_dist_info_vars_list,
            constraint_name="mean_kl_center",
        )

        self.center_trpo_optimizer = optimizer

        # Reset Local Policies to Global Policy

        assignment_operations = []

        for policy in self.local_policies:
            for param_local, param_center in zip(policy.get_params_internal(), self.policy.get_params_internal()):
                if 'std' not in param_local.name:
                    assignment_operations.append(tf.assign(param_local, param_center))

        self.reset_to_center = tf.group(*assignment_operations)

        return dict()

    def optimize_local_policies(self, itr, all_samples_data):

        dist_info_keys = self.policy.distribution.dist_info_keys
        for n, optimizer in enumerate(self.optimizers):

            obs_act_adv_values = tuple(ext.extract(all_samples_data[n], "observations", "actions", "advantages"))
            dist_info_list = tuple([all_samples_data[n]["agent_infos"][k] for k in dist_info_keys])
            all_task_obs_values = tuple([samples_data["observations"] for samples_data in all_samples_data])

            all_input_values = obs_act_adv_values + dist_info_list + all_task_obs_values
            optimizer.optimize(all_input_values)

            kl_penalty = sliced_fun(self.metrics[n], 1)(all_input_values)
            logger.record_tabular('KLPenalty%d' % n, kl_penalty)

    def optimize_global_policy(self, itr, all_samples_data):

        all_observations = np.concatenate([samples_data['observations'] for samples_data in all_samples_data])
        all_actions = np.concatenate([samples_data['agent_infos']['mean'] for samples_data in all_samples_data])

        num_itrs = 1 if itr % self.distillation_period != 0 else 30

        for _ in range(num_itrs):
            self.center_optimizer.optimize([all_observations, all_actions])

        paths = self.global_sampler.obtain_samples(itr)
        samples_data = self.global_sampler.process_samples(itr, paths)

        obs_values = tuple(ext.extract(samples_data, "observations", "actions", "advantages"))
        dist_info_list = [samples_data["agent_infos"][k] for k in self.policy.distribution.dist_info_keys]

        all_input_values = obs_values + tuple(dist_info_list)

        self.center_trpo_optimizer.optimize(all_input_values)
        self.env.log_diagnostics(paths)

    @overrides
    def optimize_policy(self, itr, all_samples_data):

        self.optimize_local_policies(itr, all_samples_data)
        self.optimize_global_policy(itr, all_samples_data)

        if itr % self.distillation_period == 0:
            sess = tf.get_default_session()
            sess.run(self.reset_to_center)
            logger.log('Reset Local Policies to Global Policies')

        return dict()

############################
#
#  KL Divergence
#
############################


class Metrics:
    @staticmethod
    def symmetric_kl(dist, info_vars_1, info_vars_2):
        side1 = tf.reduce_mean(dist.kl_sym(info_vars_2, info_vars_1))
        side2 = tf.reduce_mean(dist.kl_sym(info_vars_1, info_vars_2))
        return (side1 + side2) / 2

    @staticmethod
    def kl_on_others(n, dist, dist_info_vars):
        # \sum_{j=1} E_{\sim S_j}[D_{kl}(\pi_j || \pi_i)]
        if len(dist_info_vars) < 2:
            return 0

        kl_with_others = 0
        for i in range(len(dist_info_vars)):
            if i != n:
                kl_with_others += Metrics.symmetric_kl(dist, dist_info_vars[i][i], dist_info_vars[i][n])

        return kl_with_others / (len(dist_info_vars) - 1)
