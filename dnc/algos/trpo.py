from dnc.algos.npo import NPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class TRPO(NPO):
    """
    Trust Region Policy Optimization
    
    Please refer to the following classes for parameters:
        - dnc.algos.batch_polopt
        - dnc.algos.npo
        
    """

    def __init__(
            self,
            **kwargs
    ):
        super(TRPO, self).__init__(
            optimizer_class=ConjugateGradientOptimizer,
            optimizer_args=dict(),
            **kwargs
        )
