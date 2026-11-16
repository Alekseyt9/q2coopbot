"""Versioned Jacobi-PCG primitive actor update on native TRAIN rollouts."""
from train_combat_hjepa_actor_natural import ALGORITHM as BASE_ALGORITHM, main
from hjepa_fisher_diagonal import fisher_diagonal, preconditioned_cg

VERSION = 'hjepa_primitive_jacobi_pcg_surrogate_v1'
ALGORITHM = dict(BASE_ALGORITHM,optimizer='damped_fisher_exact_diagonal_jacobi_pcg_no_moments_v1',
                 preconditioner='exact_primitive_fisher_diagonal_plus_damping_v1')


def solve_direction(operator, gradient, model, features, algorithm):
    diagonal = fisher_diagonal(model,features)+algorithm['damping']
    direction,proof = preconditioned_cg(operator,gradient,diagonal,algorithm['cg_iterations'],algorithm['cg_relative_tolerance'])
    proof.update(diagonal_min=float(diagonal.min()),diagonal_max=float(diagonal.max()))
    return direction,proof


if __name__ == '__main__':
    main(version=VERSION,algorithm=ALGORITHM,direction_solver=solve_direction,
         scope='Jacobi-PCG clipped-surrogate pilot, exact local primitive Fisher diagonal. All encoder0/actor/log_std train, weapon/world heads frozen; finite nonlinear surrogate and behavior-mean-KL acceptance unchanged. CG cap may prevent convergence. No combat quality or promotion claim.')
