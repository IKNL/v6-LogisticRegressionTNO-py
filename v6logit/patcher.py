import warnings
from mpyc.sectypes import SecureFixedPoint
from tno.mpc.mpyc.stubs.asyncoro import mpc_coro_ignore, returnType
import tno.mpc.mpyc.secure_learning.utils.util_matrix_vec as mpc_utils
from tno.mpc.mpyc.secure_learning.utils import (
    Matrix,
    MatrixAugmenter,
    SecureDataPermutator,
    SeqMatrix,
    Vector,
    seq_to_list,
)

# import tno.mpc.mpyc.secure_learning.solvers.solver
from tno.mpc.mpyc.secure_learning.solvers.solver import Solver
import tno.mpc.mpyc.secure_learning.solvers.solver

def monkey_patch():

    print("patch")

    class NewSolver(tno.mpc.mpyc.secure_learning.solvers.solver.Solver):

        def __init__(self) -> None:
            super().__init__()
            print("NEW SOLVER!!!")

        mpc = None

        @mpc_coro_ignore
        async def _get_coefficients(
            self,
            X: Matrix[SecureFixedPoint],
            y: Vector[SecureFixedPoint],
            n_maxiter: int,
            print_progress: bool,
            secure_permutations: bool,
        ) -> Vector[SecureFixedPoint]:
            """
            Compute the model weights, or coefficients.
            Only solver-independent calculations are explicitely defined (and called "outer loop").

            :param X: Training data
            :param y: Target vector
            :param n_maxiter: Maximum number of iterations before method stops and result is returned
            :param print_progress: Print progress (epoch number) to standard output
            :param secure_permutations: Perform matrix permutation securely
            :return: vector with (secret-shared) weights computed by the solver
            """
            print('*'*80)
            print(self.mpc.parties)
            assert n_maxiter > 0 and self.weights_init is not None
            stype = type(self.weights_init[0])
            await returnType(stype, len(self.weights_init))

            X = X.copy()
            y = y.copy()

            ###
            # Solver-specific pre-processing
            ###
            X, y = self.preprocessing(X, y)

            # Initialize data permutator
            self.data_permutator = SecureDataPermutator(
                secure_permutations=secure_permutations
            )

            self.permutable_matrix.augment("X", X)
            self.permutable_matrix.augment(
                "y", mpc_utils.vector_to_matrix(y, transpose=True)
            )

            # Initialize weights
            weights_old = self.weights_init.copy()
            weights_new = weights_old.copy()
            weights_oldest = weights_new.copy()

            ###
            # Gradient descent outer loop (solver-independent)
            ###
            for epoch in range(1, n_maxiter + 1):
                if print_progress and epoch % 10 == 1:
                    print(f"Epoch {epoch}...")

                # Permutation
                if epoch == 0:
                    await self.data_permutator.refresh_seed()

                self.permutable_matrix.update(
                    self.iterative_data_permutation(self.permutable_matrix)
                )
                X = self.permutable_matrix.retrieve("X")
                y = mpc_utils.mat_to_vec(
                    self.permutable_matrix.retrieve("y"), transpose=True
                )

                ###
                # Gradient descent inner loop (solver-dependent)
                ###
                weights_new = self.inner_loop_calculation(X, y, weights_old, epoch=epoch)
                weights_old = weights_new.copy()

                # Check for convergence
                # Note that (update_diff <= self.tolerance) is True <=>
                # mpc.is_zero_public(update_diff >= self.tolerance) is True
                update_diff = self.mpc.in_prod(
                    self.mpc.vector_sub(weights_new, weights_oldest),
                    self.mpc.vector_sub(weights_new, weights_oldest),
                )
                prev_norm = self.mpc.in_prod(weights_oldest, weights_oldest)
                has_converged = self.mpc.is_zero_public(
                    update_diff >= self.tolerance * prev_norm
                )
                if await has_converged:
                    break
                weights_oldest = weights_new.copy()

            ###
            # Solver-specific post-processing
            ###
            weights_predict = self.postprocessing(weights_new)

            # Metadata
            self.secret_shared_weights = weights_predict
            self.nr_epochs = epoch
            plain_update_diff = float(await self.mpc.output(update_diff))
            plain_prev_norm = float(await self.mpc.output(prev_norm))
            try:
                self.rel_update_diff = plain_update_diff / plain_prev_norm
            except ZeroDivisionError:
                warnings.warn(
                    "Update difference division by zero, indicating that the \
                        weights vector is very small."
                )
                self.rel_update_diff = 0
            return weights_predict

    tno.mpc.mpyc.secure_learning.solvers.solver.Solver = NewSolver
