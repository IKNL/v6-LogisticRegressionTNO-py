from typing import List
import pandas
import numpy

from tno.mpc.mpyc.secure_learning import (
    PenaltyTypes,
    Logistic,
    SolverTypes,
    ExponentiationTypes
)

from mpyc.runtime import Runtime

from vantage6.common import info, warning
from mpc import setup


runtime = None
secint = Runtime.SecInt()
secnum = Runtime.SecFxp(l=64, f=32)


def RPC_logistic_regression(data: pandas.DataFrame, my_idx: int=0,
                            x: bool=True, columns: List[str]=[]):

    # TODO: retrieve these from the server
    parties = ['localhost:8000', 'localhost:8001', 'localhost:8002']
    my_addr = parties[my_idx]

    info(f"Addresses: {parties}")
    info(f"My address: {my_addr}")

    # create our mpc runtime object
    global runtime
    runtime = setup(parties, my_idx)

    # patch tno mpc module with the mpc object
    monkey_patch()

    # select columns, preprocess and convert to secnum
    sel = data[columns].values.transpose().tolist()
    if x:
        pre = {'x': [[secnum(x, integral=False) for x in row] for row in sel]}
    else:
        enc = [-1 if x==0 else 1 for x in sel[0]]
        pre = {'y': [secnum(y, integral=False) for y in enc]}

    # do the MPC analysis
    info(f'Runtime parties: {runtime.parties}')
    runtime.run(runtime.start())
    res = runtime.run(compute_weigths(**pre))
    runtime.run(runtime.shutdown())
    return res


async def compute_weigths(x=[[]], y=[]):

    assert runtime, "Runtime is missing..."

    # TODO: arent these lengths always the same?
    info('Computing max length')
    n = await compute_max_length(x[0])
    m = await compute_max_length(y)
    info(f'n: {n}, m: {m}')

    info('Prepare input')
    if not x[0]:
        x = [[secnum(None)] * n]
    if not y:
        y = [secnum(None)] * m


    info('Share shares')
    x_mat = numpy.transpose([runtime.input(xi, senders=[1,2]) for xi in x][0])
    y_vec = runtime.input(y, senders=0)

    # initialize logit model
    model = Logistic(solver_type=SolverTypes.GD,
                     exponentiation=ExponentiationTypes.APPROX,
                     penalty=PenaltyTypes.L1,
                     alpha=0.1)

    info('Compute model')
    warning(x_mat)
    warning(y_vec)
    weights = await model.compute_weights_mpc(x_mat, y_vec, tolerance=1e-4)
    info(f'weights: {weights}')


    return weights


async def compute_max_length(x: List) -> int:
    x_sec = runtime.input(secint(len(x)))
    x_max = runtime.max(x_sec)
    return await runtime.output(x_max)

def monkey_patch():

    assert runtime, "Runtime is missing..."

    import mpyc.runtime
    mpyc.runtime.mpc = runtime

    import tno.mpc.mpyc.secure_learning.solvers.solver
    tno.mpc.mpyc.secure_learning.solvers.solver.mpc = runtime

    import tno.mpc.mpyc.secure_learning.solvers.gd_solver
    tno.mpc.mpyc.secure_learning.solvers.gd_solver.mpc = runtime

    import tno.mpc.mpyc.secure_learning.regularizers
    tno.mpc.mpyc.secure_learning.regularizers.mpc = runtime

    import tno.mpc.mpyc.secure_learning.models.secure_model
    tno.mpc.mpyc.secure_learning.models.secure_model.mpc = runtime

    import tno.mpc.mpyc.secure_learning.utils.util_matrix_vec
    tno.mpc.mpyc.secure_learning.utils.util_matrix_vec.mpc = runtime

    import tno.mpc.mpyc.secure_learning.models.common_gradient_forms
    tno.mpc.mpyc.secure_learning.models.common_gradient_forms.mpc = runtime

    import tno.mpc.mpyc.stubs.asyncoro
    tno.mpc.mpyc.stubs.asyncoro.runtime = runtime


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    idx = int(args[0])
    data_file = [
        './local/y.csv',
        './local/x1.csv',
        './local/x2.csv'
    ][idx]
    columns = [['y', 'weight', 'cm'][idx]]
    data = pandas.read_csv(data_file)

    res = RPC_logistic_regression(data, idx, idx!=0, columns)
    print(res)