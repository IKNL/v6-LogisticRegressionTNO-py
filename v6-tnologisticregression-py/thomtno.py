import argparse
import pandas as pd
import numpy as np

from mpc import setup

from tno.mpc.mpyc.secure_learning import (
    PenaltyTypes,
    Logistic,
    SolverTypes,
    ExponentiationTypes
)
from tno.mpc.mpyc.secure_learning.models.secure_model import Model

from tno.mpc.mpyc.secure_learning.utils import Matrix, NumpyOrMatrix
from tno.mpc.mpyc.secure_learning.utils import util_matrix_vec

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument(
       "-r", "--x-value", type=argparse.FileType('r', encoding='UTF-8'), required=False
   )
   parser.add_argument(
       "-z", "--y-value", type=argparse.FileType('r', encoding='UTF-8'), required=False
   )
   parser.add_argument(
       "-p", "--port", type = int, required=True
   )
   args = parser.parse_args()
   return args

async def transfer(mpc, x=None, y=None):

    secint = mpc.SecInt()
    secnum = mpc.SecFxp(l=64, f=32)

    if x:
        b = pd.read_csv(x, header=None).to_numpy()
        if b.shape[1] ==1:
            b = b.flatten().tolist()
            print("single dim ")
            print(f"This is b, {b}")
            x = [secnum(xi, integral=False) for xi in b]
        else:
            b = b.tolist()
            print("multi dim")
            print(f"This is b: {b}")
            x = [[secnum(x, integral=False) for x in row] for row in b]

        y = []

    if y:
        c = pd.read_csv(y, header=None).to_numpy().flatten().tolist()
        print(f"This is c, {c}")
        print(f"This is c, {type(c)}")
        if all(list(map(lambda x:x in c, (0,1)))):
            c = [-1 if x==0 else 1 for x in c]
        elif not all(list(map(lambda x:x in c, (-1,1)))):
            raise ValueError("All values in the response must be {-1,1}")
        y = [secnum(yi, integral=False) for yi in c]
        x = []


    print(f"this is the number of parties:{mpc.parties}")

    print(11)

    l_init = mpc.input(secint(len(x)))
    l = mpc.max(l_init)
    n = await mpc.output(l)

    print(f"This is n: {n}")

    ly = mpc.input(secint(len(y)))
    ly = mpc.max(ly)
    ny = await mpc.output(ly)

    print(f"this is ny: {ny}")

    print(22)


    if not x:
        print(f"the previous x is: {x}")
        x = [secnum(None)] * n
        print(f"this is the party that sends xnone, {mpc.pid}")


    if not y:
        print(f"the previous x is: {y}")
        y = [secnum(None)] * ny
        print(f"this is the party that sends ynone, {mpc.pid}")

    print(5)

    X = np.transpose(mpc.input(x, senders = [0,2]))
    Y = mpc.input(y, senders = 1)
    print(f"This is X: {X}")
    print(f"This is Y: {Y}")

    model = Logistic(solver_type=SolverTypes.GD,
                exponentiation=ExponentiationTypes.APPROX,
                penalty=PenaltyTypes.L1,
                alpha=0.1)

    print(dir(Logistic))
    # print(f"X:{np.shape(Xt)}")
    # print(f"Y:{np.shape(Y)}")
    weights = await model.compute_weights_mpc(X,Y,tolerance=1e-4)
    # weights = await Model.compute_weights_mpc(model, X, y, tolerance=1e-4)

    return weights

if __name__ == "__main__":
    args = parse_args()

    p1 = 'localhost:8000'
    p2 = 'localhost:8001'
    p3 = 'localhost:8002'
    parties = [p1,p2,p3]
    print(f"this is parties, {parties}")
    print(f"this is the ports, {args.port}")
    mpc = setup(parties, args.port)
    from mpyc.runtime import mpc
    mpc.run(mpc.start())

    if args.x_value:
        print("running x")
        mpc.run(transfer(mpc, x=args.x_value))
    else:
        print("running y    ")
        mpc.run(transfer(mpc, y=args.y_value))
    mpc.run(mpc.shutdown())
