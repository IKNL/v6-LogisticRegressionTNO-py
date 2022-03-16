import argparse
import fileinput
from tabnanny import filename_only
import mpyc
from mpyc.runtime import mpc
from asyncio.windows_events import NULL
from ctypes import Union
from logging import debug, raiseExceptions
from sre_compile import isstring
from turtle import Pen
from xmlrpc.client import Boolean
from bidict import ValueDuplicationError
import pandas as pd
import numpy as np
import jwt
import os
import time
import subprocess
import asyncio
from time import sleep
from typing import Any, Dict, List, Tuple, Optional
import os.path
import logging

from mpyc.runtime import mpc
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as LogisticRegressionSK
import numpy.typing as npt
from sympy import MatrixExpr

import tno.mpc.mpyc.secure_learning.test.plaintext_utils.plaintext_objective_functions as plain_obj
from tno.mpc.mpyc.secure_learning import (
    PenaltyTypes,
    Logistic,
    SolverTypes,
    ExponentiationTypes
)
from tno.mpc.mpyc.secure_learning.models.secure_model import Model
from tno.mpc.mpyc.secure_learning.exceptions import SecureLearnTypeError
from mpyc import mpctools
from mpyc.seclists import seclist
from tno.mpc.mpyc.secure_learning.utils import Matrix, MatrixAugmenter, Vector
from mpyc.sectypes import SecureFixedPoint, SecureNumber
# from .mpc import setup
from logging import debug
secnum = mpc.SecFxp(l=64, f=32)
## These are fixed for now, but they wcan be overwriten.
random_state = 3
tolerance = 1e-4
INTERNAL_PORT = 8888
WAIT = 4
RETRY = 10


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--r', '--response',
                    type=argparse.FileType('r', encoding='UTF-8'),
                    required=False)

    parser.add_argument('--z', '--covariate',
                    type=argparse.FileType('r', encoding='UTF-8'),
                    required=False)

    args = parser.parse_args()
    print(1)
    # print(f"This is args: {args}")
    if args.r is not None:
        print(2)
        data = pd.read_csv(args.r.name)
        name = "response"


    elif args.z is not None:
        print(3)
        data = pd.read_csv(args.z.name)
        name = "covariate"

    print(99)
    ans = {"data":data, "name":name}
    return ans
    # print(ans["data"])


def get_data(data: pd.DataFrame, col: str):

    print(4)
    # secnum = mpc.SecFxp(l=64, f=32)
    # if not isinstance(data, None):
    if col=="response":
        print(5)
        # name = data.columns[0]
        # y = np.array(data[name].head())
        y = data.head().to_numpy().flatten()
          # Need to transform labels from {0,1} -> {-1,+1}
        if all(list(map(lambda x:x in y, (0,1)))):
            y = np.array([-1 if x==0 else 1 for x in y])
        # elif not all(list(map(lambda x:x in y, (-1,1)))):
        #     raise ValueError("All values in the response must be {-1,1}")
        y_mpc = [secnum(x, integral = False) for x in y.tolist()]
        print(88)
        print(f"this is y: {len(y)}")
        return y_mpc

    if col == "covariate":
        print(6)
        # assume for now that each site has 1 coln
        if data.shape[1] == 1:
            data = data.head().to_numpy().flatten()
            X = np.array(data, ndmin=1)
            X_mpc = [secnum(x, integral = False) for x in X.tolist()]
        else:
            X = data.to_numpy().T
            X_mpc = [[secnum(x, integral=False) for x in row] for row in X.tolist()]
        print(f"This is X: {len(X_mpc)}")
        return X_mpc


# solver = Model.solver

async def transfer_y(y=None, X=None):

    print(7)
    # with mpc.run:
    y = await mpc.transfer(y)
    X = await mpc.transfer(X)

    y = [x for x in y if x]
    X = [x for x in X if x]

    # X = X[0]
    y = y[0]

    # print(888)

    print(f"This is y: {y}")
    print(f"This is X: {X}")
    print("!!!")
    print(f"this is x00: {X[0][0]}")
    print(f"this is y00: {y[0]}")

    model = Logistic(solver_type=SolverTypes.GD,
                exponentiation=ExponentiationTypes.EXACT,
                penalty=PenaltyTypes.NONE,
                alpha=1.0)

    weights = await model.compute_weights_mpc(X=X,y=y ,tolerance=tolerance)
    print(222)
    # objective = plain_obj.objective(X=X, y=y, weights=weights,
    #                 model="logistic", penalty=PenaltyTypes.NONE,
    #                 alpha=1.0)
    # weights = [float(_) for _ in mpc.convert(weights, int)]

    print(weights)

    # return {
    #     "weights":weights,
    #     "objective":objective
    # }


if __name__ == "__main__":

    args = parse_args()

    mpc.run(mpc.start())

    data = get_data(data = args["data"], col = args["name"])
    print("Did the data")
    # print(data)

    if args['name'] == 'covariate':
        X = mpc.run(transfer_y(X=data))
    else:
        y = mpc.run(transfer_y(y=data))

    print("Transferring")

    # mpc.run(transfer_y(data))

    mpc.run(mpc.shutdown())
