from asyncio.windows_events import NULL
from logging import raiseExceptions
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
from typing import Any, Dict, Tuple

from vantage6.common import info
from vantage6.client import ContainerClient
from .mpc import setup

from mpyc.runtime import mpc
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as LogisticRegressionSK
import numpy.typing as npt

import tno.mpc.mpyc.secure_learning.test.plaintext_utils.plaintext_objective_functions as plain_obj
from tno.mpc.mpyc.secure_learning import (
    PenaltyTypes,
    # Logistic,
    SolverTypes,
    ExponentiationTypes
)
from tno.mpc.mpyc.secure_learning import Logistic as solverLogistic

from tno.mpc.mpyc.secure_learning.models import Logistic

### stuff I stole from Kaplan Meier

async def new_run_mpyc(self) -> None:
    """
    Runs the Shamir secret sharing part of the protocol using the MPyC
    framework
    """
    async with self.mpc as mpc:
        assert len(mpc.parties) == 3, \
            "Number of parties should be 3"
        await self._start_mpyc()
        await self.obtain_secret_sharings()
        await self.secure_multivariate_log_rank_test()

## If what I did was wrong, then at each site I have to first
solverLogistic.compute_weights_mpc() #2
solverLogistic.gradient_function() #3
solverLogistic.initialize_solver() #1
# solverLogistic.cross_validate()
solverLogistic.score() # runs at the end when everything has been fitted

## Then I would secu

# from tno.mpc.mpyc.secure_learning.solvers import GD
# from tno.mpc.mpyc.secure_learning.solvers.solver import Solver
# from tno.mpc.mpyc.secure_learning.utils import Matrix, MatrixAugmenter, Vector

# from mpyc import sectypes
# from mpyc import asyncoro
# from mpyc.runtime import Runtime, Party
from .mpc import setup

secnum = mpc.SecFxp(l=64, f=32)
## These are fixed for now, but they wcan be overwriten.
random_state = 3
tolerance = 1e-4
INTERNAL_PORT = 8888
WAIT = 4
RETRY = 10
# WORKER_TYPES = {'event', 'groups'}

def main(client: ContainerClient, data, response, covariates, random_state,
         tolerance, fit_intercept=True, penalty=NULL, alpha=NULL, solver=NULL,
         exponentiation=NULL, organization_ids=NULL):
    """
    """
    # create tasks for the organizations at the server

    if not organization_ids:
        organizations = client.get_organizations_in_my_collaboration()
        organization_ids = [organization.get("id") for organization in
                            organizations]

    info("Dispatching get_data")
    task_get_data = client.create_new_task(
        input_={
            'method': 'get_data',
            "kwargs":{
                "response":response,
                "covariates":covariates
            }
        },
        organization_ids=organization_ids
    )

    task_id = task_get_data.get("id")
    task = client.get_task(task_id)
    i = 1

    while not task.get("complete") and i < 180:
        task = client.get_task(task_id)
        info(f"Waiting for results {i}")
        time.sleep(1)
        i += 1

    alpha = get_alpha(alpha)

    info("Obtaining Secure DataFrame")
    df = client.get_results(task_id=task_id)

    info('Running Secure Logistic Regression')

    secure = train_secure_model(df, tolerance, penalty, alpha,
                                    exponentiation)
    info("Completed Secure Logistic Regression Model")

    info('Running Plaintext Logistic Regression')

    plaintext = train_plaintext_model(df, alpha, random_state, fit_intercept,
                                    penalty, solver)

    info("Completed Plaintext Logistic Regression")

    res = {
        "secure":secure,
        "plaintext":plaintext
    }
    return(res)

def RPC_get_data(data: pd.DataFrame, response: str, covariates: list):
    """"Data has to pd.DataFrame, need to build response vector and
        model matrix.

        Response should be single string that exists in dataframe
        Covariates is a list of strings containing
    """

    addresses = _prework()

    if not all(list(map(lambda x: isinstance(x, str), covariates))):
        raise ValueError("Column names should be a str")

    check = covariates.append(response)

    if not all(cols in data.columns for cols in check):
        raise ValueError("Not all columns are present in DataFrame")

    del(check)

    y = np.array(data[response])

    # Need to transform labels from {0,1} -> {-1,+1}
    if all(list(map(lambda x:x in y, (0,1)))):
        y = np.array([-1 if x==0 else 1 for x in y])
    elif not all(list(map(lambda x:x in y, (-1,1)))):
        raise ValueError("All values in the response must be {-1,1}")

    X = np.array(data[covariates]).T

    y_mpc = [secnum(y, integral=False) for y in y.tolist()]
    y_mpc = mpc.input(y_mpc, senders=0)
    X_mpc = [[secnum(x, integral=False) for x in row] for row in X.tolist()]
    X_mpc = [mpc.input(row, senders = 0) for row in X_mpc]
    return([y_mpc,X_mpc])


def _prework():

    client = _temp_fix_client()

    task_id = _find_my_task_id(client)

    organization_id = _find_my_organization_id(client)

    info('Fetch other parties ips and ports')
    results = ContainerClient.get_algorithm_addresses(task_id)
    info(' -> Port numbers available ...')
    results = client.request(f"task/{task_id}/result")
    results = sorted(results, key=lambda d: d['id'])
    assert len(results) == 3, f"There are {len(results)} workers?!"
    other_results = []
    for result in results:
        if result['organization'] != organization_id:
            other_results.append(result)
        else:
            my_result = result
    info(f'my info: {my_result}')
    info(f'Others info: {other_results}')
    info('Extracting ip/ports')

    return [{'ip': r["node"]["ip"], 'port': r['port']} for r in results]

def _temp_fix_client():
    token_file = os.environ["TOKEN_FILE"]
    info(f"Reading token file '{token_file}'")
    with open(token_file) as fp:
        token = fp.read().strip()
    host = os.environ["HOST"]
    port = os.environ["PORT"]
    api_path = os.environ["API_PATH"]
    return ContainerClient(
        token=token,
        port=port,
        host=host,
        path=api_path
    )

def _find_my_task_id(client):
    id_ = jwt.decode(client._access_token, verify=False)['identity']
    return id_.get('task_id')

def _find_my_organization_id(client):
    id_ = jwt.decode(client._access_token, verify=False)['identity']
    return id_.get('organization_id')

def _await_port_numbers(client, task_id):
    result_objects = client.get_other_node_ip_and_port(task_id=task_id)

    c = 0
    while not _are_ports_available(result_objects):
        if c >= RETRY:
            raise Exception('Retried too many times')

        info('Polling results for port numbers...')
        result_objects = client.get_other_node_ip_and_port(task_id=task_id)
        c += 1
        sleep(WAIT)

#     return result_objects


def _are_ports_available(result_objects):
    for r in result_objects:
        _, port = _get_address_from_result(r)
        if not port:
            return False

    return True

def _get_address_from_result(result: Dict[str, Any]) -> Tuple[str, int]:
    address = result['ip']
    port = result['port']

    return address, port


def get_alpha(alpha=NULL):
    """Alpha is a float, ndarray of shape (n_targets,), default=1.0.
       Regularization strength; must be a positive float.
    """

    if isinstance(alpha, NULL):
        info('default alpha will be 1.0')
        alpha = 1.0
    elif (not isinstance(alpha, float) & alpha < 0):
        raise ValueError('alpha should be a positive float')

    return(alpha)


def train_secure_model(df, tolerance, alpha, penalty=NULL,
                exponentiation=NULL):

    if isinstance(exponentiation, NULL):
        info('Computing both Exact and Approximated Logistic regression')
        exponentiation_e = ExponentiationTypes.EXACT
        exponentiation_a = ExponentiationTypes.APPROX
    elif exponentiation == "exact":
        info('computing Exact Logistic regression')
        exponentiation = ExponentiationTypes.EXACT
    elif exponentiation == "approx":
        info('computing Approximated Logistic regression')
        exponentiation = ExponentiationTypes.APPROX
    else:
        raise ValueError(
            f"Recieved unknown exponentiation type: {exponentiation}")

    if isinstance(penalty, NULL):
        penalty = PenaltyTypes.NONE
    elif penalty == "1":
        penalty = PenaltyTypes.L1
    elif penalty == "2":
        penalty = PenaltyTypes.L2
    else:
        raise ValueError(f"Recieved uknown penalty type: {penalty}")

    if df:
        X, y = df[1], df[0]

    if not isinstance(exponentiation, NULL):
        model = Logistic(solver_type=SolverTypes.GD,
                        exponentiation=exponentiation, penalty=penalty,
                        alpha=alpha)

        weights = model.compute_weights_mpc(X=X, y=y,
                                            tolerance=tolerance)

        objective = plain_obj.objective(X=X, y=y, weights=weights,
                                    model="logistic", penalty=penalty,
                                    alpha=alpha)

        result = {
            "Coeffecients": weights,
            "Objective": objective
            }
    else:
        model_approx = Logistic(solver_type=SolverTypes.GD,
                                exponentiation=exponentiation_a,
                                penalty=penalty, alpha=alpha)

        weights_approx = model_approx.compute_weights_mpc(X=X, y=y,
                                                        tolerance=tolerance)

        objective_approx = plain_obj.objective(X=X, y=y,
                                            weights=weights_approx,
                                            model="logistic",
                                            penalty=penalty, alpha=alpha)

        model_exact = Logistic(solver_type=SolverTypes.GD,
                        exponentiation=exponentiation_e,
                        penalty=penalty, alpha=alpha)

        weights_exact = model_exact.compute_weights_mpc(X=X, y=y,
                                                    tolerance=tolerance)

        objective_exact = plain_obj.objective(X=X, y=y,
                                    weights=weights_exact,
                                    model="logistic", penalty=penalty,
                                    alpha=alpha)

        result = {"exact":{}, "approx":{}}

        result["exact"] = {
            "Coefficients": weights_exact,
            "Objective": objective_exact
        }

        result["approx"] = {
            "Coefficients": weights_approx,
            "Objective": objective_approx
        }

    return(result)

def train_plaintext_model(df, alpha, random_state, fit_intercept: bool,
                        penalty=NULL, solver=NULL):

    solvers = ("newton-cg", "lbfgs", "liblinear", "sag", "saga")
    pens = ("l1", "l2", "none")
    pen_solv_pair = {
        solvers[0]:(pens[1], pens[2]),
        solvers[1]:(pens[1], pens[2]),
        solvers[2]:(pens[0], pens[1]),
        solvers[3]:(pens[1], pens[2]),
        solvers[4]:(pens[0], pens[1], pens[2])
    }

    if isinstance(solver, NULL):
        solver = solvers[4]
        info(f"Using default solver, {solver}.")
    elif solver not in solvers:
        raise ValueError(f"Recieved unknown solver type, {solver}")
    else:
        solver = solvers[solvers.index(solver)]
        info(f"Using solver, {solver}")

    if isinstance(penalty, NULL):
        penalty = PenaltyTypes.NONE
        pen = pens[2]
    elif penalty == "1":
        penalty = PenaltyTypes.L1
        pen = pens[0]
    elif penalty == "2":
        penalty = PenaltyTypes.L2
        pen = pens[1]
    else:
        raise ValueError(f"Recieved uknown penalty type: {penalty}")

    if pen not in pen_solv_pair[solver]:
        raise ValueError(f"{solver} requires penalty type in {pen_solv_pair[solver]}")

    if alpha:
        C = 1/(len(df[1]*alpha))

    if df:
        X, y = df[1], df[0]

    model = LogisticRegressionSK(solver=solver, random_state=random_state,
                                fit_intercept=fit_intercept,
                                penalty=pen, C=C)

    model_fit = model.fit(X=X, y=y)

    weights = np.append([model_fit.intercept_], model_fit.coef_).tolist()

    objective = plain_obj.objective(X=X, y=y, weights=weights,
                                    model="logistic", penalty=penalty,
                                    alpha=alpha)

    return(
        {
            "Coeffecients": weights,
            "Objective": objective
        }
    )