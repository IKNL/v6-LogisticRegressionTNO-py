from audioop import add
import pandas
import numpy

from typing import List, Union

from mpyc.runtime import Runtime
from mpyc.sectypes import SecFxp, SecureFixedPoint
from tno.mpc.mpyc.secure_learning import (
    PenaltyTypes,
    Logistic,
    SolverTypes,
    ExponentiationTypes
)
from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper
from vantage6.common import info

from .mpc import setup
from .utils import obtain_algorithm_adresses

runtime = None
secint = Runtime.SecInt()
secnum = Runtime.SecFxp(l=64, f=32)

# Assumptions
# party 0: alice
# party 1: bob
# party 2: henri

# parties and ports will be overwritten by the RPC method. For testing
# these values are used
ports = [8000, 8001, 8002]
ips = ['localhost', 'localhost', 'localhost']
parties = [f'localhost:{p}' for p in ports]
players = ['alice', 'bob', 'henri']

def RPC_vertical_logistic_regression(data: pandas.DataFrame, metadata: dict,
                                     idx: int):

    addresses = obtain_algorithm_adresses()
    info(f'Adresses ready: {addresses}')

    # the order in which the input from the tasks is posted is essential
    # in otder to make this work. Thus alice needs to be first, then bob
    # an finaly the helper
    global parties
    parties = [f'{a["ip"]}:{a["port"]}' for a in addresses]
    parties[idx] = 'localhost:8888'
    global ports
    ports = [_['port'] for _ in addresses]
    global ips
    ips = [_['ip'] for _ in addresses]
    info(f'mpc parties {parties}')
    # create runtime
    create_runtime(parties, idx)
    info(runtime.parties)
    info(f'my id:{idx}')
    # exit(1)

    # the global parties is passed arround
    return run(data, idx, metadata)


def run(data: pandas.DataFrame, idx: int, metadata: dict):

    print('=> start runtime')
    runtime.run(runtime.start())
    print('=> Run Matching')
    res = runtime.run(generate_instance(data, idx, metadata))

    info("=> Run model fitting")
    res = numpy.transpose(res)
    y = res[-1]
    x_mat = numpy.transpose(res[0:-2])
    res = runtime.run(compute_weigths(x_mat=x_mat, y=y))

    print('=> shutdown')
    runtime.run(runtime.shutdown())

    return res

async def compute_weigths(x_mat=[[]], y=[]):

    assert runtime, "Runtime is missing..."

    info(f'==> parties: {runtime.parties}')

    # initialize logit model
    model = Logistic(solver_type=SolverTypes.GD,
                     exponentiation=ExponentiationTypes.APPROX,
                     penalty=PenaltyTypes.L1,
                     alpha=0.1)

    info('==> Compute weigths')
    weights = await model.compute_weights_mpc(x_mat, y, tolerance=1e-4)

    info(f'==> Weights: {weights}')
    return weights


async def compute_max_length(x: List) -> int:
    x_sec = runtime.input(secint(len(x)))
    x_max = runtime.max(x_sec)
    return await runtime.output(x_max)


async def convert_additive_to_shamir(
    shares_to_convert: Union[FixedPoint, List[FixedPoint]],
    precision: int = None,
    secfxp: SecureFixedPoint = SecFxp(),
) -> Union[SecureFixedPoint, List[SecureFixedPoint]]:
    is_list = isinstance(shares_to_convert, list)
    if is_list:
        to_convert_list = shares_to_convert[:]
    else:
        to_convert_list = [shares_to_convert]

    # Might as well call this cheating,
    # but suffices for now (after all the eventual additive shares will
    # no longer be enourmous after statistical security is merged in Paillier.
    if runtime.pid in (0, 1):
        values_to_convert = [
            secfxp(
                float(str(element)[-16:])
                if element.value > 0
                else float("-" + str(element)[-16:])
            )
            for element in to_convert_list
        ]
    else:
        values_to_convert = [secfxp(None)] * len(to_convert_list)

    shares_0 = runtime.input(values_to_convert, senders=0)
    shares_1 = runtime.input(values_to_convert, senders=1)

    shares = [val0 + val1 for (val0, val1) in zip(shares_0, shares_1)]

    return shares


async def secure_join(player_instance):
    info('==> secure join')
    await player_instance.run_protocol()
    if player_instance.identifier in player_instance.data_parties:
        print("Gathered shares:")
        print(player_instance.feature_names)
        print(player_instance.shares)

    # async with runtime:
    shape = await runtime.transfer(
        player_instance.shares.shape if runtime.pid == 0 else None, senders=[0]
    )
    if player_instance.identifier in player_instance.data_parties:
        shares = player_instance.shares
    else:
        shares = numpy.empty(shape[0])
    X = await convert_additive_to_shamir(shares.flatten().tolist(), precision=8)
    X = numpy.asarray(X).reshape(shape[0]) # = X_runtime

    # These lines hould not be there when running the experiment (but helps for debugging
    X_reveal = [await runtime.output(_.tolist()) for _ in X]
    print("Revealed inner join")
    print(X_reveal)
    print(numpy.transpose(X_reveal))
    print('y:', numpy.transpose(X_reveal)[-1])

    # contains all features + y
    return X


async def generate_instance(df, idx, metadata):

    port = ports[idx]
    print("port: ", port, "idx: ", idx)
    player = players[idx]

    pool = Pool()
    pool.add_http_server(port=8888)
    pool.http_server.external_port = ports[idx]

    for i, p in enumerate(ports):
        if i == idx:
            continue
        print(f'adding: {players[i]}')

        pool

        pool.add_http_client(players[i], ips[i], port=p)

    # for name, party in parties.items():
    #     assert "address" in party
    #     pool.add_http_client(
    #         name, party["address"], port=party["port"] if "port" in party else 80
    #     )  # default port=80

    if idx == 2:
        player_instance = Helper(
            identifier=player,
            pool=pool,
        )
    else:
        columns = metadata[idx]["x"]
        if metadata[idx]["y"]:
            columns.append(metadata[idx]["y"])
        columns = tuple(columns)
        print(columns)
        # if idx == 0:
        #     # df = pandas.read_csv("data/dataset_A_500_5_.csv")
        #     columns = ('mediastinoscopie', 'expl_chir')
        # elif idx == 1:
        #     # df = pandas.read_csv("data/dataset_B_500_5_.csv")
        #     columns = ('age', 'gender', 'os_36')
        matching = metadata["matching"]
        player_instance = DatabaseOwner(
            identifier=player,
            identifiers=df[matching["identifiers"]].to_numpy(dtype="object"),
            identifiers_phonetic=df[[*matching["identifiers_phonetic"]]].to_numpy(
                dtype="object"
            ),
            identifiers_phonetic_exact=df[[*matching["identifiers_phonetic_exact"]]].to_numpy(dtype="object"),
            identifier_date=df[[*matching["identifier_date"]]].to_numpy(dtype="object"),
            identifier_zip6=df[[*matching["identifier_zip6"]]].to_numpy(dtype="object"),
            # data=df.to_numpy(dtype="object")[:, -1, None],
            # feature_names=(df.columns[-1],),
            data=df[[*columns]].to_numpy(dtype="object"), # only ints or floats
            feature_names=columns,
            pool=pool,
        )

    res = await secure_join(player_instance)
    return res


def create_runtime(parties, idx):
    # create our mpc runtime object

    global runtime
    runtime = setup(parties, idx)

    # patch tno mpc modules with the mpc object
    monkey_patch()


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
    create_runtime(parties, idx)
    # data_file = [
    #     './local/y.csv',
    #     './local/x1.csv',
    #     './local/x2.csv'
    # ][idx]
    data_file = [
        './local/dataset_A_10_4_.csv',
        './local/dataset_B_10_4_.csv',
        './local/x2.csv'
    ][idx]

    metadata = {
        0: {
            "x": ["mediastinoscopie", "expl_chir"],
            "y": None
        },
        1: {
            "x": ["age", "gender"],
            "y": "os_36"
        },
        "matching": {
            "identifiers": [
                "Sex",
                "Date-of-birth",
                "Postcode"
            ],
            "identifiers_phonetic": ["Name"],
            "identifiers_phonetic_exact": ["Sex"],
            "identifier_date": ["Date-of-birth"],
            "identifier_zip6": ["Postcode"]
        }
    }

    columns = [['y', 'weight', 'cm'][idx]]
    data = pandas.read_csv(data_file)

    # transform y value
    if idx == 1:
        data['os_36'] = [-1 if x==0 else 1 for x in data['os_36']]

    res = run(data, idx, metadata)

    print(res)