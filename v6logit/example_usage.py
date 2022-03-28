"""
    Example usage for performing secure set intersection
    Run three separate instances e.g.,
    $ python example_usage.py -p Alice -M3 -I0
    $ python example_usage.py -p Bob -M3 - I1
    $ python example_usage.py -p Henri -M3 - I2
"""
import argparse
import asyncio
import math
from typing import List, Union

import numpy as np
import pandas as pd
from mpyc.runtime import mpc
from mpyc.sectypes import SecFxp, SecureFixedPoint

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.utils import FixedPoint

from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper


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

    # Might as well call this cheating, but suffices for now (after all the eventual additive shares will no longer be enourmous after statistical security is merged in Paillier.
    if mpc.pid in (0, 1):
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

    shares_0 = mpc.input(values_to_convert, senders=0)
    shares_1 = mpc.input(values_to_convert, senders=1)

    shares = [val0 + val1 for (val0, val1) in zip(shares_0, shares_1)]

    return shares


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--player",
        help="Name of the sending player",
        type=str.lower,
        required=True,
        choices=["alice", "bob", "henri", "all"],
    )
    args = parser.parse_args()
    return args


async def main(player_instance):
    await player_instance.run_protocol()
    if player_instance.identifier in player_instance.data_parties:
        print("Gathered shares:")
        print(player_instance.feature_names)
        print(player_instance.shares)
    async with mpc:
        shape = await mpc.transfer(
            player_instance.shares.shape if mpc.pid == 0 else None, senders=[0]
        )
        if player_instance.identifier in player_instance.data_parties:
            shares = player_instance.shares
        else:
            shares = np.empty(shape[0])
        X = await convert_additive_to_shamir(shares.flatten().tolist(), precision=8)
        X = np.asarray(X).reshape(shape[0])

        # These lines hould not be there when running the experiment (but helps for debugging
        X_reveal = [await mpc.output(_.tolist()) for _ in X]
        print("Revealed inner join")
        print(X_reveal)


async def generate_instance(player):
    parties = {
        "alice": {"address": "127.0.0.1", "port": 8080},
        "bob": {"address": "127.0.0.1", "port": 8081},
        "henri": {"address": "127.0.0.1", "port": 8082},
    }

    port = parties[player]["port"]
    del parties[player]

    pool = Pool()
    pool.add_http_server(port=port)
    for name, party in parties.items():
        assert "address" in party
        pool.add_http_client(
            name, party["address"], port=party["port"] if "port" in party else 80
        )  # default port=80

    if player == "henri":
        player_instance = Helper(
            identifier=player,
            pool=pool,
        )
    else:
        if player == "alice":
            df = pd.read_csv("data/player_1.csv")
        elif player == "bob":
            df = pd.read_csv("data/player_2.csv")
        player_instance = DatabaseOwner(
            identifier=player,
            identifiers=df[
                [
                    "first_name",
                    "last_name",
                    "date_of_birth",
                    "zip6_code",
                    "gender_at_birth",
                ]
            ].to_numpy(dtype="object"),
            identifiers_phonetic=df[["first_name", "last_name"]].to_numpy(
                dtype="object"
            ),
            identifiers_phonetic_exact=df[["gender_at_birth"]].to_numpy(dtype="object"),
            identifier_date=df[["date_of_birth"]].to_numpy(dtype="object"),
            identifier_zip6=df[["zip6_code"]].to_numpy(dtype="object"),
            data=df.to_numpy(dtype="object")[:, -1, None],
            feature_names=(df.columns[-1],),
            pool=pool,
        )

    await main(player_instance)
    return player_instance


if __name__ == "__main__":
    # Parse arguments and acquire configuration parameters
    args = parse_args()
    player = args.player
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_instance(player))
