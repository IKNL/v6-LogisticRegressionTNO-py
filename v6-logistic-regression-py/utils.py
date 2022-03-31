import os
import jwt

from time import sleep
from typing import Dict, Any, Tuple

from vantage6.common import info
from vantage6.client import ContainerClient

RETRY = 10
WAIT = 10

def obtain_algorithm_adresses():

    # obtain container client from token
    info('obtaining client')
    client = get_client()

    # task and organization id where this tasks runs
    info('obtaining my task id')
    task_id = get_task_id(client)

    # poll if the ports are available
    info('polling for ports')
    i = 0
    n = 0
    while n!=3 and i < RETRY:
        n = 0
        i = i + 1
        results = client.request(f'task/{task_id}/result')
        for result in results:
            if result["ports"]:
                n = n + 1

        info(f'--> Found {n} parties(s)')
        if n!=3:
            sleep(WAIT)

    info('sort ports by result id')
    results = sorted(results, key=lambda d: d['id'])

    assert len(results) == 3, f"There are {len(results)} workers?!"

    return [{'ip': r["node"]["ip"], 'port': r['ports'][0]['port']} for r in results]

def get_client():
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

def get_task_id(client):
    id_ = jwt.decode(client._access_token, verify=False)['identity']
    return id_.get('task_id')

def get_my_organization_id(client):
    id_ = jwt.decode(client._access_token, verify=False)['identity']
    return id_.get('organization_id')

def _find_my_ip_and_port(client):
    own_id =  get_task_id(client)
    tasks: list = client.request(f'task/{own_id}/result')
    assert len(tasks) == 1, "Multiple master tasks?"
    result = tasks.pop()
    return (result['node']['ip'], result['port'])

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

    return result_objects


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