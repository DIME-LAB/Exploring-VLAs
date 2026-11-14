"""Explicit host profiles; an unknown machine has no writable collection root."""
import json
from pathlib import Path
import socket


def profile(hostname=None):
    hostname = socket.gethostname() if hostname is None else hostname
    matches = []
    for name in ('dual-a4500', 'rtx6000'):
        path = Path(__file__).with_name('hosts') / (name + '.json')
        config = json.loads(path.read_text())
        if hostname in config['hostnames']:
            matches.append(dict(config, name=name))
    if len(matches) != 1:
        raise RuntimeError('Collection requires one qualified host profile: ' + repr(hostname))
    return matches[0]


if __name__ == '__main__':
    print(profile()['root'])
