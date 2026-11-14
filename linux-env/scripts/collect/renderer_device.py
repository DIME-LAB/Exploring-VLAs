"""Resolve Kit GPU indices by physical UUID + PCI bus, never nvidia-smi index.

NVIDIA Kit Linux troubleshooting Q9 defines the startup GPU table as authority.
A previous table is a launch hint; the CURRENT table must verify active selection.
"""
import hashlib,os,re
from pathlib import Path
from storage import confined


def read_table(path):
    path=confined(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:text=stream.read()
    return parse_table(text),dict(path=str(path),sha256=hashlib.sha256(text.encode()).hexdigest())


def parse_table(text):
    lines=text.splitlines();headers=[i for i,l in enumerate(lines) if '| GPU |' in l and '| Active |' in l]
    if not headers:raise RuntimeError('Unverified renderer: no Kit GPU table; provide a prior owned Kit log before launching')
    rows=[];start=headers[-1]
    for i in range(start+1,len(lines)-2):
        cells=[v.strip() for v in lines[i].split('|')[1:-1]]
        if len(cells)!=7 or not cells[0].isdigit():continue
        details=[v.strip() for v in lines[i+1].split('|')[1:-1]]
        bus=[v.strip() for v in lines[i+2].split('|')[1:-1]]
        if len(details)!=7 or len(bus)!=7 or details[0] or bus[0]:raise RuntimeError('Unverified renderer table layout')
        prefix=details[6].split('..')[0].removeprefix('GPU-').lower()
        if not re.fullmatch('[0-9a-f-]{8,}',prefix) or not re.fullmatch('[0-9a-fA-F]{1,2}',bus[5]):
            raise RuntimeError('Unverified renderer physical identity')
        rows.append(dict(kit_index=int(cells[0]),active=cells[2].startswith('Yes'),uuid_prefix=prefix,pci_bus=int(bus[5],16)))
    if not rows or len({r['kit_index'] for r in rows})!=len(rows):raise RuntimeError('Unverified/ambiguous Kit GPU table')
    return rows


def resolve(rows,cards,assigned_uuid,require_active=False):
    matched=[]
    for row in rows:
        matches=[card for card in cards if card['uuid'].removeprefix('GPU-').lower().startswith(row['uuid_prefix'])
                 and int(card['pci_bus_id'].split(':')[-2],16)==row['pci_bus']]
        if len(matches)!=1:raise RuntimeError('Unverified renderer UUID/PCI correspondence')
        matched.append(dict(row,physical_uuid=matches[0]['uuid'],pci_bus_id=matches[0]['pci_bus_id']))
    selected=[r for r in matched if r['physical_uuid']==assigned_uuid]
    if len(selected)!=1:raise RuntimeError('Assigned physical GPU absent/ambiguous in Kit table')
    if require_active and [r['physical_uuid'] for r in matched if r['active']]!=[assigned_uuid]:
        raise RuntimeError('Renderer is active on another physical GPU')
    return dict(passed=True,assigned_gpu_uuid=assigned_uuid,kit_index=selected[0]['kit_index'],
                selected=selected[0],devices=matched,active_verified=require_active)
