#!/usr/bin/env python3
"""A reader must never observe a document while its replacement is being encoded."""
import argparse,json,threading
from pathlib import Path
from unittest.mock import patch
import storage
from storage import atomic_json,confined
from run_smoke import pending_receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    root=confined(a.output);root.mkdir(parents=True,exist_ok=False)
    original_dump=json.dump;results=[]
    for existing in (False,True):
        path=root/('existing.json' if existing else 'first.json')
        if existing:atomic_json(path,dict(version='old'))
        entered=threading.Event();release=threading.Event();errors=[]
        def held_encoding(value,stream,**kwargs):
            stream.write('{"version":');stream.flush();entered.set()
            if not release.wait(5):raise TimeoutError('Reader probe did not release encoder')
            stream.write('"new"}')
        def write():
            try:atomic_json(path,dict(version='new'))
            except Exception as exc:errors.append(repr(exc))
        with patch.object(storage.json,'dump',held_encoding):
            worker=threading.Thread(target=write);worker.start()
            try:
                assert entered.wait(5)
                visible=[pending_receipt(path) for _ in range(100)]
                expected=dict(version='old') if existing else None
                assert all(value==expected for value in visible),visible
            finally:release.set();worker.join(timeout=5)
        assert not worker.is_alive() and not errors
        assert json.loads(path.read_text())==dict(version='new')
        results.append(dict(existing=existing,reads=100,partial_publications=0,passed=True))
    atomic_json(root/'validation.json',dict(passed=True,cases=results))
    print(json.dumps(dict(passed=True,concurrent_reads=200,partial_publications=0)))


if __name__=='__main__':main()
