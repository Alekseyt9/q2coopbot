"""Restore exactly verified cold merged caches from retained per-case streams."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    base=Path(__file__).resolve().parents[1]/'workspace/artifacts'
    for entry in plan['entries']:
        target=Path(entry['path']).resolve()
        assert target.is_relative_to(base) and target.parent.name=='merged' and target.name in ('rollout.jsonl','sequence.jsonl')
        if target.exists():
            assert digest(target)==entry['sha256'];continue
        partial=target.with_name(target.name+'.restore-partial');assert not partial.exists()
        accumulator=hashlib.sha256()
        try:
            with partial.open('xb') as output:
                for part in entry['parts']:
                    source=Path(part['path']).resolve();assert source.is_relative_to(base) and source!=target
                    part_hash=hashlib.sha256()
                    with source.open('rb') as stream:
                        while block:=stream.read(1024*1024):output.write(block);accumulator.update(block);part_hash.update(block)
                    assert part_hash.hexdigest()==part['sha256'],str(source)
                output.flush();os.fsync(output.fileno())
            assert accumulator.hexdigest()==entry['sha256'] and partial.stat().st_size==entry['bytes']
            os.replace(partial,target)
        except BaseException:
            # Keep the named partial for inspection; never publish bad bytes.
            raise
        print(target)


if __name__=='__main__':main()
