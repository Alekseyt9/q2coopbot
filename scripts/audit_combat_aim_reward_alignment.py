"""Describe selected-target/nearest-target alignment in sealed native datasets.

No policy inference, training, target selection or command generation.
"""
import argparse
import json
import pathlib

from process_combat_architecture_pool import read, save, sha


def known_bbox(enemy):
    solid = enemy.get('observed_solid')
    return (enemy.get('clear_shot') is True and solid is not None and
            solid not in (0,31) and solid & 31 != 0 and
            ((solid >> 10) & 63)*8-32 > -((solid >> 5) & 31)*8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    proof = read(root/'recovery/verified-members.json')
    protocol = read(root/'protocol.json')
    groups = {}
    for entry in protocol['evaluations']:
        counts = dict(available_provider_steps=0, selected_known_steps=0,
                      multiple_known_steps=0, selected_non_nearest_steps=0)
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            assert sha(path) == proof['members'][str(path.parent)]['report_sha256']
            member = pathlib.Path(read(path)['results'][0]['root'])
            with (member/'dataset/rewards.jsonl').open(encoding='utf-8-sig') as stream:
                eligible = {r['step'] for line in stream if (r := json.loads(line))['available']}
            with (member/'dataset/steps.jsonl').open(encoding='utf-8-sig') as stream:
                for line in stream:
                    step = json.loads(line)
                    if step['index'] not in eligible or step['owner'] != 'provider': continue
                    counts['available_provider_steps'] += 1
                    enemies = [e for e in step['observation']['enemies'] if known_bbox(e)]
                    action = step['action']
                    selected = next((e for e in enemies if e['id'] == action.get('target_entity')
                                     and e.get('observed_track') == action.get('target_track')), None)
                    if selected is None: continue
                    counts['selected_known_steps'] += 1
                    if len(enemies) < 2: continue
                    counts['multiple_known_steps'] += 1
                    nearest = min(enemies, key=lambda e:e['distance'])
                    counts['selected_non_nearest_steps'] += selected['id'] != nearest['id']
        denominator = counts['multiple_known_steps']
        groups[entry['model']+'-'+entry['label']] = dict(counts=counts,
            selected_non_nearest_fraction_when_multiple=counts['selected_non_nearest_steps']/denominator if denominator else None)
    save(root/'aim-reward-alignment.json', dict(state='complete', groups=groups,
         member_proof_sha256=sha(root/'recovery/verified-members.json'),
         scope='Observed clear valid bboxes, available provider rewards only. Descriptive selected target versus nearest target used by current aim shaping; not proof that either target was optimal or that shaping caused regression. No NN replay.'))
    print(json.dumps(groups))


if __name__ == '__main__': main()
