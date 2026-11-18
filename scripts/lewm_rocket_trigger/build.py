"""Build an isolated Quake II client and export candidate LeWM trigger policies."""
import argparse
import json
from pathlib import Path
import subprocess

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CANDIDATES = {'risk': (.5, 0), 'near': (.5, 72), 'medium': (.5, 128), 'far': (.5, 192)}


def substitute(source, original, replacement):
    if source.count(original) != 1:
        raise ValueError(f'Unexpected H-JEPA hook count for {original!r}')
    return source.replace(original, replacement)


def build(checkpoint, out):
    out.mkdir(parents=True, exist_ok=True)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if cp['version'] != 'quake_lewm_weapon_damage_v2':
        raise ValueError('Expected a LeWM damage v2 checkpoint')
    state = cp['state']
    def layers(prefix):
        return [dict(weight=state[f'{prefix}.{i}.weight'].tolist(),
                     bias=state[f'{prefix}.{i}.bias'].tolist())
                for i in range(0, 8, 2) if f'{prefix}.{i}.weight' in state]
    base = dict(Encoder=layers('world.encode'), Step=layers('world.step'), Head=layers('head'))
    for name, (threshold, distance) in CANDIDATES.items():
        (out / f'{name}.json').write_text(json.dumps(dict(base, Probability=threshold, Clearance=distance)))
    original = REPO / 'internal/policy/hjepa.go'
    source = substitute(original.read_text(), 'type HJEPA struct {', 'type HJEPA struct {\n rocket *rocketGate')
    hook = '''
    if a.Attack && EquippedWeaponName(o.Weapon) == "Rocket Launcher" {
        if path := os.Getenv("Q2_LEWM_ROCKET_POLICY"); path != "" {
            if p.rocket == nil {
                p.rocket, e = loadRocketGate(path)
                if e != nil { return Action{}, e }
            }
            deferShot, risk, err := p.rocket.deferRocket(o, a)
            if err != nil { return Action{}, err }
            if os.Getenv("Q2_LEWM_ROCKET_AUDIT") != "" {
                fmt.Fprintf(os.Stderr,"lewm_gate frame=%d defer=%t risk=%.7f\\n",o.Identity.Frame,deferShot,risk)
            }
            if deferShot { a.Attack = false }
        }
    }
'''
    source = substitute(source, '\t_, e = Command(o, a, [3]int16{})', hook + '\t_, e = Command(o, a, [3]int16{})')
    # Native tests hold weapon requests identical across controllers. The playable
    # client uses ordinary weapon selection unless the explicit fixture flag is set.
    signature = 'func (p *HJEPA) chooseWeapon(o Observation, features []float64) (int, error) {'
    source = substitute(source, signature, signature + '''
    if os.Getenv("Q2_LEWM_FORCE_FIXTURE_ROCKET") == "1" {
        mask, err := WeaponAvailability(o)
        if err != nil { return 0, err }
        if mask[7] { return 7, nil }
        return 0, nil
    }
''')
    generated = out / 'hjepa.go'
    generated.write_text(source + '\n' + (HERE / 'filter.go.inc').read_text(), encoding='utf-8')
    subprocess.run(['gofmt', '-w', str(generated)], check=True)
    overlay = out / 'overlay.json'
    overlay.write_text(json.dumps({'Replace': {str(original): str(generated)}}))
    subprocess.run(['go', 'build', '-overlay', str(overlay), '-o', str(out/'rocket-client.exe'),
                    './cmd/q2coopbot'], cwd=REPO, check=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    build(args.checkpoint, args.output.resolve())
