"""CUDA checks for causal successor lookup and true zero-bootstrap ends."""
import ast
import copy
from pathlib import Path
import torch
from combat_critic_credit import reestimate


def main():
    assert torch.cuda.is_available()
    torch.set_default_device('cuda')
    tree = ast.parse(Path(__file__).with_name('ppo_combat.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'advantages')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<advantages>', 'exec'), namespace)
    advantages = namespace['advantages']

    class Value(torch.nn.Module):
        def forward(self, x):
            return x[..., :1] * 2, None

    context = [dict(seed=7, frame=10+i, features=[float(i+1)], memory=dict(reset=i==0)) for i in range(4)]
    rows = [dict(seed=7, index=i+1, frame=10+i, next_frame=11+i, features=[float(i+1)],
                 next_features=[float(i+2)], next_reset=False, sample=dict(value=0.), reward=1.,
                 terminal=False, truncated=i==2, bootstrap_zero=i==2, next_value=0.) for i in range(3)]
    original = copy.deepcopy(rows)
    prepared = (torch.tensor([[[1.], [2.], [3.], [4.]]]),
                torch.tensor([[0, 0], [0, 1], [0, 2]], dtype=torch.long),
                torch.arange(3), [context])
    adv, current, successors = reestimate(rows, prepared, Value(), .99, .95, advantages, torch)
    assert current == [2., 4., 6.] and successors == [4., 6., 0.]
    assert abs(adv[2] - (-5.)) < 1e-12
    assert abs(adv[1] - (1 + .99*6 - 4 + .99*.95*(-5))) < 1e-12
    assert rows == original
    rows[-1].update(terminal=True, truncated=False, bootstrap_zero=False)
    assert reestimate(rows, prepared, Value(), .99, .95, advantages, torch)[2][-1] == 0.
    rows[0]['next_features'] = [999.]
    try:
        reestimate(rows, prepared, Value(), .99, .95, advantages, torch)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched canonical successor accepted')
    print('CUDA critic-credit causal, boundary and sample-purity checks passed')


if __name__ == '__main__':
    main()
