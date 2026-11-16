import unittest
import torch
from train_combat_hjepa_weapons import tensors, loss_metrics


def row():
    x = [0.] * 845
    x[833] = x[834] = 1.
    return dict(features=x, teacher_weapon_probabilities=[.2, .8] + [0.] * 10)


class WeaponDataTests(unittest.TestCase):
    def test_unavailable_label_rejected(self):
        r = row()
        r['teacher_weapon_probabilities'] = [0., 0., 1.] + [0.] * 9
        with self.assertRaises(ValueError):
            tensors([r], 'cpu')

    def test_unknown_inventory_only_keep(self):
        r = row()
        r['features'][834] = 0
        r['teacher_weapon_probabilities'] = [1.] + [0.] * 11
        data = tensors([r], 'cpu')
        net = torch.nn.Linear(845, 12)
        with torch.no_grad():
            net.bias[9] = 10000
        loss, m = loss_metrics(net, data)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(m['switch_fraction'], 0)

    def test_nonfinite_or_unnormalized_labels_rejected(self):
        for value in [float('nan'), -.1, .5]:
            r = row()
            r['teacher_weapon_probabilities'][0] = value
            with self.assertRaises(ValueError):
                tensors([r], 'cpu')

    def test_unavailable_logit_has_no_gradient(self):
        data = tensors([row()], 'cpu')
        net = torch.nn.Linear(845, 12)
        loss, _ = loss_metrics(net, data)
        loss.backward()
        self.assertEqual(float(net.bias.grad[9]), 0)
        self.assertGreater(float(net.bias.grad[:2].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
