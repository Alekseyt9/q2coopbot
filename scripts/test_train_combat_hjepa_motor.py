import unittest
import torch
from torch import nn
from train_combat_hjepa_motor import tensors, loss_and_metrics


class MotorLabelsTest(unittest.TestCase):
    def test_soft_teacher_probabilities_and_categorical_loss(self):
        rows=[dict(features=[0]*845,action=[0,0,0,0,.75,.5,.5,0])]
        data=tensors(rows,device='cpu')
        net=nn.Linear(845,8)
        with torch.no_grad():
            net.weight.zero_();net.bias.zero_();net.bias[4]=torch.log(torch.tensor(3.))
        loss,_=loss_and_metrics(net,data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # At the teacher's fire probability, its soft BCE gradient is zero.
        self.assertAlmostEqual(float(net.bias.grad[4]),0.,places=6)

    def test_corrupt_distribution_and_nonfinite_features_rejected(self):
        for action in ([0,0,0,0,1,0,0,0],[0,0,0,0,1.1,1,0,0],[2,0,0,0,1,1,0,0]):
            with self.assertRaises(ValueError):tensors([dict(features=[0]*845,action=action)],device='cpu')
        with self.assertRaises(ValueError):
            tensors([dict(features=[float('nan')]*845,action=[0,0,0,0,1,1,0,0])],device='cpu')


if __name__=='__main__':unittest.main()
