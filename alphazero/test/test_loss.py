import torch
import torch.nn.functional as F

from alphazero.loss import (
    masked_cross_entropy,
    masked_kl_div)

torch.set_printoptions(sci_mode=False)

def test_masked_cross_entropy():
    inputs = torch.tensor([
        [ 0.,  1.,   1.,  1.],
        [ 0.,  0.,   0.,  0.],
        [ 0.,  0.,   0., -999.],
        [ 0.,  0.,   0.,  0.],
        [ 5.,  1.,   3.,  2.],
        [ 4.,  4.,   4.,  4.],
        [ 4.,  4.,   4.,  4.],
        [10., 90., -70.,  0.]], dtype=torch.float32)
    target = torch.tensor([
        [0.,   0.36, 0.48, 0.16],
        [0.25, 0.25, 0.25, 0.25],
        [0.,   0.,   0.5,  0.5 ],
        [0.,   0.,   1.,   0.  ],
        [0.5,  0.,   0.3,  0.2 ],
        [0.25, 0.25, 0.25, 0.25],
        [0.,   0.5,  0.5,  0.  ],
        [0.05, 0.9,  0.,   0.05]], dtype=torch.float32)
    mask = torch.tensor([
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 1]], dtype=torch.bool)

    print('\ninputs\n', inputs)
    print('target\n', target)
    print('mask\n', mask)

    N, D = inputs.shape
    assert (N, D) == target.shape, 'Inputs must be same shape'
    assert (N, D) == mask.shape, 'Inputs must be same shape'
    assert torch.all(target.sum(axis=1) == 1.), 'Targets must be valid distribution'
    assert torch.all(target >= 0.), 'Targets must be valid distribution'

    Hpq = masked_cross_entropy(inputs, target, mask)
    assert (N, 1) == Hpq.shape

    KLpq = masked_kl_div(inputs, target, mask)
    assert (N, 1) == KLpq.shape

    for i in range(N):
        print('========================')
        print('testing batch example', i)
        m = mask[i]
        log_q = F.log_softmax(inputs[i][m], dim=0)
        print('dist', torch.exp(log_q))
        print('targ', target[i][m])

        Hpqi_expected = -torch.sum(target[i][m] * log_q)
        print('Expected Hpq', Hpqi_expected)
        print('Hpq', Hpq[i])
        assert torch.all(Hpq[i] == Hpqi_expected)

        KLpqi_expected = F.kl_div(log_q, target[i][m], reduction='sum')
        print('Expected KLpq', KLpqi_expected)
        print('KLpq', KLpq[i])
        assert torch.all(KLpq[i] == KLpqi_expected)
