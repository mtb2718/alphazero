import torch
import torch.nn.functional as F

from alphazero.loss import masked_cross_entropy


def test_masked_cross_entropy():

    # TODO: Test case for no valid
    inputs = torch.tensor([
        [ 0.,  0.,   0., -999.],
        [ 0.,  0.,   0.,  0.],
        [ 5.,  1.,   3.,  2.],
        [ 4.,  4.,   4.,  4.],
        [ 4.,  4.,   4.,  4.],
        [10., 90., -70.,  0.]], dtype=torch.float32)
    target = torch.tensor([
        [0.,   0.,   0.5,  0.5 ],
        [0.,   0.,   1.,   0.  ],
        [0.5,  0.,   0.3,  0.2 ],
        [0.25, 0.25, 0.25, 0.25],
        [0.,   0.5,  0.5,  0.  ],
        [0.05, 0.9,  0.,   0.05]], dtype=torch.float32)
    mask = torch.tensor([
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

    result = masked_cross_entropy(inputs, target, mask)
    assert (N, 1) == result.shape

    for i in range(N):
        print('========================')
        print('testing batch example', i)
        m = mask[i]
        log_p_hat = F.log_softmax(inputs[i][m], dim=0)
        print('dist', torch.exp(log_p_hat))
        print('targ', target[i][m])
        H = -torch.sum(target[i][m] * log_p_hat)
        print('cross entropy', H)
        print('result', result[i])
        assert torch.all(result[i] == H)

