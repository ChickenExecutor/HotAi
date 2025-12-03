from model.utils import lookahead_mask


def test_lookahead_mask():
    shape = 5
    mask = lookahead_mask(shape)
    assert mask.shape == (shape, shape)
    for i in range(shape):
        for j in range(shape):
            if i >= j:
                assert mask[i, j] == 0
            else:
                assert mask[i, j] == 1