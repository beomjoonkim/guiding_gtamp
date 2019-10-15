from generators.learning.utils.model_creation_utils import create_imle_model
from generators.learning.weight_regularizers.gershgorin_regularizer import gershgorin_reg

import numpy as np
import tensorflow as tf


def test_correctness():
    seed = 1
    #test_mat = np.array([[1, -1, -1, -1], [4, 1, 2, 3]])
    policy = create_imle_model(seed)
    layers = policy.policy_model.layers

    gershgorin_vals = 1
    sess = tf.Session()
    for layer in layers:
        if len(layer.get_weights()) == 0:
            continue
        weight = layer.get_weights()[0].squeeze()
        if len(weight.shape) <= 1:
            continue
        test_mat = tf.constant(weight, dtype=tf.float32)
        loss = gershgorin_reg(test_mat).eval(session=sess)
        gershgorin_vals *= loss
        print gershgorin_vals

    # Currently, the values on some layers are zero
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    test_correctness()
