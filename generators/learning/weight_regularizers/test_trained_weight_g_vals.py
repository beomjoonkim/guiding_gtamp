from generators.learning.utils.model_creation_utils import create_imle_model
from generators.learning.weight_regularizers.gershgorin_regularizer import compute_gershgorin_disc_lbs

import numpy as np
import tensorflow as tf


def test_correctness():
    seed = 1
    #test_mat = np.array([[1, -1, -1, -1], [4, 1, 2, 3]])
    policy = create_imle_model(seed)
    layers = policy.policy_model.layers

    gershgorin_lb = 1
    sess = tf.Session()
    for layer in layers:
        #print layer.name
        if len(layer.get_weights()) == 0:
            continue
        weight = layer.get_weights()[0].squeeze()
        test_mat = tf.constant(weight, dtype=tf.float32)
        lb = compute_gershgorin_disc_lbs(test_mat).eval(session=sess)
        gershgorin_lb *= lb
        print layer.name, lb

    # if the gershgorin_lb is less than 0, then it becomes a trivial bound on the output
    # if the difference in z is more than 1, then the output values should differ by 1 too
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    test_correctness()
