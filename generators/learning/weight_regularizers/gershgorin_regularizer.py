

import tensorflow as tf


def compute_gershgorin_disc_lbs(weight_matrix):

    if weight_matrix.shape.ndims == 1:
        gershgorin_lb = tf.tensordot(weight_matrix, weight_matrix, 1)
    else:
        weight_matrix = tf.squeeze(weight_matrix)
        wdotw = tf.matmul(tf.transpose(weight_matrix), weight_matrix)
        diag_vals = tf.linalg.diag_part(wdotw)
        abs_sum_offdiag_vals = tf.reduce_sum(tf.abs(wdotw), axis=1) - tf.abs(diag_vals)
        gershgorin_lb = tf.reduce_min(diag_vals - abs_sum_offdiag_vals)

    return gershgorin_lb


def gershgorin_reg(weight_matrix):
    gershgorin_lb = compute_gershgorin_disc_lbs(weight_matrix)

    # regularize it if the lower bound is less than 1, but otherwise don't penalize
    delta = 1.0 - gershgorin_lb
    hinge_loss_on_gershgorin_lb = tf.maximum(tf.cast(0., tf.float32), delta)

    return hinge_loss_on_gershgorin_lb

