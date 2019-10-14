

import tensorflow as tf


#
def gershgorin_reg(weight_matrix):
    weight_matrix = tf.squeeze(weight_matrix)
    wdotw = tf.matmul(tf.transpose(weight_matrix), weight_matrix)

    ### compute the Gershgorion circles
    # diagonal entries
    diag_vals = tf.linalg.diag_part(wdotw)

    # off-diagnoal entires
    abs_sum_offdiag_vals = tf.reduce_sum(tf.abs(wdotw), axis=1) - tf.abs(diag_vals)

    # compute the non-zero lower bounds, because we know that the eigenvalues of w^T*w is non-negative
    gershgorin_lb = tf.reduce_min(diag_vals - abs_sum_offdiag_vals)

    # regularize it if the lower bound is less than 1, but otherwise don't penalize
    delta = 1.0 - gershgorin_lb
    hinge_loss_on_gershgorin_lb = 5*tf.maximum(tf.cast(0., tf.float32), delta)

    """
    if sess is not None:
        print diag_vals.eval(session=sess)
        print abs_sum_offdiag_vals.eval(session=sess)
        print 'G lower bound', gershgorin_lb.eval(session=sess)
        print "Hinge loss", hinge_loss_on_gershgorin_lb.eval(session=sess)
    """

    return hinge_loss_on_gershgorin_lb
