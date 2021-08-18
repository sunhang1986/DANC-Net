"""
	Based on the PointNet++ codebase 
    https://github.com/charlesq34/pointnet2
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import tf_util_srn
import tensorflow.contrib.slim as slim


def SupConLoss(features, labels=None, mask=None):
    """Compute loss for model. If both `labels` and `mask` are None,
            it degenerates to SimCLR unsupervised loss:
            https://arxiv.org/pdf/2002.05709.pdf

            Args:
                features: hidden vector of shape [bsz, n_views, ...].
                labels: ground truth of shape [bsz].
                mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                    has the same class as sample i. Can be asymmetric.
            Returns:
                A loss scalar.
            """
    features = tf.convert_to_tensor(features)
    labels = tf.convert_to_tensor(labels) if labels is not None else None

    temperature = 0.1
    contrast_mode = 'all'
    base_temperature = 0.07
    batch_size, n_views, C = features.get_shape().as_list()
    print(features)


    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = tf.reshape(features, [batch_size, n_views, -1])

    if features.dtype != tf.float32:
        features = tf.cast(features, tf.float32)

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = tf.eye(batch_size, dtype=tf.float32)
    elif labels is not None:
        labels = tf.reshape(labels, [-1, 1])
        print(labels.shape)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = tf.equal(labels, tf.transpose(labels))
        mask = tf.cast(mask, tf.float32)
    else:
        mask = tf.cast(mask, tf.float32)

    contrast_count = n_views
    contrast_feature = tf.reshape(
        tf.transpose(features, perm=[1, 0, 2]),
        [n_views * batch_size, -1])
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))
    # compute logits
    temperature = tf.cast(temperature, tf.float32)
    base_temperature = tf.cast(base_temperature, tf.float32)
    anchor_dot_contrast = tf.matmul(anchor_feature, tf.transpose(contrast_feature)) / temperature
    # for numerical stability
    logits = (anchor_dot_contrast - tf.reduce_max(tf.stop_gradient(anchor_dot_contrast), axis=1, keep_dims=True))

    # tile mask
    mask = tf.tile(mask, [anchor_count, contrast_count])
    # mask-out self-contrast cases

    logits_mask = tf.ones_like(mask)
    mask2 = tf.diag(tf.ones(mask.shape[0]))
    logits_mask = logits_mask - mask2
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.log(tf.reduce_sum(exp_logits, axis=1, keep_dims=True)) # 求和但不降维

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / tf.reduce_sum(mask, axis=1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]), axis=0)
    return loss

def NonLocalBlock1(input, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock1'):
    batchsize, width, height, in_uchannels, = input.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('g1') as scope:
            g1 = slim.conv2d(input, out_channels, [1,1], stride=1, scope='g1')
            # print("g1", g1.shape)
            # if sub_sample:
            #     g = slim.max_pool2d(g, [2,2], stride=2, scope='g_max_pool')

        with tf.variable_scope('phi') as scope:
            phi = slim.conv2d(input, out_channels, [1,1], stride=1, scope='phi')
            # if sub_sample:
            #     phi = slim.max_pool2d(phi, [2,2], stride=2, scope='phi_max_pool')

        with tf.variable_scope('theta') as scope:
            theta = slim.conv2d(input, out_channels, [1,1], stride=1, scope='theta')

        g_x = tf.reshape(g1, [batchsize,-1, width*height])
        g_x = tf.transpose(g_x, [0,2,1])                             # proj_value
        print("g_x", g_x.shape)

        theta_x = tf.reshape(theta, [batchsize, -1, width*height])   # proj_key
        theta_x = tf.transpose(theta_x, [0,2,1])
        print("theta_x", theta_x.shape)
        phi_x = tf.reshape(phi, [batchsize, -1, width*height])      # proj_query
        print("phi_x", phi_x.shape)

        f = tf.matmul(g_x, phi_x)  # energy
        print("f", f.shape)
        # ???
        f_softmax = tf.nn.softmax(f, -1)   # attention
        print("f_softmax", f_softmax.shape)
        y = tf.matmul(f_softmax, theta_x)
        print("y", y.shape)
        y = tf.reshape(y, [batchsize, width, height, out_channels])
        print("y", y.shape)
        with tf.variable_scope('w') as scope:
            w_y = slim.conv2d(y, in_uchannels, [1,1], stride=1, scope='w')
            print("w_y", y.shape)
        z = input + w_y
        print("z", z.shape)
        return z

def NonLocalBlock(input, out_channels, sub_sample, is_bn, scope='NonLocalBlock'):
    batchsize, width, height, in_uchannels, = input.get_shape().as_list()

    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('g') as scope:
            g = slim.conv2d(input, out_channels, [1,1], stride=1, scope='g')
            # print("g", g.shape)
            if sub_sample:
                g = slim.max_pool2d(g, [1,1], stride=1, padding='same', scope='g_max_pool')
                # print("g", g.shape)

        with tf.variable_scope('phi') as scope:
            phi = slim.conv2d(input, out_channels, [1,1], stride=1, scope='phi')
            if sub_sample:
                phi = slim.max_pool2d(phi, [1,1], stride=1, padding='same', scope='phi_max_pool')
                # print("phi", phi.shape)

        with tf.variable_scope('theta') as scope:
            theta = slim.conv2d(input, out_channels, [1,1], stride=1, scope='theta')

        g_x = tf.reshape(g, [batchsize,-1, width*height])
        g_x = tf.transpose(g_x, [0,2,1])                             # proj_value
        print("g_x", g_x.shape)

        theta_x = tf.reshape(theta, [batchsize, -1, width*height])   # proj_key
        theta_x = tf.transpose(theta_x, [0,2,1])
        print("theta_x", theta_x.shape)
        phi_x = tf.reshape(phi, [batchsize, -1, width*height])      # proj_query
        print("phi_x", phi_x.shape)

        f = tf.matmul(g_x, phi_x)  # energy
        print("f", f.shape)
        # ???
        f_softmax = tf.nn.softmax(f, -1)   # attention
        print("f_softmax", f_softmax.shape)
        y = tf.matmul(f_softmax, theta_x)
        print("y", y.shape)
        y = tf.reshape(y, [batchsize, width, height, out_channels])
        print("y", y.shape)
        with tf.variable_scope('w') as scope:
            w_y = slim.conv2d(y, in_uchannels, [1,1], stride=1, scope='w')
            print("w_y", w_y.shape)
            if is_bn:
                w_y = slim.batch_norm(w_y)
                print("w_y", w_y.shape)
        z = input + w_y
        print("z", z.shape)
        return z

def RCAB(input, reduction):
    """
    @Image super-resolution using very deep residual channel attention networks
    Residual Channel Attention Block
    """
    batch, height, width, channel = input.get_shape()  # (B, W, H, C)

    x = tf.reduce_mean(input, axis=(1, 2), keep_dims=True)  # (B, 1, 1, C)
    x = tf.layers.conv2d(x, channel // reduction, 1, activation=tf.nn.relu)  # (B, 1, 1, C // r)
    x = tf.layers.conv2d(x, channel, 1, activation=tf.nn.sigmoid)  # (B, 1, 1, C)
    x = tf.multiply(input, x)  # (B, W, H, C)

    x = tf.add(input, x)
    return x

def SRNBlock(relation_u, relation_v, scope, bn, is_training, bn_decay):
    print("relation_u", relation_u.shape)
    print("relation_v", relation_v.shape)
    batchsize, width, height, in_uchannels,  = relation_u.get_shape().as_list()
    _, _, _, in_vchannels = relation_v.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('gu') as scope:
            gu_output = RCAB(relation_u, 3)
            gu_output = NonLocalBlock(gu_output, sub_sample=False, out_channels = 128, is_bn=False)
            # gu_output = RCAB(gu_output, 2)
            gu_output = tf_util_srn.conv2d(gu_output, 387, [1, 1], padding='VALID', stride=[1, 1],bn=bn, is_training=is_training,scope='conv1', bn_decay=bn_decay)
            print(gu_output.shape)
            gu_output = tf.reshape(gu_output, [batchsize, -1, 387])
            print("gu_output", gu_output.shape)

        with tf.variable_scope('gv') as scope:
            gv_output = RCAB(relation_v, 3)
            gv_output = NonLocalBlock(gv_output, sub_sample=False, out_channels=3, is_bn=False)
            gv_output = tf_util_srn.conv2d(gv_output, 387, [1, 1], padding='VALID', stride=[1, 1],bn=bn, is_training=is_training,scope='conv2', bn_decay=bn_decay)
            gv_output = tf.reshape(gv_output, [batchsize, -1, 387])
            print("gv_output", gv_output.shape)

        return gu_output, gv_output


