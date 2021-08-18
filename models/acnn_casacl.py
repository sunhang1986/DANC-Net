import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util_srn
from pointnet_util import pointnet_sa_module, acnn_module_rings
from SRN import SRNBlock, SupConLoss


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    normals_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl, normals_pl

def get_model(point_cloud, normals, is_training, bn_decay=None):
    """ Classification A-CNN, input is points BxNx3 and normals BxNx3, output Bx40 """
    print(point_cloud.shape, normals.shape, "到底有没有法向量")
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_normals = normals
    l0_points = None


    # Abstraction layers.   第一层提取出512个局部区域，第二层提取出128个局部区域
    l1_xyz, l1_points, l1_normals = acnn_module_rings(l0_xyz, l0_points, l0_normals, 512, [[0.0, 0.1], [0.1, 0.2]], [16,48], [[32,32,64], [64,64,128]], is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_normals = acnn_module_rings(l1_xyz, l1_points, l1_normals, 128, [[0.1, 0.2], [0.3, 0.4]], [16,48], [[64,64,128], [128,128,256]], is_training, bn_decay, scope='layer2')
    #####################
    print("l2_points", l2_points.shape)
    _, npoints, C = l2_points.get_shape().as_list()
    l2_xyz = tf.transpose(l2_xyz, perm=[0, 2, 1])
    _, D, npoint = l2_xyz.get_shape().as_list()

    #########自注意力
    relation_u = tf.reshape(l2_points, [batch_size, npoints, 1, C])
    relation_v = tf.reshape(l2_xyz, [batch_size, npoints, 1, D])

    u_output, v_output = SRNBlock(relation_u, relation_v, scope='layer3', bn=True, is_training=is_training, bn_decay=bn_decay)
    l2_xyz = tf.transpose(l2_xyz, perm=[0, 2, 1])
    print("l2_xyz", l2_xyz.shape)
    l3_points = u_output + v_output
    print("l3_points", l3_points.shape)
    #####################
    _, l4_points, _ = pointnet_sa_module(l2_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Fully connected layers
    net = tf.reshape(l4_points, [batch_size, -1])
    net = tf_util_srn.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util_srn.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net1 = tf_util_srn.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    print(net.shape, "net1")
    net = tf_util_srn.dropout(net1, keep_prob=0.4, is_training=is_training, scope='dp2')
    print(net, "net")
    net = tf_util_srn.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, net1, end_points


# def get_loss(pred, label, end_points):
def get_loss(pred, net1, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    net1 = tf.expand_dims(net1, 1)
    supconloss = SupConLoss(net1, label)
    return classify_loss + supconloss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)