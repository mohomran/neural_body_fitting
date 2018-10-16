import os
import sys
import logging
import numpy as np
import cPickle as pickle
import scipy.misc as sm
import matplotlib
from PIL import Image

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import tensorflow.contrib.slim as slim
from utils import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from utils import resnet_utils
from tensorflow.python.ops import nn_ops

from utils import conversions

from up_tools.model import landmark_mesh_91
from models.smpl import smpl

LOGGER = logging.getLogger(__name__)

class Model():

    def __init__(self, config, graph, inputs, model_trans, latent_mean, latent_std, is_training=True):
     
        self.encoder_type = config["encoder_type"]
        self.model_graph = graph
        
        self.interm_size = config['interm_size']
        
        self.n_smpl_params = config["nz"]
        
        self.model_trans = model_trans #[-0.01,0.115,20.3]
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        
        self.use_absrot = config['use_absrot'] 
        self.apply_svd_to_rot_mats = config['use_svd']
        self.use_gt_trans_params = True
        
        self.cmap = self.get_colourmap(config)
        
        self.outputs = {}
        
        assert self.encoder_type in ['vgg16', 'resnet50v1', 'resnet101v1', 'resnet50v2'],\
               "Unsupported encoder type: '%s'!" % (self.encoder_type)

        if self.encoder_type == 'vgg16':
            enc_func = lambda x: vgg16(x, is_training=is_training)
        elif self.encoder_type == 'resnet50v1':
            enc_func = lambda x: resnet_v1_50(x, is_training=is_training)
        elif self.encoder_type == 'resnet50v2':
            enc_func = lambda x: resnet_v2_50(x, is_training=is_training)

        with self.model_graph.as_default():
            # (0) initialise SMPL model
            self.smpl_instance = smpl.SMPL(config)
            
            # (1) OPTIONAL: produce part segmentation
            if config['mode'] == 'infer_segment_fit':
                feature_maps = resnet_v1_101(inputs, output_stride=32, is_training=False)
                predictions = refinenet_decoder(feature_maps, num_classes=config['num_classes'], is_training=False)
                predictions = tf.image.resize_images(predictions, [config['input_size'], config['input_size']],
                                                     method=tf.image.ResizeMethod.BILINEAR)
                inputs = tf.py_func(self.apply_colourmap, [predictions], tf.float32) * 2. / 255. - 1.
                inputs.set_shape((predictions.shape[0], self.interm_size, self.interm_size, 3))
            
            # (2) estimate SMPL parameters
            with tf.variable_scope("generator") as sf:
                pre_smpl_latent = enc_func(inputs)
                self.smpl_latent = tf.contrib.layers.fully_connected(tf.squeeze(pre_smpl_latent, [1,2]), self.n_smpl_params, activation_fn=None,
                                                                     weights_initializer=tf.random_normal_initializer(0, 0.02), scope=sf)
            
            # (3) split and postprocess SMPL param estimates
            smpl_shape_pred, smpl_pose_pred, smpl_trans_pred =\
                self.postprocess_smpl_params(self.smpl_instance.n_smpl_shape_params,
                                             self.smpl_instance.n_smpl_pose_params,
                                             self.smpl_instance.n_smpl_joints)
            
            # (4) get joint/landmark locations
            joints3D_prepose, joints3D, joints2D =\
                self.smpl_instance.get_smpl_joint_locations( 
                                              smpl_shape_pred, 
                                              smpl_pose_pred, 
                                              smpl_trans_pred)
        
        self.outputs['latent'] = self.smpl_latent #TODO do we need latter as class var?
        self.outputs['joints3D_prepose'] = joints3D_prepose
        self.outputs['joints3D'] = joints3D
        self.outputs['joints2D'] = joints2D
        self.outputs['intermediate_rep'] = inputs

        return

    #TODO: these two functions need to go somewhere else
    def get_colourmap(self, config):

        cmap = np.squeeze(sm.imread(config['colour_map']))
        cmaplist = [tuple(c/255.0) for c in cmap[:256]]

        return matplotlib.colors.ListedColormap(cmaplist, name='u2p')
    
    def apply_colourmap(self, predictions):
        
        colour_maps = []
        for prediction in predictions:
            labels = prediction.argmax(axis=2)
            im = Image.fromarray(np.uint8(self.cmap((labels)/255.0)*255))
            im.thumbnail((self.interm_size, self.interm_size), resample=0)
            colour_maps.append(np.array(im))
        
        colour_maps = [np.float32(c[None,:,:,:3]) for c in colour_maps]
        return np.concatenate(colour_maps, axis=0)
    
    def get_trainable_vars(self):
        raise NotImplementedError()
        trainable_vars = []
        for c in self.components:
            trainable_vars += c.get_trainable_vars()

        return trainable_vars

    def get_outputs(self):
        return self.outputs

    def postprocess_smpl_params(self, n_shape, n_pose, n_joints):
        smpl_shape_pred = self.smpl_latent[:,:n_shape] + self.latent_mean[:n_shape]
        smpl_pose_pred = tf.reshape(self.smpl_latent[:,n_shape:n_shape+n_pose] + self.latent_mean[n_shape:n_shape+n_pose], [-1, n_joints, 3, 3])
        if self.apply_svd_to_rot_mats:
            w, u, v = tf.svd(smpl_pose_pred, full_matrices=True)
            smpl_pose_pred = tf.matmul(u, tf.transpose(v, perm=[0,1,3,2]))
        if self.use_gt_trans_params:
            smpl_trans_pred = self.model_trans #TODO: fix
        else:
            smpl_trans_pred = tf.concat([self.smpl_latent[:,n_shape+n_pose:] + self.latent_mean[n_shape+n_pose:], self.model_trans[:,2:]], axis=1)

        return smpl_shape_pred, smpl_pose_pred, smpl_trans_pred

#### FITTING NETWORKS ####

def vgg16(inputs, is_training=True):

    dropout_keep_prob = 1.0
    if is_training:
        dropout_keep_prob = 0.5

    with tf.variable_scope("vgg_16"):
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

    return net

def resnet_v1_50(inputs, is_training=True):

    blocks = [
        resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        with tf.variable_scope('resnet_v1_50', 'resnet_v1', [inputs]):
            with slim.arg_scope([slim.conv2d, resnet_v1.bottleneck,
                                 resnet_utils.stack_blocks_dense]):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
                    net = resnet_utils.stack_blocks_dense(net, blocks)
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
    
    return net

def resnet_v2_50(inputs, is_training=True):

    blocks = [
        resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2.resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v2.resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        with tf.variable_scope('resnet_v2_50', 'resnet_v2', [inputs]):
            with slim.arg_scope([slim.conv2d, resnet_v2.bottleneck,
                                 resnet_utils.stack_blocks_dense]):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                        net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                    net = resnet_utils.stack_blocks_dense(net, blocks)
                    # This is needed because the pre-activation variant does not have batch
                    # normalization or activation functions in the residual unit output. See
                    # Appendix of [2].
                    net = slim.batch_norm(net, activation_fn=nn_ops.relu, scope='postnorm')
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)

    return net
    
#### SEGMENTATION NETWORKS ####

def resnet_v1_101(inputs, output_stride=8, is_training=True):

    blocks = [
        resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
        resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),
        resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),
    ]

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        with tf.variable_scope('resnet_v1_101', 'resnet_v1', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, resnet_v1.bottleneck,
                                 resnet_utils.stack_blocks_dense],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    output_stride /= 4
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
                    net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
                    # Convert end_points_collection into a dictionary of end_points.
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)

        outputs = {}
        outputs['conv1'] = end_points['resnet_v1_101/conv1']
        outputs['conv2'] = end_points['resnet_v1_101/block1']
        outputs['conv3'] = end_points['resnet_v1_101/block2']
        outputs['conv4'] = end_points['resnet_v1_101/block3']
        outputs['conv5'] = end_points['resnet_v1_101/block4']

    return outputs

def refinenet_decoder(inputs, num_classes, is_training=True):

    batch_norm_params = {
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,
    'is_training': is_training
    }
    with slim.arg_scope([slim.conv2d], activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        weights_initializer=tf.random_normal_initializer(0, 0.005)):

        block1 = inputs['conv2']
        block2 = inputs['conv3']
        block3 = inputs['conv4']
        block4 = inputs['conv5']

        refinenet4 = refinenet_module(block4, input_higherlevel=None,       depth_in=512)
        refinenet3 = refinenet_module(block3, input_higherlevel=refinenet4, depth_in=256)
        refinenet2 = refinenet_module(block2, input_higherlevel=refinenet3, depth_in=256)
        refinenet1 = refinenet_module(block1, input_higherlevel=refinenet2, depth_in=256)

    pred = slim.conv2d(refinenet1, num_classes, [3, 3], padding='SAME', normalizer_fn=None, scope='logits')
    
    return pred

def residual_conv_unit(input, depth=256):
    net = tf.nn.relu(input)
    net = slim.conv2d(net, depth, [3, 3])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, depth, [3, 3])
    return input + net

def chained_residual_pooling(input, depth=256):
    relu = tf.nn.relu(input)
    net = slim.conv2d(relu, depth, [3, 3])
    pool1 = slim.max_pool2d(net, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(pool1, depth, [3, 3])
    pool2 = slim.max_pool2d(net, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(pool2, depth, [3, 3])
    pool3 = slim.max_pool2d(net, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(pool3, depth, [3, 3])
    pool4 = slim.max_pool2d(net, [5, 5], stride=1, padding='SAME')
    return relu + pool1 + pool2 + pool3 + pool4

def refinenet_module(input, input_higherlevel=None, depth_in=256):
    net = slim.conv2d(input, depth_in, [3, 3])
    net = residual_conv_unit(net, depth=depth_in) + net
    net = residual_conv_unit(net, depth=depth_in) + net
    if input_higherlevel is not None:
        net = slim.conv2d(net, depth_in, [3, 3])
        input_higherlevel = slim.conv2d(input_higherlevel, depth_in, [3, 3])
        input_higherlevel = tf.image.resize_bilinear(input_higherlevel, size=(tf.shape(net)[1], tf.shape(net)[2]))
        net = net + input_higherlevel
    net = chained_residual_pooling(net, depth=depth_in)
    net = residual_conv_unit(net, depth=depth_in) + net
    return net


