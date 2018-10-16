import os
import sys
import itertools
import numpy as np
import tensorflow as tf
import cPickle as pickle

from utils import conversions

_JOINT_OFFSETS = 'helper_data/joint_offsets.pkl'
_TUKEY_PARAMS = 'helper_data/stats/tukey_mad_%d.npy'

class Optimiser():

    def __init__(self, config, model, targets, latent_mean, latent_std):

        self.model = model
        self.targets = targets
        self.predictions = self.model.get_outputs()
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        
        self.loss_full = None
        self.losses         = {}
        self.track_scalars  = {}
        self.track_other    = {}
        self.prepare_loss_ops(config)
        
        self.learning_rate = tf.Variable(0, trainable=False)
        self.global_step = None
        self.max_steps = None
        self.train_op  = None
        self.optimisers = []
        
        return

    def prepare_train_op(self, config, global_step, max_steps):
        assert self.train_op is None, 'train_op already initialised!'
        self.global_step = global_step
        self.max_steps = max_steps
        self.create_optimiser(config)
        return self.train_op
    
    def get_loss_op(self):
        return self.loss_full

    def get_loss_dict(self):
        return self.losses                

    def prepare_loss_ops(self, config):
        
        #TODO: rename
        z = self.predictions['latent']
        jointsloc_betas_pred = self.predictions['joints3D_prepose']
        jointsloc_posed_pred = self.predictions['joints3D']
        jointsloc_proj_pred = self.predictions['joints2D']
        
        smplparams = self.targets.smplparams
        jointsloc_betas = self.targets.joint_locations_betas
        jointsloc_posed = self.targets.joint_locations_posed
        jointsloc_proj = self.targets.joint_locations_projected
        latent_flags = self.targets.latent_flag
        
        with tf.name_scope("generator_loss"):
            # prepare placeholders for losses
            gen_loss = tf.constant(0, tf.float32)
            gen_loss_latent = tf.constant(0, tf.float32)
            gen_loss_jointsloc_betas = tf.constant(0, tf.float32)
            gen_loss_jointsloc_posed = tf.constant(0, tf.float32)
            gen_loss_jointsloc_proj = tf.constant(0, tf.float32)
            gen_loss_weights = tf.constant(0, tf.float32)
            sqerr_latent = tf.zeros_like(z)
            gen_msqerr_latent = tf.constant(0, tf.float32)
            gen_msqerr_shape = tf.constant(0, tf.float32)
            gen_msqerr_pose = tf.constant(0, tf.float32)
            gen_msqerr_trans = tf.constant(0, tf.float32)
            gen_quat_dist = tf.constant(0, tf.float32)
            gen_dist_jointsloc_betas = tf.constant(0, tf.float32)
            gen_dist_jointsloc_3D = tf.constant(0, tf.float32)
            gen_dist_jointsloc_2D = tf.constant(0, tf.float32)
            learning_rate = tf.constant(0, tf.float32)

            # helpers
            sqeuclidist = lambda pred, gt: tf.reduce_sum(sqerr(pred, gt), axis=2)

            with open(_JOINT_OFFSETS, 'rb') as f:
                joint_offsets = pickle.load(f)['euclidist']['std']

            num_landmarks = config['num_landmarks']
            mad_estimate = np.load(_TUKEY_PARAMS % (num_landmarks))
            mad_est_jlps = tf.constant(mad_estimate[72:72+3*num_landmarks].reshape((num_landmarks, 3)))
            mad_est_jlpj = tf.constant(mad_estimate[72+3*num_landmarks:].reshape((num_landmarks, 2)))
            
            if config['num_landmarks'] in [14, 91]:
                joint_offsets = tf.Variable(initial_value=joint_offsets, trainable=False, name="joint_offsets",
                                            expected_shape=(config['num_landmarks'], 1), dtype=tf.float32)
            else:
                joint_offsets = (config['interm_size']/2.) * tf.ones_like(jointsloc_proj[:,:,0])
            
            t_const = tf.constant(4.6851, tf.float32)
            # neck_position = jointsloc_proj[:,12,:]
            # avghip_position = 0.5 * (jointsloc_proj[:,2,:] + jointsloc_proj[:,3,:]) 
            # define loss functions
            loss_funcs = {}
            sqerr = lambda pred, gt: tf.square(pred - gt)
            proj_norm = lambda x: x / (config['interm_size']/2.) - 1
            loss_funcs['mabserr']     = lambda pred, gt: tf.abs(pred-gt)
            loss_funcs['msqerr']      = lambda pred, gt: sqerr(pred, gt)
            loss_funcs['gmc_var']     = lambda pred, gt: sqerr(pred, gt)/(self.latent_std+sqerr(pred, gt))
            loss_funcs['gmc_std']     = lambda pred, gt: sqerr(pred, gt)/(tf.square(self.latent_std)+sqerr(pred, gt)) 
            loss_funcs['euclidist']   = lambda pred, gt: tf.sqrt(sqeuclidist(pred, gt))
            loss_funcs['sqeuclidist'] = lambda pred, gt: sqeuclidist(pred, gt)
            loss_funcs['euclidist_robust_v1'] =\
                                        lambda pred, gt: tf.nn.relu(tf.sqrt(sqeuclidist(pred, gt)) - proj_norm(joint_offsets))
            loss_funcs['tukey_below'] = lambda residual: tf.square(t_const)/6 * (1 - tf.pow(1 - tf.square(residual/t_const), 3))
            loss_funcs['tukey_above'] = lambda residual: tf.square(t_const)/6 * tf.ones_like(residual)
            #ALSO SUPPORTED: '*_simtrans_neck', '*_simtrans_hips' (extensions to *euclidist*-based losses)
            
            def tukey_loss(residual, weights):
                 residual = tf.divide(residual, weights)
                 residual = tf.squeeze(tf.reshape(residual, [-1]))
                 t_a = loss_funcs['tukey_below'](tf.gather(residual, tf.where(tf.less_equal(residual, t_const))))
                 t_b = loss_funcs['tukey_above'](tf.gather(residual, tf.where(tf.greater(residual, t_const))))
                 return tf.concat((t_a, t_b), axis=0)
            
            z_subset = tf.gather(z, tf.squeeze(tf.where(latent_flags)), axis=0)
            smplparams_subset = tf.gather(smplparams, tf.squeeze(tf.where(latent_flags)), axis=0)

            # compute and collect losses
            loss_terms = {}
            if 'loss_terms' in config.keys():
                loss_terms = config["loss_terms"]
            for k in loss_terms.keys():
                if k == 'model_parameters':
                    gen_loss_latent = tf.reduce_mean(loss_funcs[loss_terms[k][0]](z_subset, smplparams_subset))
                    gen_loss += loss_terms[k][1] * gen_loss_latent
                elif k == 'joint_locations_betas':
                    gen_loss_jointsloc_betas = tf.reduce_mean(loss_funcs[loss_terms[k][0]](jointsloc_betas_pred, jointsloc_betas))
                    gen_loss += loss_terms[k][1] * gen_loss_jointsloc_betas
                elif k == 'joint_locations_posed':
                    gen_loss_jointsloc_posed = loss_funcs[loss_terms[k][0].split('_')[0]](jointsloc_posed_pred, jointsloc_posed)
                    if 'tukey' in loss_terms[k][0]:
                        gen_loss_jointsloc_posed = tukey_loss(gen_loss_jointsloc_posed, mad_est_jlps)
                    gen_loss_jointsloc_posed = tf.reduce_mean(gen_loss_jointsloc_posed)
                    gen_loss += loss_terms[k][1] * gen_loss_jointsloc_posed
                elif k == 'joint_locations_projected':
                    loss_type = loss_terms[k][0]
                    if 'simtrans' in loss_type:
                        #TODO: should apply to any nr. of landmarks, not just 14, 91
                        assert config['num_landmarks'] in [14, 91], "loss type 'euclidist_simtrans_*' is incompatible with the SMPL joints" 
                        if 'neck' in loss_type:
                            offset_gt   = jointsloc_proj[:,12,:]      - config['interm_size']/2.
                            offset_pred = jointsloc_proj_pred[:,12,:] - config['interm_size']/2.
                        elif 'hips' in loss_type:
                            offset_gt   = 0.5 * (jointsloc_proj[:,2,:] + jointsloc_proj[:,3,:])           - config['interm_size']/2.
                            offset_pred = 0.5 * (jointsloc_proj_pred[:,2,:] + jointsloc_proj_pred[:,3,:]) - config['interm_size']/2.
                        offset_gt = tf.expand_dims(offset_gt, axis=1)
                        offset_pred = tf.expand_dims(offset_pred, axis=1)
                        loss_type = loss_type.split('_simtrans')[0]
                        jointsloc_proj_pred = jointsloc_proj_pred - offset_pred
                        jointsloc_proj = jointsloc_proj - offset_gt
                    gen_loss_jointsloc_proj  = loss_funcs[loss_type.split('_')[0]](proj_norm(jointsloc_proj_pred), proj_norm(jointsloc_proj))
                    if 'tukey' in loss_type:
                        gen_loss_jointsloc_proj  = tukey_loss(gen_loss_jointsloc_proj, mad_est_jljs)
                    gen_loss_jointsloc_proj = tf.reduce_mean(gen_loss_jointsloc_proj)
                    gen_loss += loss_terms[k][1] * gen_loss_jointsloc_proj
            
            # compute other quantities for tracking performances
            # (1) joint distances
            gen_dist_jointsloc_betas = 100 * tf.reduce_mean(loss_funcs['euclidist'](jointsloc_betas_pred, jointsloc_betas))
            gen_dist_jointsloc_3D    = 100 * tf.reduce_mean(loss_funcs['euclidist'](jointsloc_posed_pred, jointsloc_posed))
            gen_dist_jointsloc_2D    = 1.0*config['input_size']/config['interm_size'] * tf.reduce_mean(loss_funcs['euclidist'](jointsloc_proj_pred, jointsloc_proj))

            # (2) mean squared error on latent params
            sqerr_latent = sqerr(z, smplparams)
            gen_msqerr_latent = tf.reduce_mean(sqerr_latent)
            
            if 'shape' in config["latent_components"]:
                # (3) mean squared error on latent shape params
                gen_msqerr_shape = tf.reduce_mean(sqerr_latent[:, :10])
                if 'pose' in config["latent_components"]:
                    gen_msqerr_pose  = tf.reduce_mean(sqerr_latent[:, 10:226])
                    if 'trans' in config["latent_components"]:
                        gen_msqerr_trans = tf.reduce_mean(sqerr_latent[:, 226:])
            elif 'pose' in config["latent_components"]:
                # (4) mean squared error on latent pose params
                gen_msqerr_pose  = tf.reduce_mean(sqerr_latent[:, :216])
                if 'trans' in config["latent_components"]:
                    gen_msqerr_trans = tf.reduce_mean(sqerr_latent[:, 216:])

            # (5) quaternion distance
            if 'latent_components' not in config.keys() or 'pose' in config['latent_components']:
                if np.all(config['latent_components'] == ['pose']):
                    assert len(config['latent_components']) == 1
                    z_pose = z + self.latent_mean
                    smplparams_pose = smplparams + self.latent_mean
                else:
                    z_pose = z[:,10:226] + self.latent_mean[10:226]
                    smplparams_pose  = smplparams[:,10:226] + self.latent_mean[10:226]
                
                z_quat = tf.py_func(conversions.rotmat_to_quaternion, [tf.reshape(z_pose, [-1])], tf.float32)
                smplparams_quat = tf.py_func(conversions.rotmat_to_quaternion, [tf.reshape(smplparams_pose, [-1])], tf.float32)
                z_quat = tf.reshape(z_quat, [-1, 4])
                smplparams_quat = tf.reshape(smplparams_quat, [-1, 4])
                gen_quat_dist = tf.acos(2 * tf.square(tf.reduce_sum(tf.multiply(z_quat, smplparams_quat), axis=1)) - 1)
                gen_quat_dist = tf.reduce_mean(gen_quat_dist)

            # (6) weight decay
            if (config["weight_decay"] > 0):
                gen_loss_weights = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name or 'filter' in v.name])
                gen_loss += config["weight_decay"] * gen_loss_weights

            # (7) for debugging joint locations
            if False:
                gen_loss_jointsloc_posed = tf.reduce_mean(loss_funcs['msqerr'](jointsloc_proj_pred, 
                                           tf.py_func(compute_mpii_joint_locations, [z_pose, jointsloc_betas_pred, jointsloc_posed_pred, z[:,:10] + self.latent_mean[:10]], tf.float32)))
            
        ##### COLLECT LOSSES #####
        self.loss_full = gen_loss

        self.losses['gen_loss_latent'] = gen_loss_latent
        self.losses['gen_loss_weights'] = gen_loss_weights
        self.losses['gen_loss_jointsloc_betas'] = gen_loss_jointsloc_betas
        self.losses['gen_loss_jointsloc_posed'] = gen_loss_jointsloc_posed
        self.losses['gen_loss_jointsloc_proj'] = gen_loss_jointsloc_proj
        
        self.track_scalars['gen_msqerr_latent'] = gen_msqerr_latent
        self.track_scalars['gen_msqerr_shape'] = gen_msqerr_shape
        self.track_scalars['gen_msqerr_pose'] = gen_msqerr_pose
        self.track_scalars['gen_msqerr_trans'] = gen_msqerr_trans
        self.track_scalars['gen_quat_dist'] = gen_quat_dist
        self.track_scalars['gen_dist_jointsloc_betas'] = gen_dist_jointsloc_betas
        self.track_scalars['gen_dist_jointsloc_3D'] = gen_dist_jointsloc_3D
        self.track_scalars['gen_dist_jointsloc_2D'] = gen_dist_jointsloc_2D
        
        self.track_other['latent_sqerr'] = sqerr_latent

        """
        joint_locations_posed=jointsloc_posed_pred,
        joint_locations_projected=jointsloc_proj_pred,
        train=train,
        global_step=global_step,
        learning_rate=learning_rate,
        """

    def create_optimiser(self, config):

        incr_global_step = tf.assign(self.global_step, self.global_step+1)

        # For batchnorm running statistics updates.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        generator_dependencies = update_ops
        # Optimization
        with tf.name_scope("generator_train"):
            with tf.control_dependencies(generator_dependencies):
                gen_tvars = [var for var in tf.trainable_variables()
                             if var.name.startswith("generator")]
                if config['lr_policy'] == 'poly':
                    self.learning_rate = tf.train.polynomial_decay(config["lr"], self.global_step, self.max_steps, 
                                                                   end_learning_rate=0, power=0.9, cycle=False)
                elif config['lr_policy'] == 'step':
                    self.learning_rate = tf.train.piecewise_constant(global_step, config["lr_boundaries"], 
                                                                     config["lr_steps"])
                else:
                    self.learning_rate = tf.Variable(config["lr"], trainable=False)
                if config['optimizer'] == 'adam':
                    self.optimisers.append(tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                  beta1=config["beta1"]))
                elif config['optimizer'] == 'momentum':
                    self.optimisers.append(tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                      momentum=config["momentum"]))
                else:
                    raise NotImplementedError('Only supported optimizers are "adam" and "momentum"') 
                train = self.optimisers[0].minimize(self.loss_full, var_list=gen_tvars)
        
        self.train_op = tf.group(incr_global_step, train)
        return

    def get_learning_rate(self):
        return self.learning_rate

    def get_losses(self):
        return self.losses
        
    def get_scalars_to_track(self):
        return self.track_scalars

    def get_otherdata_to_track(self):
        return self.track_other
