import os
import sys
import numpy as np
import tensorflow as tf

from utils import conversions
from up_tools.model import landmark_mesh_91

from config import UP_FP, SMPL_FP

sys.path.insert(0, SMPL_FP)
from smpl_webuser.serialization import load_model

_MODEL_NEUTRAL_FNAME = os.path.join(UP_FP, 'models', '3D', 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

class SMPL():

    def __init__(self, config):
    
        self.n_smpl_shape_params = 10
        self.n_smpl_joints = 24
        self.n_smpl_pose_params = self.n_smpl_joints * (3*3)
        
        self.num_landmarks = config['num_landmarks']
        self.use_absrot = config['use_absrot']
        
        self.input_size = config['input_size']
        self.interm_size = config['interm_size']
        self.focal_length = config['focal_length']
        
        self.model = self.load_source_model()
        self.initialise_smpl_variables()
        
        return
    
    def load_source_model(self):
        return load_model(_MODEL_NEUTRAL_FNAME)
    
    def get_smpl_joint_locations(self, smpl_shape_pred, smpl_pose_pred, smpl_trans_pred):
        with tf.variable_scope("smpl"):
            smpl_joints3D_prepose = self.get_joints_in_neutral_pose(smpl_shape_pred)
            if self.use_absrot:
                smpl_joints3D, rt_mats_abs_verts = self.get_smpl_joints_from_abs_rot_matrices(smpl_joints3D_prepose, smpl_pose_pred)
            else:
                smpl_joints3D, rt_mats_abs_verts = self.get_smpl_joints_from_rel_rot_matrices(smpl_joints3D_prepose, smpl_pose_pred)
              
            joints3D = self.get_posed_joints(smpl_joints3D, smpl_shape_pred, rt_mats_abs_verts, smpl_pose_pred)
            joints2D = self.get_projected_joints(joints3D, smpl_trans_pred)

        return smpl_joints3D_prepose, joints3D, joints2D
    
    def initialise_smpl_variables(self):
        with tf.variable_scope("smpl"):
            Jdirs = np.dstack([self.model.J_regressor.dot(self.model.shapedirs[:, :, i]) for i in range(self.n_smpl_shape_params)])
            Jdirs = np.reshape(Jdirs, (self.n_smpl_joints*3, self.n_smpl_shape_params)).transpose()
            Jbias = np.reshape(self.model.J_regressor.dot(self.model.v_template.r), (self.n_smpl_joints*3,))

            self.weights = tf.Variable(initial_value=Jdirs, trainable=False, name="weights", 
                                       expected_shape=(self.n_smpl_shape_params, self.n_smpl_joints*3), 
                                       dtype=tf.float32) 
            self.biases = tf.Variable(initial_value=Jbias, trainable=False, name="biases", 
                                      expected_shape=(self.n_smpl_joints*3,), 
                                      dtype=tf.float32)
                
            kintree_table = self.model.kintree_table
            id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}
            parents = np.array([id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])])
            self.parents = tf.Variable(initial_value=parents, trainable=False, name="parents",
                                       expected_shape=(23,), 
                                       dtype=tf.int32)
            
            pt = 0.5*self.interm_size*np.ones((2,))
            f = self.focal_length/(1.*self.input_size/self.interm_size)
            cam_mtx = np.array([[f, 0, pt[0]], [0, f, pt[1]]])
            self.cam_mtx = tf.Variable(initial_value=cam_mtx, trainable=False, name="cam_mtx", 
                                  expected_shape=(2,3), 
                                  dtype=tf.float32)
            model_trans = np.array([[-0.01,0.115,20.3]])
            model_trans = tf.Variable(initial_value=model_trans, trainable=False, name="model_trans", 
                                      expected_shape=(1,3,), 
                                      dtype=tf.float32)

            missing_vertices = [landmark_mesh_91['neck'], landmark_mesh_91['head_top']]
            if self.num_landmarks == 91:
                missing_vertices.extend([landmark_mesh_91[k] for k in landmark_mesh_91.keys() if k not in ['neck', 'head_top']])
            shapedirs_mpii = self.model.shapedirs[missing_vertices, :, :self.n_smpl_shape_params]
            self.shapedirs_mpii = tf.Variable(initial_value=shapedirs_mpii, trainable=False, name="shapedirs_mpii", 
                                              expected_shape=(len(missing_vertices),3,self.n_smpl_shape_params), 
                                              dtype=tf.float32)

            posedirs_mpii = self.model.posedirs[missing_vertices, :, :]
            self.posedirs_mpii = tf.Variable(initial_value=posedirs_mpii, trainable=False, name="posedirs_mpii", 
                                             expected_shape=(len(missing_vertices),3,207), 
                                             dtype=tf.float32)

            vertices_mpii = self.model.v_template.r[missing_vertices, :]
            self.vertices_mpii = tf.Variable(initial_value=vertices_mpii, trainable=False, name="vertices_mpii", 
                                             expected_shape=(len(missing_vertices),3), 
                                             dtype=tf.float32)
            
            ordering_mpii = np.array([8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20], dtype=int)
            self.ordering_mpii = tf.Variable(initial_value=ordering_mpii, trainable=False, name="ordering_mpii",
                                             expected_shape=(12,), 
                                             dtype=tf.int32)
                                        
            weights_mpii = self.model.weights.r[missing_vertices, :]
            self.weights_mpii = tf.Variable(initial_value=weights_mpii, trainable=False, name="weights_mpii",
                                            expected_shape=(len(missing_vertices),24), 
                                            dtype=tf.float32)
            
            return

    def get_joints_in_neutral_pose(self, smpl_shape_pred):
        joints3D_prepose = tf.add(
                  tf.matmul(smpl_shape_pred, self.weights), #TODO: rename
                  self.biases)
        joints3D_prepose = tf.reshape(joints3D_prepose, [-1, self.n_smpl_joints, 3])
        return joints3D_prepose

    def get_smpl_joints_from_abs_rot_matrices(self, joints3D_prepose, rot_mats_abs):
     
        t_vecs_rel_rot  = tf.concat([joints3D_prepose[:,:1],
                                    tf.squeeze(
                                    tf.matmul(tf.gather(rot_mats_abs, self.parents, axis=1), 
                                              tf.expand_dims(joints3D_prepose[:,1:] - tf.gather(joints3D_prepose, self.parents, axis=1), axis=3)), axis=3)], 
                                    axis=1)

        rt_last_row = tf.tile(tf.constant([[[[0,0,0,1]]]], dtype=tf.float32), [tf.shape(rot_mats_abs)[0], 1, 1, 1])

        rt_mats_abs = tf.concat([rot_mats_abs[:,0], tf.expand_dims(joints3D_prepose[:,0], axis=2)], axis=2)
        rt_mats_abs = tf.expand_dims(tf.concat([rt_mats_abs, tf.squeeze(rt_last_row, axis=(1))], axis=1), axis=1)
        rt_mats_abs_verts = tf.concat([rot_mats_abs[:,0],
                                       tf.expand_dims(joints3D_prepose[:,0], axis=2) - tf.matmul(rot_mats_abs[:,0], 
                                                                                                 tf.transpose(joints3D_prepose[:,:1], [0,2,1]))],
                                       axis=2)
        rt_mats_abs_verts = tf.expand_dims(tf.concat([rt_mats_abs_verts, tf.squeeze(rt_last_row, axis=(1))], axis=1), axis=1)
        c = lambda i, rt_mats_abs, rt_mats_abs_verts: i <= self.parents.shape[0]
        def update_rt_mats_abs(i, rt_mats_abs, rt_mats_abs_verts):
            t_vec_abs = tf.expand_dims(t_vecs_rel_rot[:,i], axis=2) + rt_mats_abs[:,self.parents[i-1],:3,3:]
            rt_mat_abs =  tf.concat([tf.expand_dims(tf.concat([rot_mats_abs[:,i], t_vec_abs], axis=2), axis=1),
                                     rt_last_row], axis=2)
            rt_mats_abs = tf.concat([rt_mats_abs, rt_mat_abs], axis=1)
            current_bs = tf.shape(joints3D_prepose)[0]
            rt_mat_abs_vert = tf.matmul(tf.squeeze(rt_mat_abs, axis=(1)), 
                                        tf.concat([tf.eye(4,3, batch_shape=[current_bs]),
                                                   tf.expand_dims(tf.concat([-joints3D_prepose[:,i], 
                                                                             tf.ones((current_bs,1))], axis=1), 
                                                                  axis=2)], 
                                                  axis=2))
            rt_mats_abs_verts = tf.concat([rt_mats_abs_verts, tf.expand_dims(rt_mat_abs_vert, axis=1)], axis=1)
            return (i + 1, rt_mats_abs, rt_mats_abs_verts)

        loop_vars = (tf.constant(1), rt_mats_abs, rt_mats_abs_verts)
        loop_outp = tf.while_loop(c, update_rt_mats_abs, loop_vars, parallel_iterations=1,
                                  shape_invariants=(tf.constant(1).get_shape(),
                                                    tf.TensorShape([None,None,4,4]),
                                                    tf.TensorShape([None,None,4,4])))
        rt_mats_abs_verts = loop_outp[2]
        joints3D = loop_outp[1][:,:,:3,3]
        
        return joints3D, rt_mats_abs_verts
     
     
    def get_smpl_joints_from_rel_rot_matrices(self, joints3D_prepose, rot_mats):

        t_vecs_rel  = tf.concat([joints3D_prepose[:,:1], 
                                 joints3D_prepose[:,1:] - tf.gather(joints3D_prepose, self.parents, axis=1)], 
                                 axis=1)
        rt_mats_rel = tf.concat([rot_mats, 
                                 tf.expand_dims(t_vecs_rel, axis=3)], 
                                 axis=3)
        rt_mats_rel = tf.concat([rt_mats_rel, 
                                 tf.tile(tf.constant([[[[0,0,0,1]]]], dtype=tf.float32), [tf.shape(rot_mats)[0], self.n_smpl_joints, 1, 1])],
                                 axis=2)

        t_vecs_rel_verts  = tf.concat([joints3D_prepose[:,:1] - tf.transpose(tf.matmul(rot_mats[:,0], tf.transpose(joints3D_prepose[:,:1], [0,2,1])), [0,2,1]), 
                                       joints3D_prepose[:,1:] - tf.gather(joints3D_prepose, self.parents, axis=1)
                                           - tf.squeeze(tf.matmul(rot_mats[:,1:], tf.expand_dims(joints3D_prepose[:,1:], axis=3)), axis=3)], 
                                       axis=1)
        rt_mats_rel_verts = tf.concat([rot_mats, 
                                       tf.expand_dims(t_vecs_rel_verts, axis=3)], 
                                       axis=3)
        rt_mats_rel_verts = tf.concat([rt_mats_rel_verts, 
                                       tf.tile(tf.constant([[[[0,0,0,1]]]], dtype=tf.float32), [tf.shape(rot_mats)[0], self.n_smpl_joints, 1, 1])],
                                       axis=2)

        rt_mats_abs = tf.zeros_like(rt_mats_rel[:,0], dtype=tf.float32)
        rt_mats_abs = tf.expand_dims(rt_mats_rel[:,0], axis=1)
        rt_mats_abs_verts = tf.zeros_like(rt_mats_rel_verts[:,0], dtype=tf.float32)
        rt_mats_abs_verts = tf.expand_dims(rt_mats_rel_verts[:,0], axis=1)
        c = lambda i, rt_mats_abs, rt_mats_rel, rt_mats_abs_verts, rt_mats_rel_verts: i <= self.parents.shape[0]
        def update_rt_mats_abs(i, rt_mats_abs, rt_mats_rel, rt_mats_abs_verts, rt_mats_rel_verts): 
            rt_mats_abs_ext = tf.matmul(rt_mats_abs[:,self.parents[i-1]], rt_mats_rel[:,i])
            rt_mats_abs = tf.concat([rt_mats_abs, tf.expand_dims(rt_mats_abs_ext, axis=1)], axis=1)
            rt_mats_abs_verts_ext = tf.matmul(rt_mats_abs[:,self.parents[i-1]], rt_mats_rel_verts[:,i])
            rt_mats_abs_verts = tf.concat([rt_mats_abs_verts, tf.expand_dims(rt_mats_abs_verts_ext, axis=1)], axis=1)
            return (i + 1, rt_mats_abs, rt_mats_rel, rt_mats_abs_verts, rt_mats_rel_verts)

        loop_vars = (tf.constant(1), rt_mats_abs, rt_mats_rel, rt_mats_abs_verts, rt_mats_rel_verts)
        loop_outp = tf.while_loop(c, update_rt_mats_abs, loop_vars, parallel_iterations=1,
                                  shape_invariants=(tf.constant(1).get_shape(),
                                                    tf.TensorShape([None,None,4,4]),
                                                    tf.TensorShape([None,self.n_smpl_joints,4,4]),
                                                    tf.TensorShape([None,None,4,4]),
                                                    tf.TensorShape([None,self.n_smpl_joints,4,4])))

        rt_mats_abs_verts = loop_outp[3]
        joints3D = loop_outp[1][:,:,:3,3]
        
        return joints3D, rt_mats_abs_verts


    def get_posed_joints(self, joints3D, smpl_shape_pred, rt_mats_abs_verts, rot_mats):

        if self.num_landmarks in [14, 91]:
            if self.use_absrot:
                rot_mats_rel = tf.concat([tf.expand_dims(rot_mats[:,0], axis=1),
                                          tf.matmul(tf.gather(tf.transpose(rot_mats, (0,1,3,2)),
                                                              self.parents, axis=1), rot_mats[:,1:])], axis=1)
                pose_mapped = tf.reshape(rot_mats_rel[:,1:] - tf.eye(3, batch_shape=[23]), [-1, 207])
            else:
                pose_mapped = tf.reshape(rot_mats[:,1:] - tf.eye(3, batch_shape=[23]), [-1, 207])
            vertices_mpii_shaped = self.vertices_mpii + tf.reduce_sum(tf.tile(tf.expand_dims(self.shapedirs_mpii, axis=0), (tf.shape(smpl_shape_pred)[0],1,1,1)) 
                                                                 * tf.expand_dims(tf.expand_dims(smpl_shape_pred, axis=1), axis=1), axis=3)
            vertices_mpii_posed = vertices_mpii_shaped + tf.reduce_sum(tf.tile(tf.expand_dims(self.posedirs_mpii, axis=0), (tf.shape(pose_mapped)[0],1,1,1)) 
                                                                       * tf.expand_dims(tf.expand_dims(pose_mapped, axis=1), axis=1), axis=3)

            T = tf.reduce_sum(tf.expand_dims(tf.transpose(rt_mats_abs_verts, [0, 2, 3, 1]), 4) * tf.transpose(self.weights_mpii), axis=3)
            vpt = tf.transpose(tf.concat((vertices_mpii_posed, tf.ones((tf.shape(vertices_mpii_posed)[0], self.num_landmarks-12, 1))), axis=2), [0, 2, 1])
            vpt = tf.transpose(tf.reduce_sum((tf.expand_dims(vpt, axis=1) * T), axis=2), [0,2,1])[:,:,:3]
                                                                     
            joints3D = tf.gather(joints3D, self.ordering_mpii, axis=1)
            joints3D = tf.concat((joints3D, vpt), axis=1)
        
        return joints3D
    
    def get_projected_joints(self, joints3D, smpl_trans_pred):
        joints3D_trans = joints3D + tf.expand_dims(smpl_trans_pred, axis=1)

        joints2D = joints3D_trans / joints3D_trans[:,:,2:]
        joints2D = tf.reshape(tf.transpose(
                         tf.matmul(self.cam_mtx[:], 
                                   tf.transpose(tf.reshape(joints2D,
                                                           [-1, 3]))
                                   )), [-1,self.num_landmarks,2])

        return joints2D

