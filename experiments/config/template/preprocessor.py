import os
import sys
import glob
import tqdm
import random
import fnmatch
import logging
import collections
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import skimage.draw as skdraw

from utils import conversions

LOGGER = logging.getLogger(__name__)

DataRawInfer = collections.namedtuple(
    "DataRaw",
    "path, crop, intermediate_rep")

DataRaw = collections.namedtuple(
    "DataRaw",
    "path, crop, intermediate_rep, smplparams, joints, latent_flag")

DataPreprocessed = collections.namedtuple(
    "DataPreprocessed",
    "path, crop, intermediate_rep, smplparams, smplparams_full, smplparams_orig, "
    "joint_locations_betas, joint_locations_posed, joint_locations_projected, "
    "latent_flag")

#TODO: handle nested list
#TODO: handle latent_std properly

_IMAGENET_MEAN = np.array([122.67892, 116.66877, 104.00699])

class Preprocessor():

    def __init__(self, config, mode, latent_mean=None, latent_std=None):

        self.mode = mode

        self.input_size = config['input_size']
        self.interm_size = config['interm_size']
        self.smplparam_components = config['latent_components']
        self.smplparam_len = config['nz']
        self.smplparam_len_full = config['nz_full']
        if config['use_absrot']:
            self.kintree = conversions.prepare_kintree()
        else:
            self.kintree = None
        self.num_joints = config['num_landmarks']
        self.part_colours = np.squeeze(sm.imread(config['colour_map']))
        self.input_type = config['input_type']
        assert self.input_type in ['image', 'partmap', 'jointmap'], 'Unknown input type: %s' % (self.input_type)
        
        self.buffer_size = 768
        self.num_threads = 16
        
        self.latent_mean = latent_mean
        self.latent_std = latent_std

        self.data_list = self.get_data_list(config)
        self.data_list_ph = tf.placeholder(tf.string, 
                                           shape=[len(self.data_list), len(self.data_list[0][0])])
        read_png = lambda x: tf.image.decode_png(tf.read_file(x))[:,:,:3]
        read_bin = lambda x: tf.decode_raw(tf.read_file(x), tf.float64)
        if self.mode in ['infer_segment_fit']:
            read_func = lambda x: DataRawInfer(
                                    path=x[0],
                                    crop=read_png(x[1]),
                                    intermediate_rep=tf.zeros((self.interm_size, self.interm_size, 3)))
        elif self.mode in ['infer_fit']:
            read_func = lambda x: DataRawInfer(
                                    path=x[0],
                                    crop=tf.zeros((self.input_size, self.input_size, 3)),
                                    intermediate_rep=read_png(x[2]))
        else:
            read_func = lambda x: DataRaw(
                                    path=x[0],
                                    crop=tf.zeros((self.input_size, self.input_size, 3)),
                                    intermediate_rep=read_png(x[2]),
                                    smplparams=read_bin(x[3]),
                                    joints=read_bin(x[4]),
                                    latent_flag=x[5])

        dataset = tf.data.Dataset.from_tensor_slices(self.data_list_ph)
        dataset = dataset.map(read_func)
        dataset = dataset.map(self.transform_data, num_parallel_calls=self.num_threads)
        dataset = dataset.prefetch(self.buffer_size)
        self.dataset = dataset.batch(config['batch_size'])

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        self.next_batch = self.iterator.get_next()
        self.iter_init = self.iterator.make_initializer(self.dataset)

        return
    
    def get_batching_op(self):
        return self.next_batch
        
    def initialise_iterator(self, session, shuffle=True):
        
        if shuffle:
            random.shuffle(self.data_list)
        
        session.run(self.iter_init, feed_dict={self.data_list_ph: [random.choice(dl) for dl in self.data_list]})
        return 
   
    def get_next_batch(self, session):
        
        batch = session.run(self.next_batch)
        return batch
        
    def get_jointmap(self, joints):
        img = np.zeros((self.interm_size, self.interm_size, 3))
        joints = np.reshape(joints[72+self.num_joints*3:], (self.num_joints, 2))
        for i in range(self.num_joints):
            rr, cc = skdraw.circle(joints[i,0], joints[i,1], 3, shape=(self.interm_size, self.interm_size))
            for j in range(3):
                img[cc, rr, j] = self.part_colours[i,j]
        
        return img

    def get_data_list(self, config):

        mode = config['mode']
        # TODO: remove var and if block
        joints_dir = 'joints'
        if 'use_human_annotated_joints' in config.keys(
        ) and config['use_human_annotated_joints']:
            joints_dir = 'joints_ha'
        if mode in ['train', 'val', 'trainval', 'eval_train', 'test']:
            dset_dir = config['dset_dir']
            dset_list = config[
                'train_list'] if mode == 'eval_train' else config['%s_list' %
                                                                      (mode)]
            assert os.path.exists(
                dset_list), "Given list doesn't exist: %s" % (dset_list)
            with open(dset_list, 'r') as f:
                fids = f.read().split('\n')
                fids = [fid.split(' ') for fid in fids]                

            if 'num_landmarks' in config.keys(
            ) and config['num_landmarks'] in [14, 91]:
                joints_type = 'joints_%d' % (config['num_landmarks'])
            else:
                joints_type = 'joints'

            data_list = []
  
            if mode in ['eval_train', 'val', 'test']:
                f_grps = [[f] for f in fids]  
            elif mode in ['train', 'trainval']:
                fnames = []
                f_grps = []
                
                for root, dirnames, filenames in os.walk(
                        os.path.join(dset_dir, 'images')):
                    for filename in filenames:
                        full_path = os.path.join(root, filename)
                        if fnmatch.filter([full_path], '*_image.png'):
                            fnames.append(os.path.join(root, filename))
                fnames = sorted([
                    fname.split('_image.png')[0].split(
                        os.path.join(dset_dir, 'images'))[1][1:] for fname in fnames
                ])

                f_ctr = 0
                for fid in tqdm.tqdm(fids):
                    assert np.any([(fid[0] in fname) for fname in fnames[f_ctr:]]),\
                        'No files found in specified folder (%s) corresponding to id: %s' % (dset_dir, fid[0])
                    f_grp = []
                    while f_ctr < len(fnames):
                        fname = fnames[f_ctr]
                        if fid[0] in fname:
                            f_grp.append((fname, fid[1]))
                            f_ctr += 1
                        else:
                            if len(f_grp) == 0:
                                f_ctr += 1
                            else:
                                break
                    if len(f_grp) > 0:
                        f_grps.append(f_grp)   
           
            for f_grp in f_grps:
                data_list.append([(f[0],
                                   os.path.join(dset_dir, 'images',
                                                f[0] + '_image.png'),
                                   os.path.join(dset_dir, 'colours',
                                                f[0] + '_colours.png'),
                                   os.path.join(dset_dir, 'smplparams',
                                                f[0] + '_smplparams.bin'),
                                   os.path.join(dset_dir, 'joints',
                                                f[0] + '_' + joints_type + '.bin'),
                                   str(f[1])) for f in f_grp])
        
        elif mode in ['infer_fit', 'infer_segment_fit']:
            inp_files = sorted(glob.glob(os.path.join(config['inp_fp'], '*.png')))
            fids = [os.path.basename(ifl).split('.png')[0] for ifl in inp_files]
            data_list = [[(f, 
                           os.path.join(config['inp_fp'], f + '.png'))]
                         for f in fids]
        return data_list
        
    def get_num_samples(self):
        assert self.data_list is not None, "data_list not yet set"
        return len(self.data_list)

    def transform_data(self, data_raw):
        
        path = data_raw.path
        
        #TODO: fix this
        crop = tf.cast(data_raw.crop, tf.float32) - _IMAGENET_MEAN
        crop.set_shape((self.input_size, self.input_size, 3))
        
        #TODO: if self.input_type is image and mode set to infer_segment_fit -> where do we pass image through?
        intermediate_rep = data_raw.intermediate_rep
        if self.mode not in ['infer_segment_fit']:
            # INTERMEDIATE REPRESENTATION
            if self.input_type in ['image', 'partmap']:
                if self.input_type == 'image':
                    intermediate_rep = data_raw.crop

                intermediate_rep = tf.cond(tf.equal(tf.shape(intermediate_rep)[0], self.interm_size), 
                                  true_fn=lambda: intermediate_rep,
                                  false_fn= lambda: tf.image.resize_images(intermediate_rep, [self.interm_size, self.interm_size], 
                                                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
            elif self.input_type in ['jointmap']:
                intermediate_rep = tf.py_func(self.get_jointmap, [data_raw.joints], tf.float64)
            
            intermediate_rep = tf.cast(intermediate_rep, tf.float32)
            intermediate_rep = intermediate_rep * 2. / 255. - 1.
            intermediate_rep.set_shape((self.interm_size, self.interm_size, 3))
        
        smplparams = tf.zeros((self.smplparam_len,))
        smplparams_full = tf.zeros((self.smplparam_len_full,))
        smplparams_orig = tf.zeros((85,))
        joint_locations_betas = tf.zeros((24,3))
        joint_locations_posed = tf.zeros((self.num_joints,3))
        joint_locations_projected = tf.zeros((self.num_joints,2))
        latent_flag = tf.zeros((1,))
        
        if 'infer' not in self.mode:
            # SMPLPARAMS (TO PREDICT)
            smplparams = data_raw.smplparams
            smplparams = tf.reshape(smplparams, tf.convert_to_tensor((85,), dtype=tf.int32))
            smplparams_orig = smplparams
            # split the parameters
            smplparams_shape = smplparams[0:10]
            smplparams_pose  = smplparams[10:82]
            smplparams_trans_xy = smplparams[82:84]
            smplparams_trans_z = smplparams[84:]
            # convert pose from aar to rotation matrices
            aar_convert = lambda i: conversions.aar_to_rotmat(i, self.kintree) 
            smplparams_pose = tf.py_func(aar_convert, [smplparams_pose], tf.float64)
            # join the parameters
            smplparams = tf.zeros([0,], dtype=tf.float64)
            if 'shape' in self.smplparam_components:
                smplparams = tf.concat([smplparams, smplparams_shape], 0)
            if 'pose'  in self.smplparam_components:
                smplparams = tf.concat([smplparams, smplparams_pose ], 0)
            if 'trans' in self.smplparam_components:
                smplparams = tf.concat([smplparams, smplparams_trans_xy], 0)
            smplparams = tf.reshape(smplparams, tf.convert_to_tensor((self.smplparam_len,), dtype=tf.int32))
            smplparams = tf.cast(smplparams, tf.float32)
            # mean/variance compensation
            if self.latent_mean is not None:
                smplparams = tf.subtract(smplparams, self.latent_mean)
                if False and self.latent_std is not None:
                    smplparams = tf.div(smplparams, self.latent_std)

            # SMPLPARAMS (FULL)
            smplparams_full = tf.cast(tf.concat([smplparams_shape, smplparams_pose, smplparams_trans_xy, smplparams_trans_z], 0), tf.float32)
            smplparams_full = tf.reshape(smplparams_full, tf.convert_to_tensor((self.smplparam_len_full,), dtype=tf.int32))
            smplparams_full = tf.cast(smplparams_full, tf.float32)
            
            # JOINTS
            joints = data_raw.joints
            joint_locations_betas = tf.reshape(joints[:72], (24, 3))
            joint_locations_betas = tf.cast(joint_locations_betas, tf.float32)
            joint_locations_posed = tf.reshape(joints[72:72+self.num_joints*3], (self.num_joints, 3))
            joint_locations_posed = tf.cast(joint_locations_posed, tf.float32)
            joint_locations_projected = tf.reshape(joints[72+self.num_joints*3:], (self.num_joints, 2))
            joint_locations_projected = tf.cast(joint_locations_projected, tf.float32)
            
            # LATENT FLAG
            latent_flag = data_raw.latent_flag
            latent_flag = tf.string_to_number(latent_flag, out_type=tf.int32)
            latent_flag = tf.reshape(latent_flag, (1,))
            latent_flag = tf.cast(latent_flag, tf.uint8)

        return DataPreprocessed(
            path=path,
            crop=crop,
            intermediate_rep=intermediate_rep,
            smplparams=smplparams,
            smplparams_full=smplparams_full,
            smplparams_orig=smplparams_orig,
            joint_locations_betas=joint_locations_betas,
            joint_locations_posed=joint_locations_posed,
            joint_locations_projected=joint_locations_projected,
            latent_flag=latent_flag
        )

