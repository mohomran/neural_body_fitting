"""Summaries and outputs."""
import os
import sys
import os.path as path
import numpy as np
from collections import OrderedDict
import logging
import cPickle as pickle
import scipy.misc as sm
import skimage.draw as skdraw
from PIL import Image

from utils import conversions

from up_tools.render_segmented_views import render_body_impl
logging.basicConfig()
logger = logging.getLogger('opendr.lighting')
logger.propagate = False

_PATH_TO_MESH = 'models/smpl/template-bodyparts-corrected-labeled-split12.ply'

LOGGER = logging.getLogger(__name__)

def append_index(row_infos, image_dir, mode):
    """Append or create the presentation html file for the images."""
    index_path = path.join(path.dirname(image_dir), "index.html")
    if path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html>\n<body>\n<table>\n<tr>")
        colnames = [key for key, val in row_infos[0].items()
                    if val[1] in ['image', 'text']]
        for coln in colnames:
            index.write("<th>%s</th>\n" % (coln))
        index.write("</tr>\n")
    for row_info in row_infos:
        index.write("<tr>\n")
        for coln, (colc, colt) in row_info.items():
            if colt == 'text':
                index.write("<td>%s</td>" % (colc))
            elif colt == 'image':
                filename = path.join(image_dir,
                                     row_info['name'][0] + '_' + coln + '.png')
                if isinstance(colc, str):
                    with open(filename, 'w') as outf:
                        outf.write(colc)
                else:
                    assert colc.dtype == 'uint8'
                    sm.imsave(filename, colc)
                index.write("<td><img src='images/%s'></td>" % (
                    path.basename(filename)))
            elif colt == 'plain':
                filename = path.join(image_dir,
                                     row_info['name'][0] + '_' + coln + '.npy')
                np.save(filename, colc)
                if coln == 'body':
                    with open(os.path.join(image_dir, 
                                           row_info['name'][0] + '_body.pkl'), 'wb') as f:
                        pickle.dump(colc, f)
            else:
                raise Exception("Unsupported mode: %s." % (mode))
        index.write("</tr>\n")
    return index_path

def get_body_dict(preds, target, latent_mean, config=None, kintree=None):
    #TODO: this should happen elsewhere ('latent_components', 'target' and even 'latent_mean' shouldn't be required!)
    preds = preds + latent_mean
    preds_full = np.zeros((config['nz_full'],))
    if 'latent_components' in config.keys():
        if target is not None:
            preds_full = target
        if 'shape' in config['latent_components']:
            preds_full[:10] = preds[:10]
        if 'pose' in config['latent_components']:
            if 'shape' in config['latent_components']:
                preds_full[10:226] = preds[10:226]
            else:
                preds_full[10:226] = preds
        if 'trans' in config['latent_components']:
            if 'shape' in config['latent_components']:
                preds_full[226:228] = preds[226:]
            else:
                preds_full[226:228] = preds[216:]
        else:
            preds_full[226:] = [-0.01, 0.115, 20.3]
    else:
        preds_full = preds

    pose_vector = preds_full[10:226]

    # undo mean/var compensation
    out_pkl = {}
    out_pkl['rt'] = np.array([ 0.,  0.,  0.])
    out_pkl['t'] = np.array([ 0.,  0.,  0.])
    out_pkl['f'] = 5000.0

    out_pkl['pose'] = conversions.rotmat_to_aar(pose_vector, kintree)
    out_pkl['betas'] = preds_full[:10]
    out_pkl['trans'] = preds_full[226:]

    return out_pkl

def save_images(fetches, image_dir, mode, config, latent_mean, step=None, batch=0, visualise=None):

    part_colours = np.squeeze(sm.imread(config['colour_map']))

    image_dir = path.join(image_dir, 'images')
    if not path.exists(image_dir):
        os.makedirs(image_dir)
        
    row_infos = []
    batch_size = len(fetches["paths"])
    kintree = None
    if config['use_absrot']:
        kintree = conversions.prepare_kintree()

    for im_idx in range(batch_size):
        if step is not None:
            row_info = OrderedDict([('step', (str(step), 'text')), ])
        else:
            row_info = OrderedDict()
        in_path = fetches["paths"][im_idx]
        name, _ = os.path.splitext(os.path.basename(in_path))
        if step is not None:
            name = str(step) + '_' + name
        row_info["name"] = (name, 'text')
        if 'inputs' in fetches.keys():
            row_info["inputs"] = (fetches['inputs'][im_idx], 'image')

        #TODO: not loosely coupled enough from summaries.py
        row_info["latent"] = (fetches["latent"][im_idx], 'plain')
        row_info["joints2d_pred"] = (fetches["joints2d_pred"][im_idx], 'plain')
        row_info["joints3d_pred"] = (fetches["joints3d_pred"][im_idx], 'plain')
        row_info["input"] = (fetches["input"][im_idx], 'image')
        
        #TODO: overlays + rendering should be moved into summaries function        
        segmentation = fetches["intermediate_rep"][im_idx]
        blend_indices = np.where(np.all(segmentation==0, axis=2))
        segmentation[blend_indices] = fetches["input"][im_idx][blend_indices]
        row_info["intermediate"] = (segmentation, 'image')
       
        if mode not in ['infer_fit', 'infer_segment_fit']:
            row_info["body"] = (get_body_dict(fetches["latent"][im_idx], fetches["latent_target"][im_idx], latent_mean, config, kintree), 'plain')
            row_info["latent_sqerr"] = (fetches["latent_sqerr"][im_idx], 'plain')
            row_info["joints2d_gt"] = (fetches["joints2d_gt"][im_idx], 'plain')
            row_info["joints3d_gt"] = (fetches["joints3d_gt"][im_idx], 'plain')
        else:
            row_info["body"] = (get_body_dict(fetches["latent"][im_idx], None, latent_mean, config, kintree), 'plain')

        crop_size = 224
        if visualise == 'render':
            stored_parameters = row_info["body"][0]
            #TODO: move literals elsewhere
            stored_parameters['cam_c'] = np.array([crop_size, crop_size], dtype=int)/2.0
            stored_parameters['f'] = 2187.5
            rendering = render_body_impl(stored_parameters,
                                         resolution=np.array([crop_size, crop_size], dtype=int),
                                         quiet=True,
                                         use_light=True,
                                         path_to_mesh=_PATH_TO_MESH)[0]

            blend_indices = np.where(np.all(rendering==255, axis=2))
            rendering[blend_indices] = fetches["input"][im_idx][blend_indices]
            row_info["output"] = (rendering, 'image')
            
        elif visualise == 'pose':
            img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            joints = fetches["joints2d_pred"][im_idx]
            for i in range(joints.shape[0]):
                rr, cc = skdraw.circle(joints[i,0], joints[i,1], 3, shape=(crop_size, crop_size))
                for j in range(3):
                    img[cc, rr, j] = part_colours[i,j]
            row_info["output"] = (img, 'image')

        row_infos.append(row_info)
        LOGGER.debug("Processed image %d.",
                     batch * batch_size + im_idx + 1)
    index_fp = append_index(row_infos, image_dir, mode)
    return index_fp
