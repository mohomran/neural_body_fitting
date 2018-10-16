"""Summarize the networks actions."""
import logging
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import matplotlib

LOGGER = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([122.67892, 116.66877, 104.00699])

#TODO: deprocessing code should be in preprocessor
def deprocess(config, image, input_type, cmap):
    def cfunc(x): return postprocess_colormap(x, cmap)
    if input_type in ['labels', 'probabilities']:    
        return tf.py_func(cfunc, [tf.argmax(image, 3)], tf.uint8)
    elif input_type in ['jointmap']:
        return tf.py_func(cfunc, [tf.cast(255 * tf.reduce_max(image, axis=3), tf.uint8)], tf.uint8)
    elif input_type in ['partmap']:
        return tf.image.convert_image_dtype((image + 1) / 2, dtype=tf.uint8, saturate=True)
    elif input_type in ['image']:
        image =  tf.image.resize_images((image + _IMAGENET_MEAN)/255.0, [config['interm_size'], config['interm_size']],
                                        method=tf.image.ResizeMethod.BILINEAR,
                                        align_corners=True)
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    else:
        raise NotImplementedError

def create_summaries(mode, config, examples, outputs, losses, track_scalars, track_other, learning_rate, debug=False):
    LOGGER.info("Setting up summaries and fetches...")
    
    cmaplist = [tuple(c/255.0) for c in np.squeeze(sm.imread(config['colour_map']))[:256]]
    cmap = matplotlib.colors.ListedColormap(cmaplist, name='u2p')
    # Inputs.
    with tf.name_scope("deprocess_inputs"):
        deprocessed_input =  deprocess(config,
                                       examples.crop,
                                       'image',
                                       cmap)
        deprocessed_interm = deprocess(config,
                                       outputs["intermediate_rep"],
                                       config["input_type"],
                                       cmap)

    # Collect inputs/outputs ##################################################
    display_fetches = dict()
    with tf.name_scope("encode_images"):
        results = [('input', deprocessed_input),
                   ('intermediate_rep', deprocessed_interm)]
        for name, res in results:
            if res is not None:
                display_fetches[name] = res#tf.map_fn(tf.image.encode_png,
                                           #       res,
                                           #       dtype=tf.string,
                                           #       name=name+'_pngs') 

    #display_fetches['input'] = tf.cast(examples.crop + _IMAGENET_MEAN, tf.uint8)
    display_fetches['paths'] = examples.path
    display_fetches['latent'] = outputs['latent']
    display_fetches['joints3d_pred'] = outputs['joints3D']
    display_fetches['joints2d_pred'] = outputs['joints2D']
    #TODO: add segmentation if mode = 'infer_segment_fit'

    #TODO: do we need this stuff?
    if mode not in ['infer_fit', 'infer_segment_fit']:
        display_fetches['latent_target'] = examples.smplparams_full
        display_fetches['latent_sqerr'] = track_other["latent_sqerr"]
        display_fetches['joints3d_gt'] = examples.joint_locations_posed
        display_fetches['joints2d_gt'] = examples.joint_locations_projected

    # Create the summaries. ###################################################
    if mode not in ['infer_fit', 'infer_segment_fit']:
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", deprocessed_input)

        tf.summary.histogram("latent", outputs['latent'])

    test_fetches = {}
    if mode in ['train', 'trainval']:
        tf.summary.scalar("learning_rate", learning_rate)
        for k in sorted(losses.keys()):
            tf.summary.scalar("loss/" + k, losses[k])
        for k in sorted(track_scalars.keys()):
            tf.summary.scalar("track/" + k, track_scalars[k])
    elif mode in ['val', 'test', 'eval_train']:
        # These fetches will be evaluated and averaged at test time.s
        for k in sorted(losses.keys()):
            test_fetches["loss/" + k] = losses[k]
        for k in sorted(track_scalars.keys()):
            test_fetches["track/" + k] = track_scalars[k]

    LOGGER.info("Summaries and fetches complete.")
    return display_fetches, test_fetches
