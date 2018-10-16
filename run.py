#!/usr/bin/env python2
"""Main control for the experiments."""
import ast
import glob
import imp
import logging
import math
import os
import pdb
import random
import signal
import socket
import time

import click
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import tqdm


from clustertools.log import LOGFORMAT

LOGGER = logging.getLogger(__name__)


def create_restoration_saver(ckpt_path, cur_graph, name='restore', silent=True):
    # load graph from meta file and get ckpt variables
    load_graph = tf.Graph()
    with load_graph.as_default():
        meta_file = ckpt_path + '.meta'
        rest_saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
        ckpt_vars = [(v.name, tuple(v.shape.as_list()))
                     for v in tf.global_variables()]
    # get list of variables current graph
    with cur_graph.as_default():
        graph_vars = [(v.name, tuple(v.shape.as_list()))
                      for v in tf.global_variables()]
        # list of variables to restore (i.e. intersection of
        # ckpt_vars, graph_vars)
        rest_vars = list(set(graph_vars).intersection(set(ckpt_vars)))
        rest_var_names = [v[0] for v in rest_vars]
        # stop program if the specified checkpoint has no variables of interest
        if len(rest_vars) == 0:
            raise ValueError(
                'Specified checkpoint has no variables in common with the current model.'
            )
        # determine which variables from checkpoint will be ignored
        ignored_var_names = [v[0] for v in list(set(rest_vars).symmetric_difference(set(ckpt_vars)))]
        if not silent:
            for vn in ignored_var_names:
                LOGGER.warn(
                    "Variable `%s` found in specified checkpoint will be ignored!",
                    vn)
            # determine which variables won't be restored from checkpoint
            nonrest_var_names = [v[0] for v in list(set(rest_vars).symmetric_difference(set(graph_vars)))]
            for vn in nonrest_var_names:
                LOGGER.warn("Variable `%s` not found in specified checkpoint!", vn)
        rest_saver = tf.train.Saver(
            [v for v in tf.global_variables() if v.name in rest_var_names],
            name=name)

    return rest_saver


@click.command()
@click.argument(
    "mode",
    type=click.Choice(
        ["train", "val", "trainval", "test", "eval_train", "infer_fit", "infer_segment_fit"]))
@click.argument(
    "exp_name", type=click.Path(exists=True, writable=True, file_okay=False))
@click.option(
    "--num_threads",
    type=click.INT,
    default=8,
    help="Number of data preprocessing threads.")
@click.option(
    "--no_checkpoint",
    type=click.BOOL,
    is_flag=True,
    help="Ignore checkpoints.")
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Checkpoint to use for restoring (+.meta).")
@click.option(
    "--override_dset_name",
    type=click.STRING,
    default=None,
    help="If specified, override the configure dset_name.")
@click.option(
    "--inp_fp",
    type=click.Path(exists=True, writable=False),
    default=None,
    help="Required for infer mode: Location of files to process.")
@click.option(
    "--out_fp",
    type=click.Path(writable=True),
    default=None,
    help="If specified, write test or sample results there.")
@click.option(
    "--custom_options",
    type=click.STRING,
    default="",
    help="Provide model specific custom options.")
@click.option(
    "--visualise",
    type=click.Choice(
        ["render", "pose"]),
    help="visualise the output, either by rendering the full mesh or by displaying the predicted joints")
@click.option(
    "--no_output",
    type=click.BOOL,
    default=False,
    is_flag=True,
    help="Don't store results in test modes.")
@click.option(
    "--ignore_batchnorm_stats",
    type=click.BOOL,
    is_flag=True,
    help="Ignore batchnorm statistics at test time.")
def cli(**args):
    """Main control for the experiments."""
    LOGGER.info("Running on host: %s", socket.getfqdn())
    #### SETUP OUTPUT FOLDERS ####
    exp_name = args['exp_name'].strip("/")
    assert exp_name.startswith(os.path.join("experiments", "config"))
    exp_purename = os.path.basename(exp_name)
    exp_feat_fp = os.path.join("experiments", "features", exp_purename)
    exp_log_fp = os.path.join("experiments", "states", exp_purename)
    if not os.path.exists(exp_feat_fp):
        os.makedirs(exp_feat_fp)
    if not os.path.exists(exp_log_fp):
        os.makedirs(exp_log_fp)

    if args['mode'] in ['infer_fit', 'infer_segment_fit']:
        assert 'inp_fp' in args.keys(
        ), "'--inp_fp' option required for 'infer_(segment_)fit' modes"
        assert 'out_fp' in args.keys(
        ), "'--out_fp' option required for 'infer_(segment_)fit' modes"
        assert os.path.exists(
            args['inp_fp']), "Specified input dir: '%s' doesn't exist" % (
                args['inp_fp'])
        output_fp = args['out_fp']
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)

    #### SETUP CONFIGURATION ####
    # Configuration.
    exp_config_mod = imp.load_source('_exp_config',
                                     os.path.join(exp_name, 'config.py'))
    exp_config = exp_config_mod.get_config()
    # check that mode is valid
    mode = args['mode']
    assert mode in exp_config["supp_modes"], (
        "Unsupported mode by this model: %s, available: %s." %
        (mode, str(exp_config["supp_modes"])))
    LOGGER.info("Running mode `%s` for experiment `%s`.", mode, exp_name)
    # make adjustments to config based on command line parameters
    exp_config = exp_config_mod.adjust_config(exp_config_mod.get_config(),
                                              mode)
    if args["override_dset_name"] is not None:
        LOGGER.warn("Overriding dset suffix to `%s`!",
                    args["override_dset_name"])
        exp_config["dataset"] = args["override_dset_name"]
    if args['custom_options'] != '':
        custom_options = ast.literal_eval(args["custom_options"])
        exp_config.update(custom_options)
    exp_config['num_threads'] = args["num_threads"]
    exp_config['ignore_batchnorm_training_stats'] = (
        args['ignore_batchnorm_stats'] is not None)
    exp_config['inp_fp'] = args['inp_fp']
    exp_config['out_fp'] = args['out_fp']
    # print all options
    LOGGER.info("Configuration:")
    for key, val in exp_config.items():
        LOGGER.info("%s = %s", key, val)
    # set random seed
    random.seed(exp_config["seed"])
    tf.set_random_seed(exp_config["seed"])

    #### SETUP INPUT PIPELINE ####
    # load data mean/std
    smplparams_mean = np.load(exp_config['smplparams_mean'])
    smplparams_std = np.load(exp_config['smplparams_std'])
    if 'latent_components' in exp_config.keys():
        assert len(exp_config['latent_components']) < 4
        param_selection = []
        exp_config['nz'] = 0
        if 'shape' in exp_config['latent_components']:
            exp_config['nz'] = exp_config['nz'] + 10
            param_selection += range(10)
        if 'pose' in exp_config['latent_components']:
            exp_config['nz'] = exp_config['nz'] + 216
            param_selection += range(10, 226)
        if 'trans' in exp_config['latent_components']:
            exp_config['nz'] = exp_config['nz'] + 2
            param_selection += range(226, 228)
    else:
        param_selection = range(228)
    latent_mean = smplparams_mean[param_selection]
    latent_std = smplparams_std[param_selection]

    LOGGER.info("Setting up preprocessing...")
    exp_preproc_mod = imp.load_source(
        '_exp_preprocessor', os.path.join(exp_name, 'preprocessor.py'))
    preprocessor = exp_preproc_mod.Preprocessor(exp_config, mode,
                                                latent_mean, latent_std)
    examples = preprocessor.get_batching_op()
    nsamples = preprocessor.get_num_samples()
    steps_per_epoch = int(math.ceil(1.0 * nsamples / exp_config['batch_size']))
    LOGGER.info("%d examples prepared, %d steps per epoch.", nsamples,
                steps_per_epoch)

    #### SETUP MODEL AND LOSS OPS ####
    # Checkpointing.
    # Build model.
    #TODO should be handled in preprocessor
    if mode in ['infer_segment_fit']:
        model_input = examples.crop
    else:
        model_input = examples.intermediate_rep
    
    model_mod = imp.load_source('_model', os.path.join(exp_name, 'model.py'))
    model = model_mod.Model(
        exp_config,
        tf.get_default_graph(),
        model_input,
        examples.smplparams_full[:, -3:],
        tf.constant(latent_mean, dtype=np.float32),
        tf.constant(latent_std, dtype=np.float32),
        is_training=(mode in ['train', 'trainval']))

    if mode in ['train', 'trainval', 'val', 'test', 'infer_fit', 'infer_segment_fit']:
        opt_mod = imp.load_source('_optimiser',
                                  os.path.join(exp_name, 'optimiser.py'))
        optimiser = opt_mod.Optimiser(
            exp_config, model, examples,
            tf.constant(latent_mean, dtype=np.float32),
            tf.constant(latent_std, dtype=np.float32))
        optimiser.prepare_loss_ops(exp_config)
        loss_full = optimiser.get_loss_op()
        losses = optimiser.get_loss_dict()
    global_step = tf.Variable(
        name="global_step",
        expected_shape=(),
        dtype=tf.int64,
        trainable=False,
        initial_value=0)

    # setup restoration savers
    rest_saver = None
    if args['no_checkpoint']:
        assert args['checkpoint'] is None
    if not args["no_checkpoint"]:
        LOGGER.info("Looking for checkpoints...")
        if args['checkpoint'] is not None:
            checkpoint = os.path.splitext(args['checkpoint'])[0]
        else:
            checkpoint = tf.train.latest_checkpoint(exp_log_fp)
        if checkpoint is None:
            LOGGER.info("No checkpoint found. Continuing without.")
        else:
            rest_saver = create_restoration_saver(checkpoint,
                                                  tf.get_default_graph())
    if mode in ['infer_segment_fit']:
        seg_rest_saver = create_restoration_saver(exp_config['seg_model'],
                                                  tf.get_default_graph(),
                                                  name='seg_restore')
                                 
    if mode not in ['train', 'trainval'] and rest_saver is None:
        raise Exception("The mode %s requires a checkpoint!" % (mode))

    #### SETUP OUTPUT ####
    # initialise output module
    out_mod = imp.load_source("_write_output",
                              os.path.join(exp_name, 'write_output.py'))

    # setup snapshotting saver
    if mode in ['train', 'trainval']:
        saver = tf.train.Saver(max_to_keep=exp_config["kept_saves"])
    else:
        saver = None

    # prepare writer for logger output
    fh = logging.FileHandler(os.path.join(exp_log_fp, 'run.py.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(LOGFORMAT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)
    LOGGER.info("Running on host: %s", socket.getfqdn())

    #### SETUP SESSION ####
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = False
    prepared_session = tf.Session(config=sess_config)

    epoch = 0
    with prepared_session as sess:
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, thread_name_filter="MainThread$")
        # Compute stats
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([
                tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()
            ])
        LOGGER.info("Parameter count: %d.", sess.run(parameter_count))

        if mode in ['train', 'trainval']:
            if exp_config["max_epochs"] is not None:
                max_steps = steps_per_epoch * exp_config["max_epochs"]
                total_examples_presented = nsamples * exp_config["max_epochs"]
            elif exp_config["max_steps"] is not None:
                max_steps = exp_config["max_steps"]
                total_examples_presented = (
                    max_steps // steps_per_epoch) * nsamples + (
                        max_steps % steps_per_epoch) * exp_config["batch_size"]
            else:
                raise ValueError(
                    "You need to specify either a maximum nr. of epochs or steps."
                )
            # TODO: move this into the optimiser
            if exp_config["lr_policy"] == "step":
                nr_steps = int(
                    math.ceil(1. * exp_config["max_epochs"] /
                              exp_config["lr_stepsize"]))
                exp_config["lr_boundaries"] = [
                    np.int64(
                        steps_per_epoch * (s + 1) * exp_config["lr_stepsize"])
                    for s in range(nr_steps - 1)
                ]
                exp_config["lr_steps"] = [
                    exp_config["lr"] * exp_config["lr_mult"]**s
                    for s in range(nr_steps)
                ]

            # setup optimiser
            # TODO: this might have problems
            train_op = optimiser.prepare_train_op(exp_config, global_step,
                                                  max_steps)

        # Prepare summaries
        summary_mod = imp.load_source('_summaries',
                                      os.path.join(exp_name, 'summaries.py'))
        # TODO: modify create_summaries to accept optimiser output
        display_fetches, test_fetches = summary_mod.create_summaries(
            mode, exp_config, examples, model.get_outputs(),
            optimiser.get_losses(), optimiser.get_scalars_to_track(),
            optimiser.get_otherdata_to_track(), optimiser.get_learning_rate())

        sw = tf.summary.FileWriter(os.path.join(exp_log_fp, mode))
        summary_op = tf.summary.merge_all()

        # Initialise variables
        LOGGER.info("Initializing variables...")
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        # Restore variables from checkpoint
        if mode in ['infer_segment_fit']:
            seg_rest_saver.restore(sess, exp_config['seg_model'])
        if rest_saver is not None:
            rest_saver.restore(sess, checkpoint)
            
        # Get initial step
        fetches = {}
        fetches["global_step"] = global_step
        initial_step = sess.run(fetches)["global_step"]  # [0]
        LOGGER.info("On global step: %d.", initial_step)

        if len(glob.glob(os.path.join(exp_log_fp, mode, 'events.*'))) == 0:
            LOGGER.info("Summarizing graph...")
            sw.add_graph(sess.graph, global_step=initial_step)
        if mode in ['val', 'test', 'eval_train']:
            image_dir = os.path.join(exp_feat_fp, exp_config["dataset"],
                                     'step_' + str(initial_step), mode)
        elif mode in ['infer_fit', 'infer_segment_fit']:
            image_dir = output_fp
        else:
            image_dir = exp_log_fp
        if args["out_fp"] is not None:
            image_dir = args["out_fp"]
        if not args["no_output"]:
            LOGGER.info("Writing image status to `%s`.", image_dir)
        else:
            image_dir = None
        if mode in ['val', 'test', 'eval_train', 'infer_fit', 'infer_segment_fit']:
            shutdown_requested = [False]

            def SIGINT_handler(signal, frame):  # noqa: E306
                LOGGER.warn("Received SIGINT.")
                shutdown_requested[0] = True

            signal.signal(signal.SIGINT, SIGINT_handler)

            av_results = dict((name, []) for name in test_fetches.keys())
            av_placeholders = dict((name, tf.placeholder(tf.float32))
                                   for name in test_fetches.keys())
            for name in test_fetches.keys():
                tf.summary.scalar(
                    name, av_placeholders[name], collections=['evaluation'])
            test_summary = tf.summary.merge_all('evaluation')
            display_fetches.update(test_fetches)
            b_id = 0
            preprocessor.initialise_iterator(sess, shuffle=False)
            pbar = tqdm.tqdm(total=nsamples)
            while True:
                try:
                    display_fetches['paths'] = examples.path
                    results = sess.run(display_fetches)
                    if not args['no_output']:
                        if mode == 'eval_train':
                            index_fp = out_mod.save_images(
                                results,
                                image_dir,
                                'train',
                                exp_config,
                                latent_mean,
                                batch=b_id,
                                visualise=args['visualise'])
                        else:
                            index_fp = out_mod.save_images(
                                results,
                                image_dir,
                                mode,
                                exp_config,
                                latent_mean,
                                batch=b_id,
                                visualise=args['visualise'])
                    # Check for problems with this result.
                    results_valid = True
                    for key in test_fetches.keys():
                        if not np.isfinite(results[key]):
                            if 'paths' in results.keys():
                                LOGGER.warn(
                                    "There's a problem with results for "
                                    "%s! Skipping.", results['paths'][0])
                            else:
                                LOGGER.warn("Erroneous result for example %d!",
                                            b_id)
                            results_valid = False
                            break
                    if results_valid:
                        for key in test_fetches.keys():
                            av_results[key].append(results[key])
                    pbar.update(len(results['paths']))
                    b_id += 1
                    if shutdown_requested[0]:
                        break
                except tf.errors.OutOfRangeError:
                    LOGGER.info("Finished processing the validation/test set")
                    pbar.close()
                    break
            LOGGER.info("Results:")
            feed_results = dict()
            for key in sorted(test_fetches.keys()):
                # av_results[key + '_full'] = av_results[key]
                av_results[key] = np.mean(av_results[key])
                feed_results[av_placeholders[key]] = av_results[key]
                LOGGER.info("  %s: %s", key, av_results[key])
            if shutdown_requested[0]:
                LOGGER.warn("Not writing results to tf summary due to "
                            "incomplete evaluation.")
            elif mode not in ['infer_fit', 'infer_segment_fit']:
                sw.add_summary(
                    sess.run(test_summary, feed_dict=feed_results),
                    initial_step)
            if not args['no_output']:
                LOGGER.info("Wrote index at `%s`.", index_fp)
        elif mode in ['train', 'trainval']:
            # Training.
            last_summary_written = time.time()
            shutdown_requested = [False]  # Needs to be mutable to access.

            # Register signal handler to save on Ctrl-C.
            def SIGINT_handler(signal, frame):  # noqa: E306
                LOGGER.warn("Received SIGINT. Saving model...")
                saver.save(
                    sess,
                    os.path.join(exp_log_fp, "model"),
                    global_step=global_step)
                shutdown_requested[0] = True

            signal.signal(signal.SIGINT, SIGINT_handler)
            # TODO: compute this properly
            pbar = tqdm.tqdm(
                total=(max_steps - initial_step) * exp_config["batch_size"])
            paths = np.array([], dtype=object)
            step = initial_step
            preprocessor.initialise_iterator(
                sess, shuffle=True
            )  # (step > 0)) # TODO: fast-forward depending of the step
            while step < max_steps:
                try:
                    if (False and (step == 0 or step == 10)):
                        # Save directly at first iteration to make sure this is
                        # working.
                        LOGGER.info("Saving model...")
                        saver.save(
                            sess,
                            os.path.join(exp_log_fp, "model"),
                            global_step=global_step)

                    def should(freq, epochs=False):
                        if epochs:
                            return freq > 0 and (
                                (epoch + 1) % freq == 0 and
                                (step + 1) % steps_per_epoch == 0
                                or step == max_steps - 1)
                        else:
                            return freq > 0 and ((step + 1) % freq == 0
                                                 or step == max_steps - 1)

                    options = None
                    run_metadata = None
                    if should(exp_config["trace_freq"]):
                        options = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                    # Setup fetches.
                    fetches = {
                        "train": train_op,
                        "global_step": global_step,
                        "paths": examples.path,
                        # "dbg_outputs": model.dbg_outputs,
                    }
                    if ((time.time() - last_summary_written) >
                            exp_config["summary_freq"]):
                        fetches["summary"] = summary_op
                    if (should(exp_config["display_freq"], epochs=True)
                            or should(exp_config["save_freq"], epochs=True)
                            or step == max_steps - 1):
                        fetches["display"] = display_fetches
                    # Run!
                    results = sess.run(
                        fetches, options=options, run_metadata=run_metadata)
                    # Write.
                    batch_size = len(results['paths'])
                    if (should(exp_config["save_freq"], epochs=True)
                            or results["global_step"] == 1
                            or step == max_steps - 1):
                        # Save directly at first iteration to make sure this is
                        # working.
                        LOGGER.info("Saving model...")
                        saver.save(
                            sess,
                            os.path.join(exp_log_fp, "model"),
                            global_step=global_step)
                    if "summary" in results.keys():
                        sw.add_summary(results["summary"],
                                       results["global_step"])
                        last_summary_written = time.time()
                    if "display" in results.keys():
                        LOGGER.info("saving display images")
                        out_mod.save_images(
                            results["display"],
                            image_dir,
                            mode,
                            exp_config,
                            latent_mean,
                            step=results["global_step"])  # [0])
                    if should(exp_config["trace_freq"]):
                        LOGGER.info("recording trace")
                        sw.add_run_metadata(run_metadata,
                                            "step_%d" % results["global_step"])
                        trace = timeline.Timeline(
                            step_stats=run_metadata.step_stats)
                        with open(os.path.join(exp_log_fp, "timeline.json"),
                                  "w") as trace_file:
                            trace_file.write(
                                trace.generate_chrome_trace_format())
                        # Enter 'chrome://tracing' in chrome to open the file.
                    epoch = results["global_step"] // steps_per_epoch
                    pbar.update(batch_size)
                    step += 1
                    if shutdown_requested[0]:
                        break
                except tf.errors.OutOfRangeError:
                    LOGGER.info("Epoch completed...")
                    preprocessor.initialise_iterator(sess, shuffle=True)
                    # preprocessor.enable_data_augmentation()
                    continue
            pbar.close()
        LOGGER.info("Shutting down...")
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    logging.getLogger("clustertools.db.tools").setLevel(logging.WARN)
    logging.getLogger("PIL.Image").setLevel(logging.WARN)
    cli()
