config = {
    # Model. #######################
    # Supported are: 'vgg', 'resnet50v1', 'resnet50v2'
    "encoder_type": 'resnet50v1',
    "seg_model": 'models/segmentation/refinenet_up', 
    # What modes are available for this model.
    "supp_modes": ['train', 'val', 'trainval', 'test', 'eval_train', 'infer_fit', 'infer_segment_fit'],

    # Data and preprocessing. ########################
    "dataset": "up_L12_refinenet", # see dataset/specs/
    # Type of input data: 'image', 'partmap', 'jointmap'.
    "input_type": 'partmap',
    "use_human_annotated_joints": False, 
    # Number of latent space dimensions (z = shape + pose + trans).
    "nz_full": 229,
    "latent_components": ['shape', 'pose'], #, 'trans'],
    # Output data properties:
    "num_landmarks": 24,
    # Scale the images if necessary.
    "input_size": 512,
    "interm_size": 224,
    # Misc. settings:
    "smplparams_mean": "helper_data/stats/train_mean.npy",
    "smplparams_std": "helper_data/stats/train_std.npy",
    "focal_length": 5000.,
    "use_gt_trans": True,
    "use_absrot": True,
    "use_svd": True,
    
    # Optimizer ####################
    # Supported loss terms: model_parameters, joint_locations_betas, joint_locations_posed, joint_locations_projected
    # Supported loss types: msqerr, mabserr, euclidist, sqeuclidist, gmc_std, gmc_var, euclidist_robust_v1
    "loss_terms": {"model_parameters": ("mabserr", 1.0), "joint_locations_projected": ("euclidist", 1.0)},
    "max_epochs": 75,
    "max_steps": None,
    "batch_size": 5,
    "lr": 0.00004,  # Adam lr.
    "lr_policy": "poly",
    "optimizer": "adam",
    "beta1": 0.5,  # Adam beta1 param.
    "weight_decay": 0.0001,
    # Ignore batchnorm training stats in test mode
    "ignore_batchnorm_training_stats": True,
    
    # Infrastructure. ##############
    # Save summaries after every x seconds.
    "summary_freq": 5,
    # Create traces after every x batches.
    "trace_freq": 0,
    # After every x epochs, save model.
    "save_freq": 5,
    # Keep x saves.
    "kept_saves": 150,
    # After every x epochs, render images.
    "display_freq": 10,
    # Random seed to use.
    "seed": 538728914,
}
