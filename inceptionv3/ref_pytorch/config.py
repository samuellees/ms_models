
from easydict import EasyDict as edict

inceptionv3_cfg = edict({
    'num_classes': 10,
    'lr_init': 0.0045,
    'lr_decay_rate': 0.94,
    'lr_decay_epoch': 2,
    'rmsprop_decay': 0.9,
    'rmsprop_momentum': 0.9,
    'rmsprop_epsilon': 1.0,
    'label_smoothing_eps': 0.1,
    'epoch_size': 200,
    # 'epoch_size': 1,
    'keep_checkpoint_max': 10,
    'save_checkpoint_steps': 1562,

    'batch_size': 32,
    'image_height': 299,
    'image_width': 299,
})
