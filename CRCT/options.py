import os
import argparse
import json
import numpy as np
from time import gmtime, strftime
import sys


def read_command_line(argv=None):
    parser = argparse.ArgumentParser(description='Large Scale Pretraining for Visual Dialog')
    parser.add_argument('-command', type=str, default=" ".join(sys.argv), help="The command used for running this script.")
    # -------------------------------------------------------------------------
    parser.add_argument('-start_checkpoint', default='', help='path of starting model checkpt')
    parser.add_argument('-model_config', default='', help='model definition of the bert model')

    parser.add_argument('-num_workers', default=16, type=int, help='Number of worker threads in dataloader')
    parser.add_argument('-batch_size', default=80, type=int, help='size of mini batch')
    parser.add_argument('-num_epochs', default=20, type=int, help='total number of epochs')
    parser.add_argument('-batch_multiply', default=1, type=int, help='amplifies batch size in mini-batch training')

    parser.add_argument('-lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('-image_lr', default=2e-5, type=float, help='learning rate for vision params')
    parser.add_argument('-min_lr', default=1.3e-5, type=float, help='min learning rate')

    parser.add_argument('-continue', action='store_true', help='continue training')

    parser.add_argument('-max_seq_len', default=256, type=int, help='maximum sequence length for the dialog sequence')

    parser.add_argument('-nsp_loss_coeff', default=1, type=float, help='Coeff for nsp loss')
    parser.add_argument('-reg_loss_coeff', default=1, type=float, help='Coeff for regression loss')

    parser.add_argument('-L1', action='store_true', help='train with L1 loss')

    parser.add_argument('-mask_prob', default=0, type=float, help='prob used to sample masked tokens')
    parser.add_argument('-mask_prob_img', default=0, type=float, help='prob used to sample masked image features')
    parser.add_argument('-mask_img_loc', type=float, default=0)

    parser.add_argument('-save_path', default='', help='Path to save checkpoints')
    parser.add_argument('-save_name', default='', help='Name of save directory within savePath')

    parser.add_argument('-cuda_num', default=-1, type=int, help='cuda:i')
    parser.add_argument('-eval_batch_size', default=10, type=int, help='default 10')
    parser.add_argument('-ddp', action='store_true', help='continue training')
    parser.add_argument('-rank', type=int, default=0, help='cuda:i')
    parser.add_argument('-dist_url', default='')
    parser.add_argument('-world_size', type=int, default=1, help='cuda:i')
    parser.add_argument('-num_proc', type=int, default=1, help='number of process')
    parser.add_argument('-rank_from', type=int, default=0)
    parser.add_argument('-gpu_from', type=int, default=0)
    parser.add_argument('-seed', type=int, default=0)

    # -------------------------------------------------------------------------
    parser.add_argument('-figure_feat_path', default="")
    parser.add_argument('-qa_parent_dir', default="")
    parser.add_argument('-qa_file', required=True)
    parser.add_argument('-fixed_vocab', action="store_true")
    parser.add_argument('-no_eval', action="store_true", help="No evaluation during training")
    parser.add_argument('-details', type=str, default="None")
    parser.add_argument('-pretrain', action="store_true")
    parser.add_argument('-wd', default=0.01, type=float, help='weight decay')
    parser.add_argument('-tol_margin', default=0.01, type=float, help='% tolerance margin for regression')
    parser.add_argument('-warmup', default=3000, type=int, help='warmup steps')
    parser.add_argument('-log_file', type=str, default="None")
    parser.add_argument('-hist_name', type=str, default="")
    parser.add_argument('-dataset', type=str, default="plotqa")
    parser.add_argument('-categories', type=int)
    parser.add_argument('-CE_REG', action="store_true", help='treat regression as a classification problem')

    parser.add_argument('-BOT_MODE', action="store_true", help='PlotQA Bot mode')
    parser.add_argument('-hbar_bbox_t', type=lambda x: (str(x).lower() == 'true'), default=False, help='hbar->vbar by hbar transpose for reduction to vbar')
    parser.add_argument('-binary_answers', type=lambda x: (str(x).lower() == 'true'), default=False, help='Counter Examples')
    parser.add_argument('-eval_set', type=str, default='val')
    parser.add_argument('-eval_type', type=str, help="evaluation progress type", choices=['vocab_table', 'examples'],
                        default='vocab_table')

    parser.add_argument('-tensorboard', default="", help='Path to save tensorboard files')
    parser.add_argument('-checkpoints_dir', type=str, default='')
    parser.add_argument('-dataset_config', type=str, default='config/plotqa.json')
    # -------------------------------------------------------------------------
    try:
        parsed = vars(parser.parse_args(args=argv))

    except IOError as msg:
        parser.error(str(msg))

    with open(parsed['dataset_config'], "r") as file:
        dataset_config = json.load(file)

    # update all folders to be a full path
    for sub_path in ['figure_feat_path', 'model_config', 'save_path', 'tensorboard', 'checkpoints_dir', 'qa_parent_dir']:
        dataset_config[sub_path] = os.path.join(dataset_config['main_folder'], dataset_config[sub_path])

    for key in dataset_config:
        # if key not in parsed or key in parsed and not parsed[key]:
        parsed[key] = dataset_config[key]

    if parsed['save_name']:
        # Custom save file path
        parsed['save_path'] = os.path.join(parsed['save_path'],
                                           parsed['save_name'])
    else:
        # Standard save path with time stamp
        import random
        timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())
        parsed['save_path'] = os.path.join(parsed['save_path'], timeStamp)
        parsed['save_path'] += '_{:0>6d}'.format(random.randint(0, 10e6))

    parsed['dataset_config'] = dataset_config

    if parsed['start_checkpoint'] and not os.path.isfile(parsed['start_checkpoint']):
        parsed['start_checkpoint'] = parsed['checkpoints_dir'] + parsed['start_checkpoint']
        assert os.path.exists(parsed['start_checkpoint']), f"start_checkpoint file not found: {parsed['start_checkpoint']}"

    if parsed['ddp']:
        if not parsed['dist_url']:
            parsed['dist_url'] = f"file://{parsed['main_folder']}DDP_TEMP_FILE_{np.random.randint(10000)}"
        parsed['seed'] = int(parsed['dist_url'].split("_")[-1])

    parsed['dvqa_floats'] = [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                             23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
                             38.0, 39.0, 40.0, 41.0, 43.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 1000.0, 10000.0,
                             100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0]
    return parsed
