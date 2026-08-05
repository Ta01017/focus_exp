import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from pathlib import Path
import json
import sys

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from training_run import configure_training_run
import warnings
warnings.filterwarnings("ignore")


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/swinir/train_swinir_sr_lightweight.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--train-metadata')
    parser.add_argument('--val-metadata')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=-1)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-train-steps', type=int, default=-1)
    parser.add_argument('--output-dir')
    parser.add_argument('--init-mode', choices=('scratch', 'official', 'resume'), default='scratch')
    parser.add_argument('--init-checkpoint-dir')
    parser.add_argument('--resume-dir')
    parser.add_argument('--overwrite-output', type=int, choices=(0, 1), default=0)
    parser.add_argument('--fail-on-split-overlap', type=int, choices=(0, 1), default=0)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    if args.init_mode in ('scratch', 'resume'):
        opt['n_channels'] = 3
        opt['netG']['in_chans'] = 3
        for dataset_opt in opt['datasets'].values():
            dataset_opt['n_channels'] = 3
    opt['dist'] = args.dist
    if bool(args.train_metadata) != bool(args.val_metadata):
        raise ValueError('--train-metadata and --val-metadata must be provided together')
    if args.train_metadata:
        opt['datasets']['train']['dataset_type'] = 'metadata_mff'
        opt['datasets']['train']['metadata'] = str(Path(args.train_metadata).resolve())
        opt['datasets']['test']['dataset_type'] = 'metadata_mff'
        opt['datasets']['test']['metadata'] = str(Path(args.val_metadata).resolve())
        opt['datasets']['train']['start_index'] = args.start_index
        opt['datasets']['train']['max_samples'] = args.max_samples
        opt['datasets']['test']['max_samples'] = args.max_samples
        if args.num_workers is not None:
            opt['datasets']['train']['dataloader_num_workers'] = args.num_workers

    current_step, run_manifest = configure_training_run(opt, args)
    print('[INIT] mode={init_mode} G={loaded_G} E={loaded_E} optimizer={loaded_optimizerG}'.format(**run_manifest))

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        for key, path in opt['path'].items():
            print(path)
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)
        with (Path(run_manifest['output_dir']) / 'options_resolved.json').open('w', encoding='utf-8') as handle:
            json.dump(opt, handle, indent=2)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = args.seed if args.seed is not None else opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                drop_last = len(train_set) >= dataset_opt['dataloader_batch_size']
                if not drop_last:
                    logger.info('Dataset smaller than batch size; disabling drop_last for smoke.')
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=drop_last,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=args.num_workers if args.num_workers is not None else 1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    if args.train_metadata:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from metadata_training import warn_split_overlap
        warn_split_overlap(train_set, test_set, bool(args.fail_on_split_overlap))

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if args.init_mode == 'resume':
        state_path = Path(args.resume_dir) / 'checkpoints' / 'training_state.pt'
        if state_path.is_file():
            state = torch.load(state_path, map_location='cpu')
            for scheduler, scheduler_state in zip(model.schedulers, state.get('schedulers', [])):
                scheduler.load_state_dict(scheduler_state)
            if int(state.get('current_step', current_step)) != current_step:
                raise ValueError('resume training_state step does not match checkpoint filenames')
        else:
            raise FileNotFoundError(f'resume scheduler state missing: {state_path}')
    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    need_GT = opt['datasets']['train']['dataset_type'] in ['mef_GT', 'mff_GT', 'metadata_mff']
    for epoch in range(5000):  # keep running
        if hasattr(train_set, 'set_epoch'):
            train_set.set_epoch(epoch)
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------

            model.feed_data(train_data, need_GT=need_GT)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)
            if args.max_train_steps >= 0 and current_step >= args.max_train_steps:
                for test_data in test_loader:
                    model.feed_data(test_data, need_GT=True, phase='test')
                    model.test()
                    print('one-batch metadata validation: PASS')
                    break
                return

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                save_dir = opt['path']['models'] 
                save_filename = '{}_{}.pth'.format(current_step, 'E')
                save_path = os.path.join(save_dir, save_filename)
                logger.info('Saving the model. Save path is:{}'.format(save_path))
                model.save(current_step)
                torch.save({'current_step': current_step,
                            'schedulers': [scheduler.state_dict() for scheduler in model.schedulers]},
                           Path(opt['path']['models']) / 'training_state.pt')

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                avg_psnr = 0.0
                idx = 0
                
                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['A_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data, phase='test')
                    model.test()
                    visuals = model.current_visuals(need_H=need_GT)
                    E_img = util.tensor2uint(visuals['E'])
                    if need_GT:
                        H_img = util.tensor2uint(visuals['GT'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)
                    print("save path:{}".format(save_img_path))
                    if need_GT:
                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                        avg_psnr += current_psnr
                if need_GT:
                    avg_psnr = avg_psnr / idx

                    # testing log
                    logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
