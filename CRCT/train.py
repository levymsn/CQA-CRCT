import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fig_dataloader import PlotQA_Dataset
from options import read_command_line
from backbone.encoder_decorator import VisualDialogEncoder
from utils import WarmupLinearScheduleNonZero, get_optimizer, log_line, init_log_file
from backbone.encoder_decorator import forward
import pprint
import time
from time import gmtime, strftime
from timeit import default_timer as timer
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from evaluation import print_acc_table, print_breakdown_table, plotqa_evaluate_DDP


def run_training_DDP(gpu, params):
    if params['ddp']:
        params['rank'] = gpu + params['rank_from']
        gpu = gpu + params['gpu_from']
        torch.cuda.set_device(gpu)
        print("({}) is waiting for all DDP proccess...".format(params['rank']))
        dist.init_process_group(backend="nccl", init_method=params['dist_url'],
                                world_size=params['world_size'], rank=params['rank'])

    if params['rank'] == 0:
        init_log_file(params)
        writer = SummaryWriter(log_dir=params['tensorboard'] + params['save_name'])

    log_line(params,
             line="De facto batch_size: {}*{}*{} = {}".format(params['batch_size'],
                                                              params['world_size'],
                                                              params['batch_multiply'],
                                                              params['batch_size'] * params['world_size'] * params['batch_multiply']),
             all_ranks=True)

    log_line(params, line="Use GPU: {} for training, rank: {}, seed: {}".format(gpu, params['rank'], params['seed']), all_ranks=True)

    splits_to_load = ['train', params['eval_set']]
    if params['pretrain']:
        splits_to_load = ['train']
    dataset = PlotQA_Dataset(params, splits_to_load)

    # --- original 80 is batch_size=40 ------
    batch_size = params['batch_size']
    log_line(params, line="batch_size={}".format(batch_size))
    dataset.split = 'train'
    loss_avg_lst = None

    if params['ddp']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=params['seed'])
        # train_sampler = OVERFEAT_sampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(not params['ddp']),
            num_workers=params['num_workers'],
            drop_last=True,
            pin_memory=False)
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(not params['ddp']),
            num_workers=params['num_workers'],
            drop_last=True,
            pin_memory=False)

    iters_per_epoch = (len(dataloader) / params['batch_multiply'])
    params['device'] = device = torch.device(("cuda:" + str(gpu)) if torch.cuda.is_available() else "cpu")
    if params['cuda_num'] >= 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(params['cuda_num'])
        params['device'] = device = torch.device("cuda:" + str(params['cuda_num']))

    crct_model = VisualDialogEncoder(params)
    crct_model.to(device)
    # params['cuda_num'] = gpu

    optimizer = get_optimizer(params, crct_model)

    scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=params['warmup'], min_lr=params['min_lr'], t_total=iters_per_epoch * 20)

    start_iter_id = 0
    cont_epoch = 0
    if params['start_checkpoint']:
        pretrained_dict = torch.load(params['start_checkpoint'], map_location=params['device'])
        if not params['continue']:
            if 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']

            model_dict = crct_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print("number of keys transferred", len(pretrained_dict))
            assert len(pretrained_dict.keys()) > 0
            model_dict.update(pretrained_dict)
            crct_model.load_state_dict(model_dict)
            del pretrained_dict, model_dict
            print("Current epoch: {}".format(cont_epoch))
        else:
            cont_epoch = int(params['start_checkpoint'].split("/")[-1].split("_")[2]) + 1
            model_dict = crct_model.state_dict()
            optimizer_dict = optimizer.state_dict()
            pretrained_dict_model = pretrained_dict['model_state_dict']
            pretrained_dict_optimizer = pretrained_dict['optimizer_state_dict']
            pretrained_dict_scheduler = pretrained_dict['scheduler_state_dict']
            pretrained_dict_model = {k: v for k, v in pretrained_dict_model.items() if k in model_dict}
            pretrained_dict_optimizer = {k: v for k, v in pretrained_dict_optimizer.items() if k in optimizer_dict}
            model_dict.update(pretrained_dict_model)
            optimizer_dict.update(pretrained_dict_optimizer)
            crct_model.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=params['warmup'], min_lr=params['min_lr'],
                                                    t_total=iters_per_epoch * 20, last_epoch=pretrained_dict["iter_id"])
            scheduler.load_state_dict(pretrained_dict_scheduler)
            start_iter_id = pretrained_dict['iter_id']
            if 'loss_avg' in pretrained_dict:
                loss_avg_lst = pretrained_dict['loss_avg']
            del pretrained_dict, pretrained_dict_model, pretrained_dict_optimizer, pretrained_dict_scheduler, \
                model_dict, optimizer_dict
            torch.cuda.empty_cache()

    num_iter_epoch = len(dataloader)

    # print('\n%d iter per epoch.' % num_iter_epoch)
    log_line(params, line="len(dataloader)={}".format(len(dataloader)))
    log_line(params, line="len(dataset) / batch_size = {}".format(len(dataset) / batch_size))

    if params['ddp']:
        crct_model = torch.nn.parallel.DistributedDataParallel(crct_model,
                                                                   device_ids=[gpu],
                                                                   find_unused_parameters=True
                                                                   )
        dist_grp = dist.new_group(list(range(params['world_size'])))

    crct_model.to(device)

    start_t = timer()
    optimizer.zero_grad()

    PRINT_EVERY = 100
    loss_lst = {'loss': [], 'nsp': [], 'reg': [], 'leg_loss': [], 'lm_loss': []}
    if loss_avg_lst is None:
        loss_avg_lst = {'loss': [], 'nsp': [], 'reg': [], 'leg_loss': [], 'lm_loss': []}

    log_line(params, line="Starting iterations...")

    scaler = torch.cuda.amp.GradScaler()

    num_step_iterations = 0
    for epoch_id in range(params['num_epochs']):
        if params['ddp']:
            train_sampler.set_epoch(epoch_id)

        step_iter_id = start_iter_id + num_step_iterations
        epoch_time = time.time()

        for iter_id, batch in enumerate(dataloader):
            step_iter_id = start_iter_id + num_step_iterations
            crct_model.train()
            num_regs = torch.sum(batch['needs_reg']).item()

            with torch.cuda.amp.autocast():
                loss, lm_loss, nsp_loss, img_loss, nsp_scores, regression, legend_loss = forward(crct_model, batch, params)
                reg_loss = regression[1][batch['needs_reg'].view(-1)]
                reg_5_right, reg_t_right = regression[3]
                reg_5_dist = regression[4][batch['needs_reg'].view(-1)]

                reg_loss = 0 if torch.isnan(reg_loss.mean()) else reg_loss.mean().item()
                reg_5_dist = 0 if torch.isnan(reg_5_dist.mean()) else reg_5_dist.mean().item()

                ddp_share_tensor = torch.tensor([loss.item(), lm_loss.mean().item(), nsp_loss.mean().item(),
                                                 reg_loss, reg_5_dist, legend_loss.mean().item(),
                                                 num_regs, reg_5_right, reg_t_right]).cuda()
                if params['ddp'] and params['world_size'] > 1:
                    dist.all_reduce(ddp_share_tensor,
                                    op=torch.distributed.ReduceOp.SUM,
                                    group=dist_grp)

                    ddp_share_tensor[:-3] = ddp_share_tensor[:-3] / params['world_size']

                total_loss, lm_loss, nsp_loss, reg_loss, reg_5_dist, leg_loss, num_regs, reg_5_right, reg_t_right = ddp_share_tensor.cpu()
                reg_5_accuracy = reg_5_right / num_regs
                reg_t_accuracy = reg_t_right / num_regs
                num_regs = num_regs, batch['R'].shape[0] * params['world_size']

                if not torch.isnan(reg_loss):
                    loss_lst['reg'].append(reg_loss)
                loss_lst['nsp'].append(nsp_loss)
                loss_lst['loss'].append(total_loss)
                loss_lst['lm_loss'].append(lm_loss)

                if not torch.isnan(leg_loss):
                    loss_lst['leg_loss'].append(leg_loss)

                if params['batch_multiply'] > 1:
                    loss /= params['batch_multiply']

            scaler.scale(loss).backward()

            if iter_id % params['batch_multiply'] == 0:
                # optimizer.step()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                scheduler.step()

                if params['rank'] == 0:
                    writer.add_scalar('Loss/Total Loss', total_loss, step_iter_id)
                    writer.add_scalar('Loss/nsp', nsp_loss, step_iter_id)

                    writer.add_scalar('Reg Loss/reg_MSE', reg_loss, step_iter_id)
                    writer.add_scalar('Reg Loss/reg_5_dist', reg_5_dist, step_iter_id)
                    writer.add_scalar('Accuracy/reg_acc', reg_5_accuracy, step_iter_id)
                    writer.add_scalar('Accuracy/reg_t_acc', reg_t_accuracy, step_iter_id)

                del loss

            if iter_id % PRINT_EVERY == 0:
                end_t = timer()
                cur_epoch = epoch_id + float(iter_id) / num_iter_epoch
                timestamp = strftime('%a %X', gmtime())
                print_nsp_loss = 0

                if nsp_loss is not None:
                    print_nsp_loss = nsp_loss.item()
                est = (num_iter_epoch - float(iter_id)) * ((end_t - start_t) / (PRINT_EVERY))  # left * avg_time
                est = strftime('%H:%M', gmtime(est))
                loss_avg_lst['loss'].append(np.array(loss_lst['loss'][-PRINT_EVERY:]).mean())
                loss_avg_lst['nsp'].append(np.array(loss_lst['nsp'][-PRINT_EVERY:]).mean())
                loss_avg_lst['reg'].append(np.array(loss_lst['reg'][-PRINT_EVERY:]).mean())
                loss_avg_lst['leg_loss'].append(np.array(loss_lst['leg_loss'][-PRINT_EVERY:]).mean())
                loss_avg_lst['lm_loss'].append(np.array(loss_lst['lm_loss'][-PRINT_EVERY:]).mean())

                if params['binary_answers']:
                    preds = torch.round(F.softmax(nsp_scores.float().cpu(), dim=1)[:, 0]).long()
                    pred_ones = (torch.count_nonzero(preds).item() / len(batch['next_sentence_labels']))
                    mistakes = torch.count_nonzero(preds - (1 - batch['next_sentence_labels'].view(-1)))
                    acc = 1 - (mistakes.item() / len(batch['next_sentence_labels']))
                    print_format = '[Ep: %.2f][%s][lr: %.2e][Iter: %d][Time: %5.2fs][Est: %s][Loss: %.3g][NSP: %.3g][Acc: %.3g][Preds 1s: %.3g][100 mean nsp: (%.3g)]'
                    print_info = [cont_epoch + cur_epoch,
                                  timestamp, optimizer.param_groups[0]['lr'],
                                  step_iter_id, end_t - start_t, est, total_loss.item(),
                                  print_nsp_loss, acc, pred_ones,
                                  loss_avg_lst['nsp'][-1],
                                  ]

                    if params['rank'] == 0:
                        writer.add_scalar('Accuracy/cls', acc, step_iter_id)
                        writer.add_scalar("Accuracy/Preds 1s", pred_ones, step_iter_id)
                else:
                    print_format = '[Ep: %.2f][%s][lr: %.2e][Iter: %d][Time: %5.2fs][Est: %s][Loss: %.3g][NSP: %.3g][Reg: %.3g][Regs: %d/%d][Reg_acc: %.2g | %.2g][100 mean r,n: (%.3g , %.3g)]'

                    print_info = [cont_epoch + cur_epoch,
                                  timestamp, optimizer.param_groups[0]['lr'],
                                  step_iter_id, end_t - start_t, est, total_loss.item(),
                                  print_nsp_loss, reg_loss.item(),
                                  num_regs[0], num_regs[1],
                                  reg_5_accuracy.item(),
                                  reg_t_accuracy.item(),
                                  loss_avg_lst['reg'][-1],
                                  loss_avg_lst['nsp'][-1],
                                  # loss_avg_lst['lm_loss'][-1],
                                  # loss_avg_lst['leg_loss'][-1],
                                  ]
                log_line(params, line=print_format % tuple(print_info), all_ranks=(params['num_proc'] <= 1 or (params['rank'] == params['rank_from'])))
                start_t = end_t

            if iter_id % params['batch_multiply'] == 0:
                num_step_iterations += 1
            # -------- end of iters ------------

        log_line(params, line=f"Epoch Time: {strftime('%H:%M', gmtime(time.time() - epoch_time))}")
        # save the model
        file_name = 'plotqa_encoder_%d_%d.ckpt' % (cont_epoch + epoch_id, step_iter_id + 1)
        log_line(params, line="     --> Saving model as: {}".format(os.path.join(params['save_path'], file_name)))

        if params['rank'] == 0:
            torch.save(
                {'model_state_dict': crct_model.module.state_dict(), 'scheduler_state_dict': scheduler.state_dict() \
                    , 'optimizer_state_dict': optimizer.state_dict(), 'iter_id': step_iter_id + 1},
                os.path.join(params['save_path'], file_name))
        # -------
        # fire evaluation
        if not params['no_eval']:
            log_line(params, line="Starting evaluation (on random sampling from Val set)...")
            eval_batch_size = params['eval_batch_size']
            # try:
            start_eval_time = time.time()
            dataset.split = params['eval_set']

            class sub_ddp_sampler(torch.utils.data.distributed.DistributedSampler):
                def __iter__(self):
                    iter_ = super().__iter__()
                    limit = 500
                    for obj in iter_:
                        yield obj
                        limit -= 1
                        if limit == 0:
                            break

            if params['ddp']:
                val_sampler = sub_ddp_sampler(dataset)
                val_sampler.set_epoch(epoch_id)
            else:
                val_indices = np.arange(len(dataset))
                np.random.shuffle(val_indices)
                val_sampler = torch.utils.data.SubsetRandomSampler(val_indices[:500])

            dataloader_val = DataLoader(
                dataset,
                batch_size=100,
                shuffle=False,
                sampler=val_sampler,
                num_workers=params['num_workers'],
                drop_last=True,
                pin_memory=False)

            total_correct_tensor, breakdown_tensor = plotqa_evaluate_DDP(dataloader_val,
                                                                         dataset,
                                                                         params,
                                                                         eval_batch_size,
                                                                         crct_model,
                                                                         csv=False)
            print_acc_table(lambda msg: log_line(params, msg), total_correct_tensor)
            if params['dataset'] != 'figure_qa':
                print_breakdown_table(params, lambda msg: log_line(params, msg),  breakdown_tensor)

            log_line(params, line="     -> Eval time: {}".format(round(time.time() - start_eval_time, 2)))

            if params['rank'] == 0:
                eval_acc = total_correct_tensor[4, 0] / total_correct_tensor[4, 1]
                writer.add_scalar('Accuracy/Eval Total Acc', eval_acc, epoch_id)
                eval_reg_acc = total_correct_tensor[2, 0] / total_correct_tensor[2, 1]
                writer.add_scalar('Accuracy/Eval Reg Acc', eval_reg_acc, epoch_id)
                nsp_acc = total_correct_tensor[0, 0] / total_correct_tensor[0, 1]
                writer.add_scalar('Accuracy/Eval nsp Acc', nsp_acc, epoch_id)

                writer.add_hparams(
                    {key: params[key] for key in params if type(params[key]) in [int, float, str, bool, torch.Tensor]},
                    {'hparam/Eval_acc': eval_acc})

            dataset.split = 'train'
            crct_model.train()


if __name__ == '__main__':
    params = read_command_line()
    pprint.pprint(params)

    if params['ddp']:
        mp.spawn(run_training_DDP, nprocs=params['num_proc'], args=(params,))
    else:
        run_training_DDP("0", params)
