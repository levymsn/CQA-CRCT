import os
import time
import torch
import tqdm
from torch.utils.data import DataLoader
from backbone.encoder_decorator import VisualDialogEncoder
from fig_dataloader import PlotQA_Dataset
import options
import pprint
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import math
import glob
from utils import Evaluation_Log
from backbone.encoder_decorator import forward
import torch.nn.functional as F
from pandas import DataFrame
import matplotlib.pyplot as plt


def get_encoder(params, ckpt):
    if params['cuda_num'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['cuda_num'])

    device = params['device']
    # print("CUDA_VISIBLE_DEVICES={}".format(os.environ["CUDA_VISIBLE_DEVICES"]), flush=True)
    dialog_encoder = VisualDialogEncoder(params).to(device)

    if ckpt:
        pretrained_dict = torch.load(ckpt, map_location=device)

        if not params['continue']:
            if 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']

            model_dict = dialog_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # print("number of keys transferred", len(pretrained_dict))
            assert len(pretrained_dict.keys()) > 0
            model_dict.update(pretrained_dict)
            dialog_encoder.load_state_dict(model_dict)
            del pretrained_dict, model_dict
        else:
            model_dict = dialog_encoder.state_dict()
            pretrained_dict_model = pretrained_dict['model_state_dict']
            pretrained_dict_model = {k: v for k, v in pretrained_dict_model.items() if k in model_dict}
            model_dict.update(pretrained_dict_model)
            dialog_encoder.load_state_dict(model_dict)
            start_iter_id = pretrained_dict['iter_id']

            del pretrained_dict, pretrained_dict_model, model_dict
            torch.cuda.empty_cache()

    # print('\n%d iter per epoch.' % num_iter_epoch)
    # dialog_encoder.to(device)
    if params['ddp'] and params['world_size'] > 1:
        print("Waiting for all DDP ranks...")
        dialog_encoder = torch.nn.parallel.DistributedDataParallel(dialog_encoder,
                                                                   device_ids=[params['rank']],
                                                                   find_unused_parameters=True)
        print("Done")

    dialog_encoder.to(device)

    return dialog_encoder


class sub_ddp_sampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False, done_ids=None):
        super(sub_ddp_sampler, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.indices = None
        if done_ids is not None:
            self.indices = np.array(list(set(range(len(self.dataset))) - set(done_ids)))
            assert len(self.indices) > 0
            if self.drop_last and len(self.indices) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(self.indices) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.num_samples = math.ceil(len(self.indices) / self.num_replicas)  # type: ignore

            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = list(self.indices[torch.randperm(len(self.indices), generator=g)])  # type: ignore
        else:
            indices = list(self.indices)  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def get_prev_csvs(params):
    from pandas import read_csv, concat
    files = glob.glob(f"{params['save_path']}/eval_results_{params['eval_set']}_{params['start_checkpoint'].split('/')[-1]}_*.csv")
    if len(files) == 0:
        return None
    csvs = [read_csv(file_name) for file_name in files]
    vals = concat(csvs)
    done_ids = vals['qa_ind'].to_numpy()
    return done_ids


def evaluate_plotqa(gpu, params):
    if params['ddp']:
        torch.cuda.set_device(gpu)
        params['rank'] = gpu + params['rank_from']
        print(f"({params['rank']}) is waiting for all DDP proccess...")
        dist.init_process_group(backend="nccl", init_method=params['dist_url'],
                                world_size=params['world_size'], rank=params['rank'])

    dataset = PlotQA_Dataset(params, params['eval_set'], init_split=params['eval_set'])
    dataset.get_all_answers = True

    params['device'] = torch.device(("cuda:" + str(gpu)) if torch.cuda.is_available() else "cpu")

    plotqa_encoder = get_encoder(params, params['start_checkpoint'])

    print("Model's parameters:", sum(p.numel() for p in plotqa_encoder.parameters()))

    done_ids = get_prev_csvs(params)
    cont_eval = False
    if done_ids is not None:
        print("-"*20, f"Done Ids: {len(done_ids)}", "-"*20, flush=True)
        train_sampler = sub_ddp_sampler(dataset, done_ids=done_ids)
        cont_eval = True
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # ----------------------

    if params['ddp']:
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        # train_sampler = sub_ddp_sampler(dataset, done_ids=None)

        dataloader = DataLoader(
            dataset,
            batch_size=params['eval_batch_size'],
            sampler=train_sampler,
            shuffle=False,
            num_workers=params['num_workers'],
            drop_last=False,
            pin_memory=False)
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=params['eval_batch_size'],
            sampler=train_sampler,
            shuffle=False,
            num_workers=params['num_workers'],
            # sampler=torch.utils.data.SubsetRandomSampler(range(100)),
            drop_last=False,
            pin_memory=True)
    #
    # tensor = torch.ones(7).cuda()
    # dist.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=dist.new_group(list(range(params['world_size']))))
    # print('Rank ', params['rank'], ' has data ', tensor)
    if params['ddp']:
        train_sampler.set_epoch(params['seed'])

    total_correct_tensor, breakdown_tensor = plotqa_evaluate_DDP(dataloader,
                                                                 dataset,
                                                                 params,
                                                                 params['eval_batch_size'],
                                                                 plotqa_encoder,
                                                                 progress=params['eval_type'],
                                                                 print_rank=True,
                                                                 plot_hist=True,
                                                                 csv=True,
                                                                 cont_eval=cont_eval
                                                                 )

    print("Done evaluation", flush=True)


def plotqa_evaluate_DDP(dataloader, dataset, params, eval_batch_size, dialog_encoder, progress=False,
                        plot_hist=True, print_rank=False, csv=True, cont_eval=False):
    dialog_encoder.eval()
    file_path = os.path.join(f"{params['save_path']}/eval_fix_{params['eval_set']}_{params['start_checkpoint'].split('/')[-1]}.txt")

    def log_print(msg, dont_print=False):
        if (not dont_print) and ((not print_rank) or params['rank'] == 0):
            print(msg, flush=True)
        if params['rank'] == 0 and progress:
            with open(file_path, 'a+') as eval_file:
                eval_file.write(str(msg) + "\n")
    # save environment
    log_print(str(params), dont_print=True)

    breakdown_tensor = torch.zeros(5, 4, 3, 3).cuda().double()
    total_correct_tensor = torch.zeros(6, 2).cuda().double()
    histogram = torch.zeros(13).cuda().long()

    correct_dict = {}
    if csv:
        predictions = Evaluation_Log(params, ["qa_ind", "gt_cls", 'pred_cls', "gt_reg", 'pred_reg',
                                              'reg_target', 'reg_%_error', 'reg_t_error'], cont_eval=cont_eval)
        if cont_eval:
            breakdown_tensor, total_correct_tensor, histogram = predictions.load_tensors()

    with torch.no_grad():
        batch_size = eval_batch_size
        if params['ddp']:
            dist_g = dist.new_group(list(range(params['world_size'])))

        total_inf_time = [0, 0]
        for batch in tqdm.tqdm(dataloader):
            dataset.cut_batch_padding(batch)
            if batch['id'].shape[0] == 0:
                continue

            output = []
            reg_output = []
            # accuracy by 5% measure
            reg_loss_lst = []
            # accuracy by tick measure loss
            reg_t_loss_lst = []

            for j in range(int(np.ceil(batch['tokens'].shape[0] / batch_size))):
                # create chunks of the original batch
                chunk = j * batch_size, min((j + 1) * batch_size, batch['tokens'].shape[0])
                start = time.time()
                _, _, _, _, nsp_scores, regression = forward(dialog_encoder, batch, params,
                                                             output_nsp_scores=True,
                                                             evaluation=True,
                                                             sample_ids=np.arange(chunk[0], chunk[1])
                                                             )

                # normalize nsp scores
                nsp_probs = F.softmax(nsp_scores, dim=1)
                total_inf_time[0] += time.time() - start

                assert nsp_probs.shape[-1] == 2
                output.append(nsp_probs[:, 0])
                reg_output.append(regression[0])
                reg_loss_lst.append(regression[4])
                reg_t_loss_lst.append(regression[2])
                assert regression[0].shape[0] == regression[1].shape[0] == regression[2].shape[0] == nsp_probs.shape[0]

            total_inf_time[1] += len(batch['num_ans'])
            # print(f"Avg inference time: {total_inf_time[0] / total_inf_time[1]} for {total_inf_time[1]} samples.")
            del regression
            del nsp_probs
            output = torch.cat(output, dim=0)
            reg_output = torch.cat(reg_output, dim=0)
            reg_loss_lst = torch.cat(reg_loss_lst, dim=0)
            reg_t_loss_lst = torch.cat(reg_t_loss_lst, dim=0)
            assert batch['tokens'].shape[0] == output.shape[0] == reg_output.shape[0] == reg_loss_lst.shape[0] == reg_t_loss_lst.shape[0]

            total_options = 0
            answers = []
            answers_probs = []
            reg_answers_loss = []
            reg_answers_t_loss = []
            reg_answers_output = []
            if params['binary_answers']:
                answers = np.round(output.cpu())
                reg_answers_loss = torch.zeros_like(batch['id'].view(-1)).cpu()
                reg_answers_t_loss = torch.zeros_like(batch['id'].view(-1)).cpu()
                reg_answers_output = torch.zeros_like(batch['id'].view(-1)).cpu()
                nsp_right = (answers == (1 - batch['next_sentence_labels'].view(-1)))
            else:
                for i, n in enumerate(batch['num_ans']):
                    if '_REGS' in params['qa_file']:
                        ans_id = batch['gt_id'][i][0]
                    else:
                        ans_id = torch.argmax(output[total_options: total_options + n])
                    answers.append(ans_id)
                    reg_answers_loss.append(reg_loss_lst[total_options: total_options + n][ans_id.item()])
                    reg_answers_t_loss.append(reg_t_loss_lst[total_options: total_options + n][ans_id.item()])
                    reg_answers_output.append(reg_output[total_options: total_options + n][ans_id.item()])
                    total_options += n

                assert total_options == batch['tokens'].shape[0]
                answers = torch.stack(answers, dim=0).cpu()
                reg_answers_loss = torch.stack(reg_answers_loss, dim=0).cpu()
                reg_answers_t_loss = torch.stack(reg_answers_t_loss, dim=0).cpu()
                reg_answers_output = torch.stack(reg_answers_output, dim=0).cpu()

                assert answers.shape[0] == reg_answers_loss.shape[0] == reg_answers_t_loss.shape[0] == reg_answers_output.shape[0] == batch['id'].shape[0], answers_probs.shape[0]

                nsp_right = (answers == batch['gt_id'].view(-1))
            needs_regression = batch['needs_reg'].view(-1)
            reg_right = ((reg_answers_loss <= 0.05) & needs_regression)
            reg_t_right = ((reg_answers_t_loss <= batch['tolerance_margin'].view(-1)) & needs_regression)

            correct_answers = (nsp_right & (needs_regression.logical_not() | reg_right))
            correct_answers_t_loss = (nsp_right & (needs_regression.logical_not() | reg_t_right))

            reg_cls_right = nsp_right & needs_regression

            cls_tup = (batch['id'].view(-1), batch['gt_id'].view(-1), answers)
            reg_tup = (batch['gt'].view(-1), reg_answers_output, batch['reg_target'].view(-1), reg_answers_loss, reg_answers_t_loss)
            if csv:
                # log the predictions
                data = torch.stack(cls_tup + reg_tup).permute(1, 0).numpy().copy()
                data[needs_regression.logical_not(), 3:] = np.nan
                predictions.append(data)

            reduce_total_acc(total_correct_tensor, needs_regression, nsp_right, reg_right, reg_t_right, dist_g)

            if progress:
                if params['binary_answers']:
                    log_print(f"{'='*20} Total: {total_correct_tensor[0, 1]} {'='*20}")
                print_acc_table(log_print, total_correct_tensor)

            if 'plotqa' in params['dataset']:
                reduce_breakdown_table(dataset, dist_g, params,
                                       breakdown_tensor, batch,
                                       zip(correct_answers, correct_answers_t_loss))
                if csv:
                    predictions.save_tensors(breakdown_tensor, total_correct_tensor, histogram)

                if progress:
                    if plot_hist:
                        reduce_histogram(histogram, reg_answers_loss[needs_regression].view(-1), dist_g)
                        # reg_5_dist = reg_answers_loss
                        # print([target for target in batch['reg_target'].view(-1)[(1 <= reg_5_dist) & needs_regression]])
                        if params['rank'] == 0:
                            try:
                                make_hist(params, histogram)
                            except:
                                pass
                    print_breakdown_table(params, log_print, breakdown_tensor)

                    if progress == 'correct_dict':
                        for qa_ind in batch['id'][correct_answers]:
                            raw = dataset.get_raw(qa_ind)
                            if raw['image_index'] not in correct_dict:
                                correct_dict[raw['image_index']] = []

                            correct_dict[raw['image_index']].append(qa_ind.item())
                        if total_correct_tensor[0, 1] % 1000 == 0:
                            log_print(correct_dict)

                    if progress == "examples":
                        log_print("")
                        right_regs = reg_cls_right & reg_right
                        for i, g in enumerate(batch['id'][right_regs]):
                            raw = dataset.get_raw(g)
                            log_print("{}, [I{}] Q:{}, output: {} || {} || loss: {}".format(g.item(), raw['image_index'], raw['question_string'], reg_answers_output[right_regs][i], raw['answer'], reg_answers_loss[right_regs][i]))

                        data_r_no_reg = (nsp_right) & (needs_regression.logical_not())
                        for i, g in enumerate(batch['id'][data_r_no_reg]):
                            raw = dataset.get_raw(g)
                            # if raw['template'] in ['structural', 'data_retrieval']:
                            #     continue
                            ans = dataset.get_possible_answers(raw['image_index'])[answers[data_r_no_reg][i]]
                            if str(ans) != str(raw['answer']):
                                log_print("*"*50)
                                log_print(dataset.get_possible_answers(raw['image_index']))
                                log_print((answers[data_r_no_reg].shape, batch['id'][data_r_no_reg].shape))
                                log_print(answers[data_r_no_reg][i])
                                log_print(batch['gt_id'][data_r_no_reg][i])
                                log_print(batch['id'][data_r_no_reg][i])

                                log_print("*"*50)
                            log_print("*{}, <I{}> Q: {} . A: {} || {}".format(g.item(), raw['image_index'], raw['question_string'], ans, raw['answer']))

    dialog_encoder.train()

    return total_correct_tensor, breakdown_tensor


def make_hist(params, histogram):
    plt.style.use('ggplot')
    fig_title = ''
    if params['start_checkpoint']:
        fig_title = params['start_checkpoint'].split('/')
        fig_title = fig_title[-2] + "/" + fig_title[-1] + "\n"
    fig_title += "reg_acc: [{}/{}] {}%".format(histogram[0].item(),
                                               torch.sum(histogram).item(),
                                               round((histogram[0].item() / torch.sum(histogram).item()) * 100, 2))
    hist_list = histogram.cpu().tolist()
    bars_dict = {'0-5': hist_list[0],
                 '5-10': hist_list[1],
                 '10-15': hist_list[2],
                 '15-20': hist_list[3],
                 '20-30': hist_list[4],
                 '30-40': hist_list[5],
                 '40-50': hist_list[6],
                 '50-60': hist_list[7],
                 '60-70': hist_list[8],
                 '70-80': hist_list[9],
                 '80-90': hist_list[10],
                 '90-100': hist_list[11],
                 '100+': hist_list[12]
                 }

    bars, heights = zip(*bars_dict.items())

    fig, ax = plt.subplots()
    ax.title.set_text(fig_title + "   all regression outputs")

    ax.bar(range(len(bars)), height=heights)
    plt.xticks(range(len(bars)), bars, rotation="vertical")
    for i, v in enumerate(heights):
        ax.text(i - 0.1, v, str(v), color='black')

    fig.show()
    if params['hist_name']:
        fig.savefig("results/EVAL/" + params['start_checkpoint'].split('/')[-1] + "_" + params['hist_name'] + ".png")
    else:
        fig.savefig(os.path.join(params['save_path'] + "/Eval_hist_" + params['eval_set'] + "_" + params['start_checkpoint'].split("/")[-1] + ".png"))

    plt.cla()
    plt.close(fig)


def get_qcat_by_qid(qid):
    structural_qid = ['S7', 'S17', 'S6', 'S1', 'S4', 'S3', 'S5', 'S2', 'S0', 'S8', 'S9', 'S15', 'S10', 'S13', 'S14', 'S16', 'S11', 'S12']
    data_retrieval_qid = ['D15', 'D9', 'D12', 'D8', 'D7', 'D10', 'D11', 'D14', 'D5', 'D2', 'D13', 'D3', 'D0', 'D4', 'D1']
    # reasoning_qid = ['M4', 'M5', 'A4', 'A5', 'A6', 'A7', 'A8', 'C5', 'CD5', 'CD6', 'CD7', 'D6', 'D18', 'C4', 'CD8', 'M6', 'CD10', 'D17', 'M7', 'C6', 'CD9', 'CD11', 'M0', 'M1', 'C0', 'C1', 'C2', 'CD4', 'M2', 'A0', 'A1', 'A2', 'A3', 'CD1', 'CD3', 'M3', 'D16', 'CD2', 'C3', 'CD0', 'A9']
    if qid in structural_qid:
        template = 's', 0
    elif qid in data_retrieval_qid:
        template = 'd', 1
    else:
        template = 'r', 2

    return template


def accuracy_by_cats(qid_acc_dict):
    """ Get the total accuracy by categories"""

    acc = {'s': [0, 0], 'd': [0, 0], 'r': [0, 0]}
    for qid in qid_acc_dict:
        template, _ = get_qcat_by_qid(qid)
        acc[template][0] += qid_acc_dict[qid][0]
        acc[template][1] += qid_acc_dict[qid][1]

    for c in acc:
        if acc[c][1] == 0:
            acc[c][1] = 1
            acc[c][0] = -1
    # return structural_acc, data_retrieval_acc, reasoning_acc
    return acc['s'][0] / acc['s'][1], acc['d'][0] / acc['d'][1], acc['r'][0] / acc['r'][1]


def reduce_breakdown_table(dataset, dist_g, params, breakdown_tensor, batch, corrects):
    """ Calculates the accuracy table for the paper"""
    qid_ind_map = {'Total': 0, 'line': 1, 'vbar': 2, 'hbar': 3, 'dot': 4}
    vocab_table_tensor = torch.zeros(breakdown_tensor.shape).cuda().double()

    for qa_i, (correct, t_correct) in enumerate(corrects):
        qa_id, qid, qa_type, is_reg = batch['id'][qa_i], batch['qid'][qa_i], batch['qa_type'][qa_i], batch['needs_reg'][
            qa_i]
        ans_type = dataset.get_ans_type(qa_id)
        _, qcat_id = get_qcat_by_qid(qid)

        for fig_id in [0, qid_ind_map[qa_type]]:
            vocab_table_tensor[fig_id, ans_type, qcat_id, 0] += 1 if correct else 0
            vocab_table_tensor[fig_id, ans_type, qcat_id, 1] += 1 if t_correct else 0
            vocab_table_tensor[fig_id, ans_type, qcat_id, -1] += 1
            # for this cat, update the correct regression qa counter:
            if is_reg:
                vocab_table_tensor[fig_id, -1, qcat_id, 0] += 1 if correct else 0
                vocab_table_tensor[fig_id, -1, qcat_id, 1] += 1 if t_correct else 0
                vocab_table_tensor[fig_id, -1, qcat_id, -1] += 1
                assert ans_type == 2

    if params['ddp']:
        dist.all_reduce(vocab_table_tensor,
                        op=torch.distributed.ReduceOp.SUM,
                        group=dist_g)
        breakdown_tensor += vocab_table_tensor


def reduce_total_acc(total_correct_tensor, needs_regression, nsp_right, reg_right, reg_t_right, dist_g):
    """ Calculates the total accuracy """
    assert reg_right.shape[0] == nsp_right.shape[0] == needs_regression.shape[0]
    correct_tensor = torch.zeros(total_correct_tensor.shape).cuda().double()
    # nsp acc
    correct_tensor[0, 0] = torch.sum(nsp_right)
    correct_tensor[0, 1] = nsp_right.shape[0]
    # reg_cls_acc
    correct_tensor[1, 0] = torch.sum(nsp_right & needs_regression)
    correct_tensor[1, 1] = torch.sum(needs_regression)
    # reg acc
    correct_tensor[2, 0] = torch.sum(reg_right)
    correct_tensor[2, 1] = torch.sum(needs_regression)
    # reg_t acc
    correct_tensor[3, 0] = torch.sum(reg_t_right)
    correct_tensor[3, 1] = torch.sum(needs_regression)
    # total +-5% acc
    correct_answers = (nsp_right & (needs_regression.logical_not() | reg_right))
    correct_tensor[4, 0] = torch.sum(correct_answers)
    correct_tensor[4, 1] = nsp_right.shape[0]
    # total t acc
    correct_answers_t_loss = (nsp_right & (needs_regression.logical_not() | reg_t_right))
    correct_tensor[5, 0] = torch.sum(correct_answers_t_loss)
    correct_tensor[5, 1] = nsp_right.shape[0]

    dist.all_reduce(correct_tensor,
                    op=torch.distributed.ReduceOp.SUM,
                    group=dist_g)

    total_correct_tensor += correct_tensor

    return total_correct_tensor


def reduce_histogram(histogram, reg_5_dist, dist_g):
    """ Reduces histogram information """
    correct_tensor = torch.zeros_like(histogram)
    # nsp acc
    bar_id = 0
    for i in range(4):
        correct_tensor[bar_id] = torch.sum(((i / 20) < reg_5_dist) & (reg_5_dist <= ((i + 1) / 20)))
        bar_id += 1
    for i in range(2, 10):
        correct_tensor[bar_id] = torch.sum(((i/10) < reg_5_dist) & (reg_5_dist <= ((i + 1)/10)))
        bar_id += 1

    correct_tensor[bar_id] = torch.sum(1 < reg_5_dist)

    dist.all_reduce(correct_tensor,
                    op=torch.distributed.ReduceOp.SUM,
                    group=dist_g)


    histogram += correct_tensor
    return histogram


def print_acc_table(log_print, acc_tensor):
    """ Prints accuracy table"""
    frac_lst = [[acc_tensor[3, 0] / acc_tensor[3, 1], acc_tensor[5, 0] / acc_tensor[5, 1]],
                [acc_tensor[2, 0] / acc_tensor[2, 1], acc_tensor[4, 0] / acc_tensor[4, 1]],
                ]
    frac_lst = [[('%.5g' % acc) for acc in row] for row in frac_lst]
    log_print(str(DataFrame(frac_lst,
                            ['Reg +-t', 'Reg +-5%'],
                            ['Accuracy', 'Total Accuracy']
                            )))
    log_print("-" * 10)
    nsp_lst = [acc_tensor[0, 0] / acc_tensor[0, 1], acc_tensor[1, 0] / acc_tensor[1, 1]]
    nsp_lst = [[('%.5g' % acc) for acc in nsp_lst]]
    log_print(str(DataFrame(nsp_lst, [''], ['nsp', 'reg_cls'])))
    log_print("-" * 20)


def print_breakdown_table(params, log_print, acc_tensor):
    """ Prints breakdown accuracy table"""
    qid_ind_map = {'Total': 0, 'line': 1, 'vbar': 2, 'hbar': 3, 'dot': 4}
    if params['dataset'] == 'dvqa':
        qid_ind_map = {'Total': 0}

    for tab_name, tab_id in qid_ind_map.items():
        frac_tensor = []
        for ans_cat in range(acc_tensor.shape[1]):
            ans_cat_lst = []
            for q_cat in range(acc_tensor.shape[2]):
                nominator = acc_tensor[tab_id, ans_cat, q_cat, 0]
                nominator_t = acc_tensor[tab_id, ans_cat, q_cat, 1]
                denominator = acc_tensor[tab_id, ans_cat, q_cat, -1]
                if denominator == 0:
                    ans_cat_lst.append("N/A")
                else:
                    string = f"{'%.5g' % (nominator / denominator)}"
                    if nominator_t != nominator:
                        string += f" | {'%.5g' % (nominator_t / denominator)}"
                    # string = f"{nominator} / {denominator}"
                    ans_cat_lst.append(string)
            frac_tensor.append(ans_cat_lst)

        frac_tensor.append([(torch.sum(acc_tensor[tab_id, :3, 0, 0]) / torch.sum(acc_tensor[tab_id, :3, 0, -1])).item(),
                            (torch.sum(acc_tensor[tab_id, :3, 1, 0]) / torch.sum(acc_tensor[tab_id, :3, 1, -1])).item(),
                            (torch.sum(acc_tensor[tab_id, :3, 2, 0]) / torch.sum(acc_tensor[tab_id, :3, 2, -1])).item()
                            ])
        log_print("==================== {}: {} =======================".format(tab_name, torch.sum(
            acc_tensor[tab_id, :3, :, -1]).item()))
        # print_by_rank(correct_by_vocab_tensor[value])
        log_print(str(DataFrame(frac_tensor,
                                ['Yes/No', 'Fixed Vocabulary', 'Open Vocabulary', 'Regression', 'Total'],
                                ['Structural', 'Data Retrieval', 'Reasoning']
                                )))
    log_print("-" * 20)


if __name__ == "__main__":
    params = options.read_command_line()
    pprint.pprint(params)

    if params['ddp']:
        mp.spawn(evaluate_plotqa, nprocs=params['num_proc'], args=(params,))
    else:
        evaluate_plotqa("cuda", params)

