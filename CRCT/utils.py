import os
import torch
import random 
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import json
import pandas as pd
from time import gmtime, strftime


class WarmupLinearScheduleNonZero(_LRScheduler):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to max_lr over `warmup_steps` training steps.
        Linearly decreases learning rate linearly to min_lr over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, min_lr=1.3e-5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_lr = min_lr
        super(WarmupLinearScheduleNonZero, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr_factor = float(step) / float(max(1, self.warmup_steps))
        else:
            lr_factor = max(0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

        return [base_lr * lr_factor if (base_lr * lr_factor) > self.min_lr else self.min_lr for base_lr in self.base_lrs]


def init_log_file(params):
    # log_line(params, str(params) + "\n\n")
    os.makedirs(params['save_path'], exist_ok=True)
    params['log_file'] = params['save_path'] + "/" + strftime('%d-%b-%y-%X-%a', gmtime()) + ".txt"
    if params['rank'] == 0:
        with open(params['log_file'], 'w') as file:
            file.write(str(params).replace(",", "\n"))
            file.write("\n\n ============= Details ========== \n" + params['details'])


def log_line(params, line, all_ranks=False):
    if params['rank'] == 0 or all_ranks:
        if params['log_file']:
            with open(params['log_file'], 'a') as file:
                file.write(line + "\n")
        print(line, flush=True)


def list2tensorpad(inp_list, max_seq_len):
    inp_tensor = torch.LongTensor([inp_list])
    inp_tensor_zeros = torch.zeros(1, max_seq_len, dtype=torch.long)
    # print(inp_tensor.shape[1])
    inp_tensor_zeros[0, :inp_tensor.shape[1]] = inp_tensor[0, :max_seq_len]
    inp_tensor = inp_tensor_zeros
    return inp_tensor    

def encode_input(utterances, start_segment, CLS, SEP, MASK, max_seq_len=256,max_sep_len=25,mask_prob=0.2):

    cur_segment = start_segment
    token_id_list = []
    segment_id_list = []
    sep_token_indices = []
    masked_token_list = []

    token_id_list.append(CLS)
    segment_id_list.append(cur_segment)
    masked_token_list.append(0)

    cur_sep_token_index = 0
    
    for cur_utterance in utterances:
        # add the masked token and keep track
        cur_masked_index = [1 if random.random() < mask_prob else 0 for _ in range(len(cur_utterance))]
        masked_token_list.extend(cur_masked_index)
        token_id_list.extend(cur_utterance)
        segment_id_list.extend([cur_segment]*len(cur_utterance))

        token_id_list.append(SEP)
        segment_id_list.append(cur_segment)
        masked_token_list.append(0)
        cur_sep_token_index = cur_sep_token_index + len(cur_utterance) + 1
        sep_token_indices.append(cur_sep_token_index)            
        cur_segment = cur_segment ^ 1  # cur segment osciallates between 0 and 1
    
    assert len(segment_id_list) == len(token_id_list) == len(masked_token_list) == sep_token_indices[-1] + 1 
    # convert to tensors and pad to maximum seq length
    tokens = list2tensorpad(token_id_list,max_seq_len)
    masked_tokens = list2tensorpad(masked_token_list,max_seq_len)
    masked_tokens[0,masked_tokens[0,:]==0] = -1
    mask = masked_tokens[0,:]==1
    masked_tokens[0,mask] = tokens[0,mask]
    tokens[0,mask] = MASK

    # print("mask", mask)
    # print("tokens", tokens)
    # print("masked tokens", masked_tokens)
    # print("num mask tokens", torch.sum(mask))

    segment_id_list = list2tensorpad(segment_id_list,max_seq_len)
    # segment_id_list += 2 
    return tokens, segment_id_list, list2tensorpad(sep_token_indices,max_sep_len),masked_tokens


def encode_text_input(utterances, locations, token_types, CLS, SEP, MASK, max_seq_len=256, max_sep_len=50, mask_prob=0.2):
    token_id_list = []
    segment_id_list = []
    tokens_loc = []

    sep_token_indices = []
    masked_token_list = []

    # <cls> 0
    token_id_list.append(CLS)
    segment_id_list.append(0)
    tokens_loc.append([0, 0, 0, 0])
    masked_token_list.append(0)
    #
    # # <cls> 1
    # token_id_list.append(CLS)
    # segment_id_list.append(0)
    # tokens_loc.append([0, 0, 0, 0])
    # masked_token_list.append(0)
    #
    cur_sep_token_index = 0
    for i, (cur_utterance, cur_loc, cur_segment) in enumerate(zip(utterances, locations, token_types)):
        # add the masked token and keep track
        # mask prob only if the text is question (tok type -1)
        cur_masked_index = [1 if (random.random() < mask_prob and (cur_segment == -1)) else 0 for _ in range(len(cur_utterance))]
        masked_token_list.extend(cur_masked_index)
        token_id_list.extend(cur_utterance)
        segment_id_list.extend([cur_segment] * len(cur_utterance))
        if type(cur_loc[0]) != list:
            tokens_loc.extend([cur_loc] * len(cur_utterance))
        else:
            tokens_loc.extend(cur_loc)

        token_id_list.append(SEP)
        segment_id_list.append(cur_segment)
        if type(cur_loc[0]) != list:
            tokens_loc.append(cur_loc)
        else:
            tokens_loc.append(cur_loc[0])

        masked_token_list.append(0)
        cur_sep_token_index += len(cur_utterance) + 1
        sep_token_indices.append(cur_sep_token_index)

    assert len(segment_id_list) == len(tokens_loc) == len(token_id_list) == len(masked_token_list) == sep_token_indices[-1] + 1
    # convert to tensors and pad to maximum seq length

    tokens = list2tensorpad(token_id_list, max_seq_len)
    masked_tokens = list2tensorpad(masked_token_list, max_seq_len)
    masked_tokens[0, masked_tokens[0, :] == 0] = -1
    mask = masked_tokens[0, :] == 1
    masked_tokens[0, mask] = tokens[0, mask]
    tokens[0, mask] = MASK

    segment_id_list = list2tensorpad(segment_id_list, max_seq_len)
    padded_locs = torch.zeros((1, max_seq_len, 4), dtype=torch.float32)
    padded_legend_belonging = torch.zeros((1, max_seq_len, 1), dtype=torch.int32)
    for i in range(len(tokens_loc)):
        if len(tokens_loc[i]) > 4:
            # save beloning and cut
            padded_legend_belonging[0][i][0] = int(tokens_loc[i][4])
            tokens_loc[i] = tokens_loc[i][:4]

    padded_locs[0, :len(tokens_loc), :] = torch.tensor(np.array(tokens_loc))[:max_seq_len, :]

    # segment_id_list += 2
    return tokens, segment_id_list, list2tensorpad(sep_token_indices, max_sep_len), padded_locs, masked_tokens, padded_legend_belonging


def encode_image_input(features, legend_belonging, boxes, image_target, max_regions=37, mask_prob=0.15):
    output_label = []
    num_boxes = min(int(len(boxes)), max_regions)

    mix_boxes_pad = np.zeros((max_regions, boxes.shape[-1]))
    mix_features_pad = np.zeros((max_regions, features.shape[-1]))
    mix_image_target = np.zeros((max_regions, image_target.shape[-1]))
    mix_legend_belonging = np.zeros(max_regions)

    mix_boxes_pad[:num_boxes] = boxes[:num_boxes]
    mix_features_pad[:num_boxes] = features[:num_boxes]
    mix_image_target[:num_boxes] = image_target[:num_boxes]
    if legend_belonging is not None:
        mix_legend_belonging[:num_boxes] = legend_belonging[:num_boxes]

    boxes = mix_boxes_pad
    features = mix_features_pad
    image_target = mix_image_target
    legend_belonging = mix_legend_belonging

    for i in range(num_boxes):
        prob = random.random()
        # prob = 0
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.9:
                features[i] = 0
            output_label.append(1)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    image_mask = [1] * (int(num_boxes))
    while len(image_mask) < max_regions:
        image_mask.append(0)
        output_label.append(-1)
    
    # ensure we have atleast one region being predicted
    output_label[random.randint(1, len(output_label)-1)] = 1
    image_label = torch.LongTensor(output_label)
    image_label[0] = 0  # make sure the <IMG> token doesn't contribute to the masked loss
    image_mask = torch.tensor(image_mask).float()

    features = torch.tensor(features).float()
    spatials = torch.tensor(boxes).float()
    image_target = torch.tensor(image_target).long()
    legend_belonging = torch.tensor(legend_belonging).long()

    return features, spatials, image_mask, image_target.view(-1), image_label, legend_belonging


def get_optimizer(params, dialog_encoder):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    with open('config/language_weights.json') as f:
        langauge_weights = json.load(f)

    # for_gradient_flow = []
    optimizer_grouped_parameters = []
    for key, value in dict(dialog_encoder.named_parameters()).items():
        if value.requires_grad:
            if key in langauge_weights:
                lr = params['lr']
            else:
                lr = params['image_lr']


            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0}]
            else:
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": params['wd']}]

    return torch.optim.AdamW(optimizer_grouped_parameters, lr=params['lr'])


class Evaluation_Log:
    def __init__(self, params, columns, cont_eval):
        self.columns = columns
        self.fp = f"{params['save_path']}/eval_results_{params['eval_set']}_{params['start_checkpoint'].split('/')[-1]}_{params['rank']}.csv"
        if not cont_eval:
            table = pd.DataFrame(columns=self.columns)
            table.to_csv(self.fp, mode='w')
        # tensors paths
        self.breakdown_path = f"{params['save_path']}/eval_results_{params['eval_set']}_{params['start_checkpoint'].split('/')[-1]}_breakdown.npy"
        self.total_correct_path = f"{params['save_path']}/eval_results_{params['eval_set']}_{params['start_checkpoint'].split('/')[-1]}_total_correct.npy"
        self.hostogram_path = f"{params['save_path']}/eval_results_{params['eval_set']}_{params['start_checkpoint'].split('/')[-1]}_histogram.npy"
        self.params = params

    def append(self, data):
        table = pd.DataFrame(data=data, columns=self.columns)
        table.to_csv(self.fp, mode='a', header=False)

    def save_tensors(self, breakdown_tensor, total_correct_tensor, histogram_tensor):
        if self.params['rank'] == 0:
            if breakdown_tensor is not None:
                np.save(self.breakdown_path, breakdown_tensor.cpu().numpy(), allow_pickle=True)
            if total_correct_tensor is not None:
                np.save(self.total_correct_path, total_correct_tensor.cpu().numpy(), allow_pickle=True)
            if histogram_tensor is not None:
                np.save(self.hostogram_path, histogram_tensor.cpu().numpy(), allow_pickle=True)

    def load_tensors(self):
        return torch.tensor(np.load(self.breakdown_path, allow_pickle=True)).cuda(), \
               torch.tensor(np.load(self.total_correct_path, allow_pickle=True)).cuda(), \
               torch.tensor(np.load(self.hostogram_path, allow_pickle=True)).cuda()
