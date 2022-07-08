import time
import os
import torch
import numpy as np
from .vilbert import BertForMultiModalPreTraining, BertConfig
from torch.autograd import Variable


class VisualDialogEncoder(torch.nn.Module):

    def __init__(self, params):
        super(VisualDialogEncoder, self).__init__()
        config_path = params['model_config']
        assert os.path.exists(config_path), "model_config file not found"
        config = BertConfig.from_json_file(config_path)
        self.bert_pretrained = BertForMultiModalPreTraining.from_pretrained('bert-base-uncased', config, params=params)
        self.bert_pretrained.train()

    def forward(self, input_ids, txt_loc, image_feat, image_loc, sep_indices=None, sep_len=None, token_type_ids=None,
         attention_mask=None, masked_lm_labels=None, next_sentence_label=None, head_mask=None,random_round_indices=None,
         output_nsp_scores=False, output_lm_scores=False, image_attention_mask=None,image_label=None, image_target=None,
         gt_reg=None, areas=None, legend_pred=None):

        masked_lm_loss = None
        masked_img_loss = None
        nsp_loss = None
        prediction_scores_t = None
        seq_relationship_score = None
        reg_loss = None
        start_time = time.time()
        if next_sentence_label is not None and masked_lm_labels \
            is not None and image_target is not None:
            # train mode, output losses
            masked_lm_loss, masked_img_loss, nsp_loss, _, prediction_scores_t, seq_relationship_score, reg_loss, legend_loss = \
                self.bert_pretrained(input_ids, txt_loc, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                 token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                            next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                                image_label=image_label, image_target=image_target,
                                     gt_reg=gt_reg, areas=areas, legend_pred=legend_pred)
        else:
            #inference, output scores
            prediction_scores_t, _, seq_relationship_score, _, _, reg_loss, legend_loss = \
                self.bert_pretrained(input_ids, txt_loc, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                    token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                    next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                    image_label=image_label, image_target=image_target, gt_reg=gt_reg, areas=areas, legend_pred=legend_pred)
        # print(">>> time:", round(time.time() - start_time, 2))
        out = (masked_lm_loss, masked_img_loss, nsp_loss)

        # if output_nsp_scores:
        out = out + (seq_relationship_score,)
        if output_lm_scores:
            out = out + (prediction_scores_t,)
        return out + (reg_loss, legend_loss)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    if sequence_length.is_cuda:
        seq_length_expand = seq_length_expand.cuda()
    return seq_range_expand < seq_length_expand


def forward(dialog_encoder, batch, params, output_nsp_scores=False, output_lm_scores=False,
            evaluation=False, sample_ids=None):

    if sample_ids is None:
        sample_indices = np.arange(batch['tokens'].shape[0])
    else:
        sample_indices = sample_ids

    tokens = batch['tokens'][sample_indices, :].to(params['device'])
    txt_loc = batch['loc'][sample_indices, :].to(params['device'])
    segments = batch['segments'][sample_indices, :].to(params['device'])
    sep_indices = batch['sep_indices'][sample_indices, :]
    mask = batch['mask'][sample_indices, :]
    hist_len = batch['hist_len'][sample_indices]

    features = batch['image_feat'][sample_indices, :, :]
    image_loc = batch['image_loc'][sample_indices, :, :]
    image_mask = batch['image_mask'][sample_indices, :]
    regression_target = batch['R'][sample_indices].to(params['device'])

    image_areas = None
    if 'areas' in batch:
        image_areas = batch['areas'][sample_indices, :]
        image_areas = image_areas.to(params['device'])

    next_sentence_labels = None
    image_label = None

    if not evaluation:
        next_sentence_labels = batch['next_sentence_labels'][sample_indices].to(params['device'])
        image_label = batch['image_label'][sample_indices, ...].to(params['device'])
        regression_target = [regression_target, 'L1_smooth']
    else:
        regression_target = [regression_target, 'L1']

    image_target = batch['image_target'][sample_indices, ...].to(params['device'])
    if evaluation:
        sep_indices = sep_indices.to(params['device'])
        mask = mask.to(params['device'])
        hist_len = hist_len.to(params['device'])
        image_mask = image_mask.to(params['device'])
    # -----
    features = features.to(params['device'])
    image_loc = image_loc.to(params['device'])

    sequence_lengths = torch.gather(sep_indices, 1, hist_len.view(-1, 1)) + 1
    sequence_lengths = sequence_lengths.squeeze(1)
    attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
    loss = None

    sep_len = hist_len + 1

    lm_loss, img_loss, nsp_loss, nsp_scores, regression, legend_loss = dialog_encoder(tokens,
                                                                                      txt_loc,
                                                                                      features,
                                                                                      image_loc,
                                                                                      sep_indices=sep_indices,
                                                                                      sep_len=sep_len,
                                                                                      token_type_ids=segments,
                                                                                      masked_lm_labels=mask,
                                                                                      attention_mask=attention_mask_lm_nsp,
                                                                                      next_sentence_label=next_sentence_labels,
                                                                                      output_nsp_scores=output_nsp_scores,
                                                                                      output_lm_scores=output_lm_scores,
                                                                                      image_attention_mask=image_mask,
                                                                                      image_label=image_label,
                                                                                      image_target=image_target,
                                                                                      gt_reg=regression_target,
                                                                                      areas=image_areas,
                                                                                      )

    reg_output, reg_loss, reg_right = regression[0], regression[1].mean(), regression[3]

    if not evaluation:
        loss = (params['nsp_loss_coeff'] * nsp_loss) + (params['reg_loss_coeff'] * reg_loss)
        # for loss_type in [('lm_loss_coeff', lm_loss), ('img_loss_coeff', img_loss), ('legend_loss_coeff', legend_loss)]:
        #     if params[loss_type[0]] > 0:
        #         loss += params[loss_type[0]] * loss_type[1]

        # loss = (params['nsp_loss_coeff'] * nsp_loss) + (params['lm_loss_coeff'] * lm_loss)
        loss = loss.sum()

    if evaluation:
        return loss, lm_loss, nsp_loss, img_loss, nsp_scores, regression
    else:  # train
        return loss, lm_loss, nsp_loss, img_loss, nsp_scores, regression, legend_loss

