from typing import List, Dict, Tuple, Optional, Union
import os
import argparse
import sys
# sys.path.insert(0, './CRCT/model/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.encoder_decorator import VisualDialogEncoder
from fig_dataloader import PlotQA_Dataset
import json
from train import forward
from options import read_command_line
from copy import deepcopy
import numpy as np
from colorama import init
from colorama import Fore as FC
from colorama import Back as B
from colorama import Style


def color(text, fore='', back=''):
    return f'{fore}{back}{text}{Style.RESET_ALL}'


class PlotQA_Bot():
    def __init__(self, split='test'):
        """
                Loading all the files required for the model, including weights, features, QA, etc...
                3 possible sets of images:  {train, val, test}.

                if cuda_num == -1, model will use all CUDA_VISIBLE_DEVICES
                """
        self.params = read_command_line()
        # pprint.pprint(params)

        self.dataset = PlotQA_Dataset(self.params, split, split)

        self.plotqa_encoder = Model(self.params)
        self.plotqa_encoder.eval()

        self.img_to_qas = self.get_images_and_questions()

    def get_images_and_questions(self) -> Dict:
        question_dict = {}
        for i, qa in enumerate(self.dataset.qa[self.dataset.split]):
            if qa['image_index'] not in question_dict:
                question_dict[qa['image_index']] = []
            question_dict[qa['image_index']].append(qa)

        return question_dict

    def get_answer_lst(self, top_k_ids, possible_answers, regression_output):
        answer_lst = []
        for ans_id, probability in top_k_ids:
            answer = possible_answers[ans_id]
            answer = answer[0] if type(answer) is tuple else answer
            if answer == self.dataset.R:
                answer = regression_output
            answer_lst.append((answer, probability))
        return answer_lst

    def loop(self):
        while True:
            image_id = int(input(color('Type the figure id in the {} folder:\n'.format(self.dataset.split), FC.GREEN, B.RESET)))
            q_list = []
            for i, qa in enumerate(self.img_to_qas[image_id]):
                print(color('<{}>: {}'.format(i + 1, qa['question_string']), FC.CYAN, B.RESET))

            if len(self.img_to_qas[image_id]) == 0:
                print(color("Nothing to ask about this image, try another one :(", FC.RED, B.RESET))
                continue

            keep_asking = True
            while keep_asking:
                usr_input = input(color('Choose a question id or type a question: \n\r', FC.GREEN, B.RESET))
                try:
                    q_id = int(usr_input) - 1
                    qa_pair = self.img_to_qas[image_id][q_id]
                    print(color(qa_pair['question_string'], FC.GREEN, B.RESET))
                except:
                    qa_pair = {'question_string': str(usr_input), 'image_index': image_id,
                               'answer': None, 'qid': None, 'type': 'dot'
                               }

                pred_dict = self.plotqa_encoder(qa_pair=qa_pair)
                prediction = pred_dict['reg_output'] if pred_dict['is_reg'] else pred_dict['cls_output']
                # if (self.params['dataset'] == 'dvqa' and str(pred_dict['cls_output']) == self.dataset.R) or (self.params['dataset'] != 'dvqa' and qa_pair['qid'] is None):
                if self.params['dataset'] != 'dvqa' and qa_pair['qid'] is None:
                    # print(qa_pair)
                    if str(pred_dict['cls_output']) == self.dataset.R:
                        print('<CRCT>:  ' + str(pred_dict['reg_output']))
                    else:
                        print('<CRCT>:  ' + str(pred_dict['cls_output']))
                else:
                    print('<GT>:  ' + str(qa_pair['answer']))
                    if pred_dict['is_correct']:
                        pred_str ='<CRCT>: ' + color(f"{prediction}", FC.CYAN, B.RESET)
                    else:
                        pred_str = '<CRCT>: ' + color(f"{prediction}", FC.RED, B.RESET)

                    if pred_dict['is_reg'] and pred_dict['nsp_right']:
                        sign = '+' if pred_dict['reg_output'] > float(qa_pair['answer']) else '-'
                        pred_str += ". (Error: " + color(f"{sign}{round(pred_dict['reg_loss'] * 100, 2)}%)", FC.RED, B.RESET)

                    print(pred_str)
                print("-"*10, "[Answers probabilities]", "-"*10)
                print(pred_dict['all_answers'])
                print("--------------")
                # keep_asking = input(color("Would you like to ask another question? (Y / N)\n", FC.GREEN, B.RESET))
                # keep_asking = True if keep_asking.lower() == 'y' else False


class Model(nn.Module):
    def __init__(self, params):
        gpu = 0
        super(Model, self).__init__()
        torch.cuda.set_device(gpu)
        params['rank'] = gpu + params['rank_from']
        # print(f"({params['rank']}) is waiting for all DDP proccess...")

        params['device'] = device = torch.device(("cuda:" + str(gpu)) if (torch.cuda.is_available() and params['cuda_num'] > -1) else "cpu")

        dialog_encoder = VisualDialogEncoder(params)
        dialog_encoder.to(device)
        if params['start_checkpoint']:
            pretrained_dict = torch.load(params['start_checkpoint'], map_location=params['device'])
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

        ##
        # print("Model's parameters:", sum(p.numel() for p in dialog_encoder.parameters()))
        self.dialog_encoder = dialog_encoder
        self.params = params

    def get_batch(self, qa_pair, fig_feat=None):
        if type(qa_pair) is dict:
            if fig_feat is None:
                img_id = qa_pair['image_index']
                fig_feat = deepcopy(bot.dataset.get_fig_feat(img_id))
            batch = bot.dataset.get_encoded_qa(fig_feat, qa_pair, False, -1)
        else:
            batch = bot.dataset[qa_pair]

        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].unsqueeze(0)
        bot.dataset.cut_batch_padding(batch)

        return batch

    def forward(self, qa_pair, fig_feats=None):
        batch = self.get_batch(qa_pair, fig_feats)
        output = []
        reg_output = []
        # accuracy by 5% measure
        reg_loss_lst = []
        # accuracy by tick measure loss
        reg_t_loss_lst = []
        batch_size = self.params['batch_size']
        with torch.no_grad():
            for j in range(int(np.ceil(batch['tokens'].shape[0] / batch_size))):
                # create chunks of the original batch
                chunk = j * batch_size, min((j + 1) * batch_size, batch['tokens'].shape[0])
                _, _, _, _, nsp_scores, regression = forward(self.dialog_encoder,
                                                             batch,
                                                             self.params,
                                                             output_nsp_scores=True,
                                                             evaluation=True,
                                                             sample_ids=np.arange(chunk[0], chunk[1])
                                                             )

                # normalize nsp scores
                nsp_probs = F.softmax(nsp_scores, dim=1)

                assert nsp_probs.shape[-1] == 2
                output.append(nsp_probs[:, 0])
                reg_output.append(regression[0])
                reg_loss_lst.append(regression[4])
                reg_t_loss_lst.append(regression[2])
                assert regression[0].shape[0] == regression[1].shape[0] == regression[2].shape[0] == nsp_probs.shape[0]

            del regression
            del nsp_probs
            output = torch.cat(output, dim=0)
            reg_output = torch.cat(reg_output, dim=0)
            reg_loss_lst = torch.cat(reg_loss_lst, dim=0)
            reg_t_loss_lst = torch.cat(reg_t_loss_lst, dim=0)
            assert batch['tokens'].shape[0] == output.shape[0] == reg_output.shape[0] == reg_loss_lst.shape[0] == \
                   reg_t_loss_lst.shape[0]

            total_options = 0
            answers = []
            answers_probs = []
            reg_answers_loss = []
            reg_answers_t_loss = []
            reg_answers_output = []

            for i, n in enumerate(batch['num_ans']):
                answers_certainty = F.softmax(output[total_options: total_options + n].view(-1), dim=-1)
                ans_id = torch.argmax(output[total_options: total_options + n])
                answers.append(ans_id)
                # answers_probs.append(output[total_options: total_options + n].cpu())
                # sents.append(batch['tokens'][total_options: total_options + n])
                reg_answers_loss.append(reg_loss_lst[total_options: total_options + n][ans_id.item()])
                reg_answers_t_loss.append(reg_t_loss_lst[total_options: total_options + n][ans_id.item()])
                reg_answers_output.append(reg_output[total_options: total_options + n][ans_id.item()])
                total_options += n

            assert total_options == batch['tokens'].shape[0]
            answers = torch.stack(answers, dim=0).cpu()
            # answers_probs = pad_probs(answers_probs)

            reg_answers_loss = torch.stack(reg_answers_loss, dim=0).cpu()
            reg_answers_t_loss = torch.stack(reg_answers_t_loss, dim=0).cpu()
            reg_answers_output = torch.stack(reg_answers_output, dim=0).cpu()

            assert answers.shape[0] == reg_answers_loss.shape[0] == reg_answers_t_loss.shape[0] == \
                   reg_answers_output.shape[0] == batch['id'].shape[0], answers_probs.shape[0]

            nsp_right = (answers == batch['gt_id'].view(-1))
            needs_regression = batch['needs_reg'].view(-1)
            reg_right = ((reg_answers_loss <= 0.05) & needs_regression)
            correct_answers = (nsp_right & (needs_regression.logical_not() | reg_right))

        answer_options = bot.dataset.get_possible_answers(qa_pair['image_index'], fig_feats)

        all_answers = sorted(list(zip([p for p in answers_certainty.cpu().tolist()], answer_options)), key=lambda x: -x[0])

        pred_dict = {'nsp_right': nsp_right.item(),
                     'is_correct': correct_answers.item(),
                     'is_reg': needs_regression.item(),
                     'reg_loss': reg_answers_loss.item(),
                     'reg_output': reg_answers_output.item(),
                     'cls_output': answer_options[answers.item()],
                     'all_answers': all_answers
                     }
        return pred_dict


if __name__ == "__main__":
    sys.argv = ['Interactive_demo.py', '-qa_file', 'qa_pairs_test.npy', '-eval_batch_size', '100',
                '-num_workers', '0', '-ddp', '-world_size', '1', '-num_proc', '1',
                '-save_name', 'temp', '-dataset_config', 'config/plotqa.json',
                '-eval_set', 'test', '-start_checkpoint', 'crct.ckpt',
                '-BOT_MODE']

    print(f"Loading CRCT model and data... [{sys.argv[15]}]")
    bot = PlotQA_Bot(split=sys.argv[17])  # split='test'

    bot.loop()

