import glob
import re
import torch.nn.functional as F
import torch
from torch.utils import data
import json
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
from utils import encode_text_input, encode_image_input
from copy import deepcopy


class PlotQA_Dataset(data.Dataset):

    def __init__(self, params, splits_to_load=None, init_split='train'):
        self.subsets = ['train', 'val', 'test']
        self.fig_feats = dict()
        self.qa = dict()
        self.params = params
        self.fig_classes = ['bar', 'dot_line', 'legend_label', 'line', 'preview', 'title',
                            'xlabel', 'xticklabel', 'ylabel', 'yticklabel', 'x_axis', 'y_axis']
        self.token_types = ['Q', 'A'] + self.fig_classes

        self.REGRESSIONS = ['D14', 'D15', 'M5', 'M4', 'CD6', 'CD7', 'M1', 'CD1', 'CD3', 'A1', 'A0', 'A3', 'A2', 'A5',
                            'A4', 'A7', 'A6', 'A8', 'A9', 'C5', 'C2', 'D7', 'M0']

        self.PADDING_TXT = ['tokens', 'segments', 'sep_indices', 'mask',
                            'next_sentence_labels', 'hist_len', 'loc', 'legend_belonging_t']

        self.PADDING_VIS = ['image_feat', 'image_loc', 'image_mask', 'image_target', 'image_label',
                            'legend_belonging_v',
                            'R', ]
        self.val_color_mapping = Color_Mapping()

        if params['dataset'] == 'dvqa':
            self.fixed_vocab = ['yes', 'no', 'zero', 'two', 'three', 'one', 'four', 'five', 'six', 'seven',
                                'eight', 'nine']

            self.dvqa_floats = [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                         12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                         31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 43.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 1000.0,
                         10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0]
        else:
            # for all structural:
            self.fixed_vocab = [2, 'Yes', 'No', 'vertical', 5, 'center right', 4, 'horizontal', 'bottom right', 7, 6,
                                'bottom center', 'bottom left', 0, 8, 3, 1, 'top right', 12, 10, 9, 11, 18, 14, 15, 13,
                                17, 16, 20, 24, 19, 23, 22, 21]
        self.R = "="
        self.fixed_vocab.append(self.R)
        self.fixed_vocab = [str(p) for p in self.fixed_vocab]
        self.fixed_vocab_lower = [str(p).lower() for p in self.fixed_vocab]

        self.POS = 0
        self.NEG = 1
        self.IMG_TOKEN_FEATURES_CLASS = 1000
        self.get_all_answers = False
        self._split = init_split
        # ---- figures features + qa_pairs
        self.print_by_rank("Loading qa_pairs...")
        if splits_to_load is None:
            splits_to_load = ['train', params['eval_set']]
        self.load_files(splits_to_load)

        self.print_by_rank("Done.")
        # ------

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
        # fetching token indicecs of [CLS] and [SEP]
        tokens = ['[CLS]', '[MASK]', '[SEP]']
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        self.CLS = indexed_tokens[0]
        self.MASK = indexed_tokens[1]
        self.SEP = indexed_tokens[2]
        self._max_region_num = params['max_vis_features']
        self.EVAL_PADDED_SIZE = 120
        self.type_to_qid = {'vbar': [], 'hbar': [], 'dot': [], 'line': []}
        self.type_to_img_id = {'vbar': {}, 'hbar': {}, 'dot': {}, 'line': {}}
        self.img_id_to_qa = {}

    def print_by_rank(self, msg, all_rank=False):
        if all_rank or self.params['rank'] == 0 and not self.params['BOT_MODE']:
            print(msg)

    def load_files(self, splits):
        if type(splits) is not list:
            splits = [splits]
        for split in splits:
            self.print_by_rank("Loading " + split + " figure features...")
            npys_path = self.params['figure_feat_path'] + self.split_path(split) + "/*.npy"
            files = glob.glob(npys_path)
            files = sorted(files, key=lambda x: float(re.findall(r"(\d+)", x)[-1]))
            assert len(files) > 0, npys_path
            self.fig_feats[split] = dict()
            for i, file in enumerate(files):
                self.fig_feats[split][i] = file
            qa_file_path = self.params['qa_parent_dir'] + self.split_path(split) + "/" + self.params['qa_file']
            if self.params['qa_file'].split('.')[-1] == 'npy':
                self.qa[split] = np.load(qa_file_path, allow_pickle=True)
            elif self.params['qa_file'].split('.')[-1] == 'json':
                with open(qa_file_path) as file:
                    self.qa[split] = json.load(file)
                    if 'qa_pairs' in self.qa[split]:
                        self.qa[split] = self.qa[split]['qa_pairs']

    def get_qa(self, split, idx):
        orig_len = self.orig_len()
        if self.split == 'train' and idx >= orig_len:
            return self.qa[split][idx - orig_len]
        return self.qa[split][idx]

    def __len__(self):
        multiplier = 2 if (self.split == 'train' and not self.params['binary_answers']) else 1
        return self.orig_len() * multiplier

    def orig_len(self):
        return len(self.qa[self._split])

    def split_path(self, split):
        if split == 'train':
            return self.params['splits'][0]
        elif split == 'val':
            return self.params['splits'][1]
        elif split == 'test':
            return self.params['splits'][2]
        elif split == 'test1':
            return self.params['splits'][3]
        elif split == 'test2':
            return self.params['splits'][4]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets
        self._split = split

    def get_division(self):
        return self.params['dataset_config']['dataset_files_divisions'][self._split]

    def get_loaded_fig_file(self, image_id):
        """ Get the exact data file which contains image_id data """
        fig_file_id = image_id // self.get_division()
        fig_data = self.fig_feats[self._split][fig_file_id]
        if type(fig_data) is str:
            self.fig_feats[self._split][fig_file_id] = np.load(fig_data, allow_pickle=True)

        return self.fig_feats[self._split][fig_file_id]

    def get_fig_feat(self, image_id):
        image_index = image_id if self.params['dataset'] != 'dvqa' else (image_id - 1)
        fig_feat = self.get_loaded_fig_file(image_index)[image_index % self.get_division()]
        assert fig_feat['image_id'] == image_id, str((image_id, fig_feat['image_id'], self.get_division()))
        return fig_feat

    def get_token_type(self, token_type):
        """ Get the id of a given class (according to the detector) """
        # To not confuse with padding, we denote 'Q' (question) token type as -1. The network will handle that
        return self.token_types.index(token_type) if token_type != 'Q' else -1

    def get_fig_caption(self, text_feat, is_hbar=False, qa_pair=None):
        """ Get the first information in text"""
        caption = []
        possible_answers = []
        ticks_values = {'x_axis': [], 'y_axis': []}
        tot_len = 0
        if self.params['dataset'] != 'figure_qa' and 'title' in text_feat:
            # --- title
            if type(text_feat['title']) is dict:
                title_txt = text_feat['title']['text']
                title_loc = text_feat['title']['bbox']
            else:
                assert False, "Title location"
                # title_txt = text_feat['title']
                # title_loc = [0, 0, 0, 0]
            title = self.tokenizer.encode(title_txt)
            caption.append((title, title_loc, self.get_token_type('title')))
            tot_len += len(title) + 2  # add a 1 for the CLS token as well as the sep tokens which follows the caption
            if self.params['dataset'] != 'dvqa':
                possible_answers.append((title_txt, None))
        # -- axes:
        for ax in ['x_axis', 'y_axis']:
            if ax not in text_feat:
                continue
            if self.params['dataset'] != 'figure_qa':
                # axis label:
                axis_label_loc = [0.5, 0, 0.5, 0] if (
                            (ax == 'y_axis' and is_hbar) or (ax == 'x_axis' and not is_hbar)) else [0, 0.5, 0, 0.5]
                if len(text_feat[ax]['label']) > 0:
                    possible_answers.append((text_feat[ax]['label'], None))
                    axis_label = self.tokenizer.encode(text_feat[ax]['label'])
                    caption.append((axis_label, axis_label_loc, self.get_token_type(ax[0] + "label")))
                    tot_len += len(axis_label) + 1  # + SEP
            # ticks:
            for t, l in text_feat[ax]['ticks']:
                if l > 0:
                    try:
                        ticks_values[ax].append((float(t), float(l)))
                    except:
                        pass
                # possible_answers.append(str(l))
                tick_label = self.tokenizer.encode(t)
                if self.params['dataset'] == 'dvqa':
                    orientation = (ax == 'y_axis' and not text_feat['values_are_x']) or (
                                ax == 'x_axis' and text_feat['values_are_x'])
                    tick_label_loc = [0, l, 0, l] if orientation else [l, 0, l, 0]
                else:
                    tick_label_loc = [l, 0, l, 0] if (
                                (ax == 'y_axis' and is_hbar) or (ax == 'x_axis' and not is_hbar)) else [0, l, 0, l]

                if ax == 'x_axis' or '_cls' in self.params['qa_file']:
                    #
                    possible_answers.append((t, tick_label_loc))

                caption.append((tick_label, tick_label_loc, self.get_token_type(ax[0] + 'ticklabel')))
                tot_len += len(tick_label) + 1  # + SEP
        # -- legend:
        legend_pred = -torch.ones(6, 2).long()
        if 'legend' in text_feat:
            for i in range(len(text_feat['legend']['label'])):
                legend_label = self.tokenizer.encode(text_feat['legend']['label'][i])
                legend_label_loc = text_feat['legend']['bbox'][i]
                possible_answers.append((text_feat['legend']['label'][i], legend_label_loc))
                caption.append((legend_label, legend_label_loc, self.get_token_type('legend_label')))
                tot_len += len(legend_label) + 1  # + SEP

        # == So far:  [title, xlabel, x_1,...,x_k, ylabel, y_1, .., y_l, legend_1, ..., legend_r]
        return caption, tot_len, possible_answers, ticks_values, legend_pred

    def tokens2str(self, seq):
        dialog_sequence = ''
        for word in seq.numpy():
            dialog_sequence += self.tokenizer._convert_id_to_token(word) + " "
        dialog_sequence = dialog_sequence.encode('utf8')
        return dialog_sequence

    def get_right_answer(self, caption, tot_len, qa_pair, possible_answers):
        # ============ right answer ====================
        # check if need regression:
        cur_rnd_utterance = caption.copy()
        if str(qa_pair['answer']) not in possible_answers:
            # answer is <r> special token (")
            tokenized_answer = self.tokenizer.encode(self.R)
        else:
            # fixed vocabulary
            tokenized_answer = self.tokenizer.encode(str(qa_pair['answer']))

        # assert tot_len + len(tokenized_answer) + 1 < self.params['max_seq_len'], "answer"
        cur_rnd_utterance.append((tokenized_answer, [0, 0, 0, 0], self.get_token_type('A')))
        # ==============================================
        return cur_rnd_utterance, self.POS

    def get_random_answer(self, caption, tot_len, qa_pair, possible_answers):
        # ============ random answer ====================
        cur_rnd_utterance_random = caption.copy()
        if str(qa_pair['answer']).lower() in ['yes', 'no']:
            random_ans = 'yes' if str(qa_pair['answer']).lower() == 'no' else 'no'
        else:
            random_ans = str(np.random.choice(possible_answers))
            while str(qa_pair['answer']) == random_ans and len(possible_answers) > 1:
                random_ans = str(np.random.choice(possible_answers))

        tokenized_answer = self.tokenizer.encode(random_ans)
        # assert tot_len + len(tokenized_answer) + 1 < self.params['max_seq_len'], "random answer"
        cur_rnd_utterance_random.append((tokenized_answer, [0, 0, 0, 0], self.get_token_type('A')))
        return cur_rnd_utterance_random, self.NEG
        # ==============================================

    def cat_answers(self, qa_pair, tot_len, caption, possible_answers, qa_ind):
        """ This function returns all the sequences that should be concatenated for the model.
        In train mode, returns only two (gt and bad random answer).
        In evaluation mode, should return all possible options to let the model choose. """

        utterances = []
        if self._split == 'train' and not self.get_all_answers:
            # In case of random answers:
            cat_func = self.get_right_answer if (qa_ind < self.orig_len()) else self.get_random_answer
            utterances.append(cat_func(caption, tot_len, qa_pair, possible_answers))

        else:  # if evaluation, concat all answers
            # --- give all possible answers (<r> included in self.fixed_vocab !)
            gt_ans = str(qa_pair['answer']) if str(qa_pair['answer']) in possible_answers else self.R
            for ans in possible_answers:
                cur_rnd_utterance = caption.copy()
                tokenized_answer = self.tokenizer.encode(ans)
                # assert tot_len + len(tokenized_answer) + 1 < self.params['max_seq_len'], "answer"
                label = self.POS if gt_ans == str(ans) else self.NEG
                cur_rnd_utterance.append((tokenized_answer, [0, 0, 0, 0], self.get_token_type('A')))
                utterances.append((cur_rnd_utterance, label))

        return utterances

    def filter_preview(self, fig_feat):
        vis_bbox = deepcopy(fig_feat['vis_bbox'])
        vis_bbox[:, [0, 2]] *= fig_feat['text_feat']['x_axis']['w']
        vis_bbox[:, [0, 2]] += fig_feat['text_feat']['y_axis']['x']

        vis_bbox[:, [1, 3]] *= fig_feat['text_feat']['y_axis']['h']
        vis_bbox[:, [1, 3]] = fig_feat['text_feat']['x_axis']['y'] - vis_bbox[:, [1, 3]]

        b = (vis_bbox[:, 2] - vis_bbox[:, 0]) * (vis_bbox[:, 3] - vis_bbox[:, 1])
        # take these with area that is not ~400
        b = (b < 380) | (b > 420)
        return b

    def encode_and_reshape_img(self, fig_feat):
        """ This function reshapes and encodes the visual features data."""
        # temporary for mistake
        assert fig_feat['class'][0] == 100 or fig_feat['class'][0] == 999 or fig_feat['class'][
            0] == self.IMG_TOKEN_FEATURES_CLASS
        # <IMG> token doesn't need location
        fig_feat['vis_bbox'][0, :4] = 0

        fig_feat['legend_belonging_v'] = None
        if fig_feat['vis_bbox'].shape[-1] >= 5:
            fig_feat['legend_belonging_v'] = fig_feat['vis_bbox'][:, 4]
        else:
            fig_feat['legend_belonging_v'] = np.zeros(fig_feat['vis_bbox'].shape[0])

        # turn 100 class for <IMG> to len()
        assert fig_feat['class'][0] == self.IMG_TOKEN_FEATURES_CLASS, fig_feat['class']

        if False and self.params['dataset'] == 'figure_qa':
            fig_feat['class'][0] = 0
            vis_classes = torch.tensor(fig_feat['class'])
            # un = torch.unique(vis_classes)
            # un = un[torch.cat((torch.tensor([0]), torch.randperm(len(un) - 1) + 1))].tolist()
            # vis_classes = torch.tensor([un.index(c) for c in vis_classes]).unsqueeze(1)
            # vis_classes[vis_classes >= self.params['max_previews']] = 0
            vis_classes[(vis_classes >= 8) & (vis_classes <= 107)] = 8  # bar
            vis_classes[(vis_classes >= 108) & (vis_classes <= 157)] = 9  # line
            vis_classes[(vis_classes >= 158) & (vis_classes <= 207)] = 10  # dot
            vis_classes[(vis_classes >= 208) & (vis_classes <= 257)] = 11  # pie
            vis_classes = vis_classes.unsqueeze(1)
        else:
            fig_feat['class'][0] = self.params['categories']
            vis_classes = torch.tensor(fig_feat['class']).unsqueeze(1)

        if self.params['dataset'] == 'dvqa':
            vis_classes[vis_classes >= 62] -= 58
            vis_classes[0, 0] = self.params['categories']

        vis_feats = fig_feat['vis_feat']
        vis_boxes = fig_feat['vis_bbox'][:, :4]

        # get image features
        mask_prob_img = self.params["mask_prob_img"] if self._split == "train" else 0
        # mask_prob_img = 1

        features, spatials, image_mask, image_target, image_label, legend_belonging = \
            encode_image_input(vis_feats,
                               fig_feat['legend_belonging_v'],
                               vis_boxes,
                               vis_classes,
                               max_regions=self._max_region_num,
                               mask_prob=mask_prob_img
                               )

        return features, spatials, image_mask, image_target, image_label, legend_belonging

    def encode_and_reshape(self, utterances, print_sent=False):
        """ This function reshapes the data and encodes it (mask, token types, etc) """
        MAX_SEQ_LEN = self.params['max_seq_len']

        tokens_all = []
        mask_all = []
        segments_all = []
        sep_indices_all = []
        next_labels_all = []
        hist_len_all = []
        loc_all = []
        belonging_all = []
        mask_prob = self.params["mask_prob"] if self._split == "train" else 0

        for j, (context, sent_label) in enumerate(utterances):
            # print("{}: {}".format(j, tokens2str(context)))
            # unzip
            utt, loc, tok_types = zip(*context)

            if print_sent:
                print("Utterance: {}".format(self.tokens2str(utt)))

            tokens, segments, sep_indices, padded_locs, mask, legend_belonings = encode_text_input(utt, loc,
                                                                                                   tok_types,
                                                                                                   self.CLS,
                                                                                                   self.SEP,
                                                                                                   self.MASK,
                                                                                                   max_seq_len=MAX_SEQ_LEN,
                                                                                                   mask_prob=mask_prob)
            tokens_all.append(tokens)
            mask_all.append(mask)
            sep_indices_all.append(sep_indices)
            next_labels_all.append(torch.LongTensor([sent_label]))
            segments_all.append(segments)
            hist_len_all.append(torch.LongTensor([len(utt) - 1]))
            loc_all.append(padded_locs)
            belonging_all.append(legend_belonings)

        tokens_all_rnd = torch.cat(tokens_all, 0)
        mask_all_rnd = torch.cat(mask_all, 0)
        segments_all_rnd = torch.cat(segments_all, 0)
        sep_indices_all_rnd = torch.cat(sep_indices_all, 0)
        next_labels_all_rnd = torch.cat(next_labels_all, 0)
        hist_len_all_rnd = torch.cat(hist_len_all, 0)
        locations_all = torch.cat(loc_all, 0)
        legend_belonging_all = torch.cat(belonging_all, 0)

        return tokens_all_rnd, mask_all_rnd, segments_all_rnd, sep_indices_all_rnd, \
               next_labels_all_rnd, hist_len_all_rnd, locations_all, legend_belonging_all

    def pad_1st_dim(self, x, to):
        shape = list(x.shape)
        shape[0] = to
        shape = tuple(shape)
        padded_tensor = torch.zeros(shape, dtype=x.dtype)
        padded_tensor[:min(x.shape[0], to), ...] = x[:min(x.shape[0], to), ...]
        return padded_tensor

    def get_raw(self, qa_ind):
        # get data
        return self.get_qa(self._split, qa_ind)

    def __getitem__(self, qa_ind):
        """ Gets an index of QA pair, and returns the dict of all encoded data for the mode,
        including the image features + QA features."""
        print_sent = False

        qa_pair = self.get_qa(self._split, qa_ind)

        img_id = qa_pair['image_index']
        fig_feat = deepcopy(self.get_fig_feat(img_id))

        if self.params['dataset'] == 'figure_qa':
            # change names of colors
            qa_pair = deepcopy(self.get_qa(self._split, qa_ind))
            if 'test' in self.split:
                self.val_color_mapping.feature_replace(self.params, qa_pair, fig_feat)

        return self.get_encoded_qa(fig_feat, qa_pair, print_sent, qa_ind)

    def get_possible_answers(self, img_id, fig_feat=None):
        if fig_feat is None:
            fig_feat = self.get_fig_feat(img_id)
        text_feat = fig_feat['text_feat']
        is_hbar = self.is_hbar(fig_feat)
        if self.params['dataset'] != 'dvqa' and is_hbar:
            text_feat['x_axis'], text_feat['y_axis'] = text_feat['y_axis'], text_feat['x_axis']
            for ax in ['x_axis', 'y_axis']:
                text_feat[ax]['w'], text_feat[ax]['h'] = text_feat[ax]['h'], text_feat[ax]['w']

            if self.params['hbar_bbox_t']:
                fig_feat['vis_bbox'] = fig_feat['vis_bbox'][:, [3, 2, 1, 0]]
        _, _, possible_answers, _, _ = self.get_fig_caption(text_feat, is_hbar=is_hbar)

        possible_answers = [txt[0] for txt in possible_answers]
        # --- answers:
        return possible_answers + [opt for opt in self.fixed_vocab if opt not in possible_answers]

    def is_float(self, string):
        try:
            float(string)
            return True
        except:
            return False

    def tokenize_question_with_loc(self, ocr_features, qa_pair):
        """ Adding the location of each possible ocr feature detected in the question."""
        triplets = []
        ocr_in_question = []
        for string, loc in ocr_features:
            if loc is None:
                continue
            start_id = qa_pair['question_string'].find(string)
            if start_id > -1:
                ocr_in_question.append((string, loc, start_id))
        ocr_in_question = sorted(ocr_in_question, key=lambda x: x[-1])
        prev_id = 0
        for string, loc, start_id in ocr_in_question:
            if start_id > prev_id:
                tokenized_question = self.tokenizer.encode(qa_pair['question_string'][prev_id: start_id])
                triplets.append((tokenized_question, [0, 0, 0, 0], self.get_token_type('Q')))

            tokenized_question = self.tokenizer.encode(qa_pair['question_string'][start_id: start_id + len(string)])
            triplets.append((tokenized_question, loc, self.get_token_type('Q')))

            prev_id = start_id + len(string)
        if prev_id < len(qa_pair['question_string']) - 1:
            tokenized_question = self.tokenizer.encode(qa_pair['question_string'][prev_id:])
            triplets.append((tokenized_question, [0, 0, 0, 0], self.get_token_type('Q')))

        tokens, locs = [], []
        for toks, loc, _ in triplets:
            locs += [loc] * len(toks)
            tokens += toks

        return tokens, locs, self.get_token_type('Q')

    def is_hbar(self, fig_feat):
        if fig_feat['class'] is None or fig_feat['class'].shape[0] <= 1:
            return False
        if 'x_axis' not in fig_feat['text_feat']:
            return False
        vis_cls = (fig_feat['class'] != self.IMG_TOKEN_FEATURES_CLASS)
        if self.params['dataset'] == 'plotqa':
            num_bars = np.sum((8 <= fig_feat['class'][vis_cls]) & (fig_feat['class'][vis_cls] <= 80))
        elif self.params['dataset'] == 'plotqa_colorless':
            num_bars = np.sum((fig_feat['class'][vis_cls] == 0))
            if num_bars > 0:
                num_bars = np.sum((fig_feat['class'][vis_cls] == 0) | (fig_feat['class'][vis_cls] == 4))
        elif self.params['dataset'] == 'dvqa':
            num_bars = np.sum((62 <= fig_feat['class'][vis_cls]) & (fig_feat['class'][vis_cls] <= 120))
        else:
            assert False
        if num_bars / (fig_feat['class'].shape[0] - 1) >= 0.5:
            x_len = (fig_feat['vis_bbox'][vis_cls, 2] - fig_feat['vis_bbox'][vis_cls, 0])
            y_len = (fig_feat['vis_bbox'][vis_cls, 1] - fig_feat['vis_bbox'][vis_cls, 3])
            arg_max_bar = np.argmax((x_len * y_len))
            if y_len[arg_max_bar] / x_len[arg_max_bar] < 1:
                return True
        return False

    def get_encoded_qa(self, fig_feat, qa_pair, print_sent=False, qa_ind=-1, pretrain_ans=None):
        text_feat = fig_feat['text_feat']

        is_hbar = False
        if self.params['dataset'] == 'plotqa' and self.is_hbar(fig_feat):
            is_hbar = True
            text_feat['x_axis'], text_feat['y_axis'] = text_feat['y_axis'], text_feat['x_axis']
            for ax in ['x_axis', 'y_axis']:
                text_feat[ax]['w'], text_feat[ax]['h'] = text_feat[ax]['h'], text_feat[ax]['w']

            if self.params['hbar_bbox_t']:
                fig_feat['vis_bbox'] = fig_feat['vis_bbox'][:, [3, 2, 1, 0]]

        # ========= "caption": add fig text features: ======
        caption, tot_len, ocr_features, ticks_values, legend_pred = self.get_fig_caption(text_feat, is_hbar=is_hbar,
                                                                                         qa_pair=qa_pair)
        # --- question:
        tokenized_question = self.tokenizer.encode(qa_pair['question_string'])
        # caption.append((tokenized_question, [0, 0, 0, 0], self.get_token_type('Q')))
        caption.append(self.tokenize_question_with_loc(ocr_features, qa_pair))

        tot_len += len(tokenized_question) + 1  # + SEP
        if self.params['dataset'] != 'figure_qa':
            possible_answers = [txt[0] for txt in ocr_features]
            # --- answers:
            possible_answers = self.fixed_vocab if self.params['fixed_vocab'] else (
                        possible_answers + [opt for opt in self.fixed_vocab if opt not in possible_answers])
            if '_REGS' in self.params['qa_file']:
                possible_answers = [self.R, self.R]
        else:
            possible_answers = ['Yes', 'No']

        if self.params['binary_answers']:
            if 'answer' not in qa_pair:
                gt_answer = -1
            gt_answer = qa_pair['answer']
            utterances = [(caption, gt_answer)]

        else:
            utterances = self.cat_answers(qa_pair, tot_len, caption, possible_answers, qa_ind)

        # ----- encode and reshape text sequence:
        txt_input_sequence = self.encode_and_reshape(utterances, print_sent=print_sent)

        tokens_all_rnd, mask_all_rnd, segments_all_rnd, sep_indices_all_rnd, next_labels_all_rnd, \
        hist_len_all_rnd, locations_all, legend_belonging_all = txt_input_sequence
        item = {}
        item['id'] = torch.LongTensor([qa_ind])

        item['tokens'] = tokens_all_rnd.long().squeeze(0)
        item['segments'] = segments_all_rnd.long().squeeze(0)
        item['sep_indices'] = sep_indices_all_rnd.long().squeeze(0)
        item['mask'] = mask_all_rnd.long().squeeze(0)
        item['legend_belonging_t'] = legend_belonging_all.long().squeeze(0)
        item['loc'] = locations_all.squeeze(0)

        item['hist_len'] = hist_len_all_rnd.long()
        item['next_sentence_labels'] = next_labels_all_rnd.long()

        # for evaluation. 2nd dim is different for every sample, thus padded
        if (not self.params['fixed_vocab']) and (self.get_all_answers or self._split != 'train') and (
        not self.params['binary_answers']):
            for to_pad in self.PADDING_TXT:
                item[to_pad] = self.pad_1st_dim(item[to_pad], self.EVAL_PADDED_SIZE)

        item['gt'] = str(qa_pair['answer'])

        gt_ind = possible_answers.index(item['gt']) if (item['gt'] in possible_answers and '_REGS' not in self.params['qa_file']) else -1
        # item['DEBUGG'] = torch.tensor([0])
        if gt_ind == -1 and not self.params['BOT_MODE']:
            try:
                float(item['gt'])
            except:
                gt_ind = np.random.randint(len(possible_answers))
                if self.params['dataset'] != "dvqa" and not self.params['binary_answers'] and not self.params['BOT_MODE']:
                    # print("no GT in options: ", qa_ind, item['next_sentence_labels'], flush=True)
                    item['next_sentence_labels'][0] = self.NEG
                # item['DEBUGG'] = torch.tensor([1])

        # return item
        if gt_ind == -1 and (not self.params['binary_answers'] and ('_cls' not in self.params['qa_file'])):
            gt_ind = possible_answers.index(self.R)
            # kind of assert that the element can be regressed.
            # **Linear** scaling:
            tolerance_margin = np.mean([abs(float(ticks_values['y_axis'][i][1]) - float(ticks_values['y_axis'][i + 1][1])) for i in range(len(ticks_values['y_axis']) - 1)])
            tolerance_margin /= 2
            y_length = [abs(float(ticks_values['y_axis'][i][0]) / float(ticks_values['y_axis'][i][1])) for i in
                        range(len(ticks_values['y_axis']))]

            if self.params['BOT_MODE'] and qa_pair['answer'] is None:
                gt_value = 1
            else:
                gt_value = float(item['gt'])

            if len(y_length) == 0:
                # print(">>>> ", qa_ind, qa_pair, flush=True)
                item['R'] = [gt_value, True, 1, float(item['gt']) if float(item['gt']) != 0 else 1]
            else:
                y = np.mean(y_length)
                # print(self.params['tol_margin'], np.mean([abs(float(ticks_values['y_axis'][i][1]) - float(ticks_values['y_axis'][i + 1][1])) for i in range(len(ticks_values['y_axis']) - 1)]) / 10)
                item['R'] = [gt_value, True, self.params['tol_margin'], y]
            item['gt'] = torch.FloatTensor([gt_value])
            item['reg_target'] = torch.FloatTensor([item['R'][0] / item['R'][3]])
            # assert qa_pair['qid'] in self.REGRESSIONS, qa_pair['type']
            if self.params['CE_REG']:
                item['R'][0] = self.params['dvqa_floats'].index(item['R'][0])
        else:
            item['R'] = [0, False, 0, 0]
            item['gt'] = torch.FloatTensor([0])
            item['reg_target'] = torch.FloatTensor([0])

        item['needs_reg'] = torch.BoolTensor([item['R'][1]])

        item['tolerance_margin'] = torch.FloatTensor([item['R'][2]])
        item['R'] = torch.FloatTensor(item['R'])

        if self.params['dataset'] == 'figure_qa':
            if 'answer' not in qa_pair:
                item['gt_id'] = torch.LongTensor([-1])
            else:
                item['gt_id'] = torch.LongTensor([1 - qa_pair['answer']])
        else:
            item['gt_id'] = torch.LongTensor([gt_ind])

        item['num_ans'] = torch.LongTensor([len(possible_answers)])

        if 'plotqa' in self.params['dataset']:
            item['qid'] = str(qa_pair['qid'])
            item['qa_type'] = qa_pair['type'].replace('dot_line', 'dot')
            item['fig_type_id'] = torch.LongTensor([fig_type_to_id(qa_pair['type'])])
        elif self.params['dataset'] == 'dvqa':

            if qa_pair['template_id'] == 'structure':
                item['qid'] = 'S7'
            elif qa_pair['template_id'] == 'data':
                item['qid'] = 'D14'
            else:
                item['qid'] = 'A4'

            item['qa_type'] = 'vbar'

        # ----- encode and reshape visual sequence:
        features, spatials, image_mask, image_target, image_label, legend_belonging_v = self.encode_and_reshape_img(
            fig_feat)

        if self.params['dataset'] in ['figure_qa']:
            item['area'] = torch.zeros(self._max_region_num).double()
            if 'pie' in text_feat:
                areas = []
                for a in text_feat['pie']['areas']:
                    if a is None:
                        areas.append(0)
                    else:
                        areas.append(a)
                if len(areas) > 0:
                    item['area'][:len(text_feat['pie']['areas'])] = F.softmax(torch.tensor(areas).float(), dim=0)
                # item['area'][item['area'] != item['area']] = 0

        item['image_feat'] = features
        item['image_loc'] = spatials
        item['image_mask'] = image_mask.long()
        item['image_target'] = image_target.long()
        item['image_label'] = image_label.long()
        item['legend_belonging_v'] = legend_belonging_v.long()
        item['legend_pred'] = legend_pred

        if (self.get_all_answers or self.split != 'train') and (not self.params['binary_answers']):
            for to_exp_pad in self.PADDING_VIS:
                item[to_exp_pad] = item[to_exp_pad].expand((item['num_ans'],) + item[to_exp_pad].shape).contiguous()
                item[to_exp_pad] = self.pad_1st_dim(item[to_exp_pad], self.EVAL_PADDED_SIZE)

        return item

    def cut_batch_padding(self, item):
        if self.params['binary_answers']:
            return
        for to_cut in (self.PADDING_VIS + self.PADDING_TXT):
            x = item[to_cut]
            relevant = [x[i, :item['num_ans'][i], ...] for i in range(x.shape[0])]
            item[to_cut] = torch.cat(relevant, dim=0)

    def get_ans_type(self, qa_ind):
        ans = str(self.get_raw(qa_ind)['answer']).lower()

        if ans.lower() in ['yes', 'no']:
            # YES \ NO
            return 0

        elif ans in self.fixed_vocab_lower and '_REGS' not in self.params['qa_file']:
            # Fixed Vocabulary
            return 1
        else:
            # Open Vocabulary
            return 2


def fig_type_to_id(str_type):
    if str_type == "line":
        return 0
    elif str_type == 'vbar':
        return 1
    elif str_type == 'hbar':
        return 2
    elif str_type == 'dot' or str_type == 'dot_line':
        return 3
    else:
        assert False


class Color_Mapping:
    def __init__(self):
        self.cid_to_color = {8: 'Royal Blue',
                             9: 'Pale Green',
                             10: 'Dark Red',
                             11: 'Light Green',
                             12: 'Dark Salmon',
                             13: 'Coral',
                             14: 'Medium Purple',
                             15: 'Purple',
                             16: 'Dark Turquoise',
                             17: 'Orange Red',
                             18: 'Saddle Brown',
                             19: 'Navy Blue',
                             20: 'Violet',
                             21: 'Salmon',
                             22: 'Teal',
                             23: 'Dark Khaki',
                             73: 'Peru',
                             100: 'Light Slate',
                             104: 'Cyan',
                             90: 'Red',
                             24: 'Lawn Green',
                             25: 'Yellow Green',
                             26: 'Medium Orchid',
                             27: 'Blue',
                             28: 'Forest Green',
                             29: 'Turquoise',
                             30: 'Cornflower',
                             31: 'Medium Aqua',
                             32: 'Medium Seafoam',
                             33: 'Gold',
                             34: 'Deep Pink',
                             88: 'Green Yellow',
                             35: 'Rosy Brown',
                             36: 'Sky Blue',
                             37: 'Olive Drab',
                             38: 'Medium Mint',
                             39: 'Web Green',
                             40: 'Green',
                             41: 'Chartreuse',
                             75: 'Orange',
                             42: 'Medium Periwinkle',
                             43: 'Sandy Brown',
                             44: 'Lime Green',
                             45: 'Dark Cyan',
                             46: 'Indian Red',
                             47: 'Chocolate',
                             48: 'Tan',
                             49: 'Light Coral',
                             50: 'Dark Seafoam',
                             51: 'Rebecca Purple',
                             52: 'Yellow',
                             53: 'Web Purple',
                             54: 'Indigo',
                             55: 'Medium Turquoise',
                             56: 'Dodger Blue',
                             57: 'Dark Periwinkle',
                             58: 'Cadet Blue',
                             59: 'Dark Violet',
                             60: 'Dark Slate',
                             61: 'Black',
                             89: 'Aqua',
                             62: 'Dark Olive',
                             63: 'Light Sky Blue',
                             64: 'Burlywood',
                             65: 'Deep Sky Blue',
                             66: 'Medium Blue',
                             67: 'Steel Blue',
                             68: 'Gray',
                             69: 'Light Seafoam',
                             70: 'Violet Red',
                             71: 'Dark Orange',
                             72: 'Khaki',
                             74: 'Crimson',
                             76: 'Periwinkle',
                             107: 'Dark Blue',
                             94: 'Seafoam',
                             77: 'Light Salmon',
                             78: 'Tomato',
                             79: 'Blue Violet',
                             80: 'Light Gold',
                             81: 'Olive',
                             82: 'Dark Magenta',
                             83: 'Firebrick',
                             84: 'Bubblegum',
                             85: 'Dark Green',
                             86: 'Dim Gray',
                             87: 'Midnight Blue',
                             102: 'Brown',
                             91: 'Mint',
                             92: 'Slate',
                             93: 'Web Gray',
                             95: 'Dark Gold',
                             96: 'Dark Gray',
                             97: 'Web Maroon',
                             98: 'Sienna',
                             99: 'Maroon',
                             101: 'Orchid',
                             105: 'Dark Orchid',
                             103: 'Hot Pink',
                             106: 'Magenta'}

    def get_previews(self, fig_feat):
        vis_bbox = deepcopy(fig_feat['vis_bbox'])
        if 'pie' in fig_feat['text_feat']:
            r = fig_feat['text_feat']['pie']['radius']
            vis_bbox = vis_bbox * r
            # x1, x2:
            vis_bbox[:, 0] += 0
            vis_bbox[:, 2] += 0
            # y1, y2:
            vis_bbox[:, 1] *= -1
            vis_bbox[:, 1] -= 0
            vis_bbox[:, 3] *= -1
            vis_bbox[:, 3] -= 0
        else:
            vis_bbox[:, [0, 2]] *= fig_feat['text_feat']['x_axis']['w']
            vis_bbox[:, [0, 2]] += fig_feat['text_feat']['y_axis']['x']

            vis_bbox[:, [1, 3]] *= fig_feat['text_feat']['y_axis']['h']
            vis_bbox[:, [1, 3]] = fig_feat['text_feat']['x_axis']['y'] - vis_bbox[:, [1, 3]]

        b = (vis_bbox[:, 2] - vis_bbox[:, 0]) * (vis_bbox[:, 3] - vis_bbox[:, 1])
        # take these with area that is not ~400
        b = (b > 350) & (b < 455)
        return b

    def closest_node(self, node, nodes):
        nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        if len(dist_2) == 0:
            return None
        return np.argmin(dist_2)

    def feature_replace(self, params, qa_pair, fig_feat):
        mapping_dict = self.get_nearest_mapping(fig_feat)
        if mapping_dict is None:
            return
        mapping = lambda key: mapping_dict[key] if key in mapping_dict else key

        for ax in ['x_axis', 'y_axis']:
            if ax in fig_feat['text_feat']:
                new_ticks = []
                for t_name, t in fig_feat['text_feat'][ax]['ticks']:
                    if t_name in mapping_dict:
                        t_name = mapping(t_name)
                    new_ticks.append((t_name, t))
                # and replace
                fig_feat['text_feat'][ax]['ticks'] = new_ticks

        if 'legend' in fig_feat['text_feat']:
            new_labels = []
            for label in fig_feat['text_feat']['legend']['label']:
                new_labels.append(mapping(label))
            # and replace
            fig_feat['text_feat']['legend']['label'] = np.array(new_labels)

        c1_holder, c2_holder = "$_c1_$", "$_c2_$"
        c1 = mapping(qa_pair['color1_name'])
        c2 = c2_holder
        if qa_pair['color2_name'] != '--None--':
            c2 = mapping(qa_pair['color2_name'])

        new_question = params['question_templates'][str(qa_pair['question_id'] + 1)]
        new_question = new_question.replace(c1_holder, c1)
        new_question = new_question.replace(c2_holder, c2)
        qa_pair['question_string'] = new_question

    def get_nearest_mapping(self, ex_i):
        mapping = {}
        if 'legend' in ex_i['text_feat']:
            b = self.get_previews(ex_i)

            x = (ex_i['vis_bbox'][:, 0] + ex_i['vis_bbox'][:, 2]) / 2
            y = (ex_i['vis_bbox'][:, 1] + ex_i['vis_bbox'][:, 3]) / 2
            nodes = np.stack((x, y), axis=1)[b]
            # mapping = {label: None for label in ex_i['text_feat']['legend']['label']}

            for i, bbox in enumerate(ex_i['text_feat']['legend']['bbox']):
                point = np.array([bbox[0], (bbox[1] + bbox[3]) / 2])
                closest = self.closest_node(point, nodes)
                if closest is None:
                    return None
                if abs(nodes[closest][1] - point[1]) <= 5e-2:
                    nearest_c = ex_i['class'][b][closest]
                    mapping[ex_i['text_feat']['legend']['label'][i]] = self.cid_to_color[nearest_c]
        else:
            for ax in ['x_axis', 'y_axis']:
                t, l = ex_i['text_feat'][ax]['ticks'][1]
                try:
                    float(t)
                    continue
                except:
                    break

            # mapping = {label: None for label, _ in ex_i['text_feat'][ax]['ticks']}
            if ax == 'x_axis':
                x = (ex_i['vis_bbox'][1:, 0] + ex_i['vis_bbox'][1:, 2]) / 2
                y = ex_i['vis_bbox'][1:, 3] * 0
                nodes = np.stack((x, y), axis=1)
                same_ax = 0
            else:
                x = ex_i['vis_bbox'][1:, 0] * 0
                y = (ex_i['vis_bbox'][1:, 1] + ex_i['vis_bbox'][1:, 3]) / 2
                nodes = np.stack((x, y), axis=1)
                same_ax = 1

            for i, (name, l) in enumerate(ex_i['text_feat'][ax]['ticks']):
                if ax == 'x_axis':
                    point = np.array([l, 0])
                else:
                    point = np.array([0, l])

                closest = self.closest_node(point, nodes)
                if closest is None:
                    return None
                if abs(nodes[closest][same_ax] - point[same_ax]) <= 5e-2:
                    mapping[name] = self.cid_to_color[ex_i['class'][1:][closest]]

        return mapping
