import time
import glob
import re
import json
import torch
# Some basic setup:

# assert torch.__version__.startswith("1.6")
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from frcnn import get_plotqa_cfg, get_class_list
from detectron2.structures.boxes import Boxes
from feature_extraction.bbox_match import OCR_oracle
from feature_extraction.pie_area import get_pie_areas, Point
import difflib
import os, cv2
import numpy as np
import argparse
import pytesseract


CLASSES = get_class_list()


def sec_to_hhmmss(secs):
    return time.strftime('[%H:%M:%S]', time.gmtime(secs))


def est(num_ready, total, total_time):
    """  Returns estimated time for the process"""
    avg_time = total_time / num_ready
    return sec_to_hhmmss((total - num_ready) * avg_time)


def get_input(args, img_path):
    im = cv2.imread(img_path)
    height, width = im.shape[:2]
    image = args.predictor.aug.get_transform(im).apply_image(im)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]
    return inputs


def get_features(args, inputs, orig_shape, visual_only=True):
    """  Get features from inputs list """
    model = args.model
    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN
        # add whole image feature
        img_token = transform_bbox(args.on_cpu,
                                   orig_shape,
                                   inputs[0]['image'].shape[1:],
                                   np.array([0, 0, orig_shape[1], orig_shape[0]])[np.newaxis, ...].astype(np.float32))
        proposals[0].proposal_boxes.tensor = torch.cat((img_token.tensor, proposals[0].proposal_boxes.tensor))
        #
        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size

        img_token_loc = torch.tensor([0, 0, orig_shape[1], orig_shape[0]]).unsqueeze(0)
        s = torch.tensor([1])
        c = torch.tensor([1000])
        z = torch.tensor([0])

        if not args.on_cpu:
            img_token_loc = img_token_loc.cuda()
            s = s.cuda()
            z = z.cuda()
            c = c.cuda()

        if 0 in pred_inds[0]:
            pred_instances[0]['instances'] = pred_instances[0]['instances'][pred_inds[0] != 0]
            pred_inds[0] = pred_inds[0][pred_inds[0] != 0]

        pred_instances[0]['instances'].pred_boxes.tensor = torch.cat((img_token_loc,
                                                                     pred_instances[0]['instances'].pred_boxes.tensor))

        pred_instances[0]['instances'].scores = torch.cat((s, pred_instances[0]['instances'].scores))
        pred_instances[0]['instances'].pred_classes = torch.cat((c, pred_instances[0]['instances'].pred_classes))

        # vis_ids = visual_features_ids(pred_instances[0]['instances'], no_inc_classes)

        # if 0 not in vis_ids:
        #     vis_ids = np.concatenate((np.array([0]), vis_ids))

        pred_inds[0] = torch.cat((z, pred_inds[0]))
        # features of the proposed boxes
        feats = box_features[pred_inds]
        # cls scores:
        classes = pred_instances[0]['instances'].pred_classes.cpu().numpy()
        # bbox
        boxes = pred_instances[0]['instances'].pred_boxes.tensor.cpu().numpy()

    return feats, classes, boxes, images[0].shape[1:]


def transform_bbox(on_cpu, prev_size, new_size, bbox):
    """
    Get new bbox coordination according to previous image size.
    """
    y_, x_ = prev_size
    y, x = new_size

    x_scale = x / x_
    y_scale = y / y_

    # bbox = np.array(bbox).astype(np.float32)[np.newaxis, ...]
    bbox[:, 0] = bbox[:, 0] * x_scale
    bbox[:, 1] = bbox[:, 1] * y_scale
    bbox[:, 2] = bbox[:, 2] * x_scale
    bbox[:, 3] = bbox[:, 3] * y_scale

    new_boxes = torch.tensor(np.round(bbox).astype(np.int32))
    if not on_cpu:
        new_boxes = new_boxes.cuda()

    return Boxes(new_boxes)



COLORS_FIGUREQA = ['Dark Turquoise', 'Light Slate', 'Sandy Brown', 'Slate', 'Cadet Blue', 'Indigo', 'Deep Pink',
          'Light Sky Blue', 'Web Gray', 'Turquoise', 'Dark Salmon', 'Coral', 'Saddle Brown',
          'Light Salmon', 'Dark Periwinkle', 'Dark Magenta', 'Black', 'Violet', 'Orange Red', 'Brown',
          'Crimson', 'Dark Blue', 'Dark Orchid', 'Midnight Blue', 'Purple', 'Dark Red', 'Peru',
          'Medium Aqua', 'Dark Gold', 'Light Gold', 'Medium Seafoam', 'Green Yellow', 'Aqua',
          'Orchid', 'Royal Blue', 'Gold', 'Medium Purple', 'Blue Violet', 'Pale Green', 'Dark Seafoam',
          'Rebecca Purple', 'Deep Sky Blue', 'Web Purple', 'Green', 'Olive Drab', 'Chocolate', 'Mint',
          'Dark Cyan', 'Burlywood', 'Olive', 'Seafoam', 'Light Green', 'Cornflower', 'Cyan',
          'Medium Orchid', 'Firebrick', 'Violet Red', 'Dark Khaki', 'Teal', 'Hot Pink', 'Sienna',
          'Dodger Blue', 'Gray', 'Salmon', 'Sky Blue', 'Web Green', 'Dark Gray', 'Web Maroon',
          'Dark Olive', 'Maroon', 'Periwinkle', 'Red', 'Dark Slate', 'Forest Green', 'Rosy Brown',
          'Chartreuse', 'Lime Green', 'Dim Gray', 'Medium Mint', 'Tan', 'Navy Blue', 'Steel Blue',
          'Light Seafoam', 'Khaki', 'Yellow', 'Light Coral', 'Bubblegum', 'Orange', 'Medium Periwinkle',
          'Indian Red', 'Lawn Green', 'Dark Orange', 'Dark Violet', 'Medium Blue', 'Blue', 'Tomato',
          'Medium Turquoise', 'Yellow Green', 'Magenta', 'Dark Green']


def path_to_img_id(path):
    return int(os.path.basename(path).split(".")[0])


def validate_both_axes(axes_boxes, boxes, img_path):
    if len(axes_boxes['x_axis']) == 0 and len(axes_boxes['y_axis']) == 0:
        if args.dataset in ['plotqa', 'plotqa_colorless']:
            print("Warning: No axes were detected! ", img_path)
        axes_boxes['x_axis'] = np.array([np.min(boxes[:, 0]), np.max(boxes[:, 1]), np.max(boxes[:, 2]), np.max(boxes[:, 1])])
        axes_boxes['y_axis'] = np.array([np.min(boxes[:, 0]), np.min(boxes[:, 1]), np.min(boxes[:, 0]), np.max(boxes[:, 3])])
        return None

    elif len(axes_boxes['x_axis']) == 0:
        axes_boxes['y_axis'] = axes_boxes['y_axis'][0]
        new_x1 = (axes_boxes['y_axis'][0] + axes_boxes['y_axis'][2]) / 2
        new_x2 = np.max(boxes[:, 2])
        new_y = axes_boxes['y_axis'][3]
        axes_boxes['x_axis'] = np.array([new_x1, new_y, new_x2, new_y])
        return False
    elif len(axes_boxes['y_axis']) == 0:
        axes_boxes['x_axis'] = axes_boxes['x_axis'][0]
        new_x = axes_boxes['x_axis'][0]
        new_y1 = np.max(boxes[:, 1])
        new_y2 = (axes_boxes['x_axis'][1] + axes_boxes['x_axis'][3]) / 2
        axes_boxes['y_axis'] = np.array([new_x, new_y1, new_x, new_y2])
        return False
    else:
        axes_boxes['y_axis'] = axes_boxes['y_axis'][0]
        axes_boxes['x_axis'] = axes_boxes['x_axis'][0]
        return True


def get_GT_texts(args, img_path, cls_dict):
    """ Get textual data from oracle instead of OCR's output """
    image_id = path_to_img_id(img_path)
    if args.dataset == 'dvqa':
        x = args.gt_ann[image_id - 1]
        assert image_id == int(x['image'].split("_")[-1].split(".")[0])
    else:
        x = args.gt_ann[image_id]
        assert image_id == x['image_index']

    # ======= Collect GT texts =======
    bboxes, texts, classes = [], [], []

    if args.dataset == 'dvqa':
        # easy organized here
        for text in x['texts']:
            if text['text_function'] == 'legend_heading':
                continue
            box = text['bbox']
            if text['text_function'] == 'legend':
                # get our annotations for this labels
                orig_box = text['bbox']
                leg = 10 + orig_box[2] / 2
                box = [orig_box[0] - leg, orig_box[1], orig_box[2] + leg, orig_box[3]]

            bboxes.append(np.array([box[0] + 10, box[1] + 10, box[0] + box[2] - 10, box[1] + box[3] - 10]))
            texts.append(text['text'])

        return {'bboxes': np.array(bboxes), 'text': np.array(texts)}

    n_models = len(x['models'])
    # ----- title text:
    if args.dataset != 'figure_qa':
        bbox = x['general_figure_info']['title']['bbox']
        title_loc = np.array([bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']])
        # add
        texts.append(x['general_figure_info']['title']['text'])
        bboxes.append(title_loc)
        classes.append(cls_dict['title'])

    # ----- axis labels:
    for ax in ['x_axis', 'y_axis']:
        if ax not in x['general_figure_info']:
            continue

        # ---- ticks:
        labels = x['general_figure_info'][ax]['major_labels']['values']
        ticks = x['general_figure_info'][ax]['major_labels']['bboxes'][: len(labels) // 2]
        assert labels[: len(labels) // 2] == labels[len(labels) // 2:]
        labels = labels[: len(labels) // 2]
        # add
        texts += labels
        bboxes += [[box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h']] for box in ticks]
        classes += [cls_dict[ax[0] + 'ticklabel']] * len(labels)
        # ---- axis label
        if args.dataset != 'figure_qa':
            texts.append(x['general_figure_info'][ax]['label']['text'])
            # add
            bbox = x['general_figure_info'][ax]['label']['bbox']
            bboxes.append([bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']])
            classes.append(cls_dict[ax[0] + 'label'])

    # ----- legend labels:
    if 'legend' in x['general_figure_info']:
        assert len(x['general_figure_info']['legend']['items']) == n_models
        for item in x['general_figure_info']['legend']['items']:
            bbox = item['label']['bbox']
            bbox = [bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']]
            texts.append(item['label']['text'])
            bboxes.append(bbox)
            classes.append(cls_dict['legend_label'])

    return {'bboxes': np.array(bboxes), 'text': np.array(texts), 'class': np.array(classes)}


def get_axes_info(axes_boxes, ocr_output, cls_dict, img_path):
    axes_info = {'x_axis': dict(), 'y_axis': dict(), 'values_are_x': axes_boxes['values_are_x']}
    for ax in ['x_axis', 'y_axis']:
        # bbox x,y,w,h format
        bbox = {'x': axes_boxes[ax][0],
                'y': axes_boxes[ax][1],
                'w': axes_boxes[ax][2] - axes_boxes[ax][0],
                'h': axes_boxes[ax][3] - axes_boxes[ax][1]
                }
        # axis rule location
        axes_info[ax]['x'] = bbox['x'] + bbox['w'] / 2
        axes_info[ax]['y'] = bbox['y'] + bbox['h'] / 2
        # length of the axis
        # l = 'w' if ax == 'x_axis' else 'h'
        axes_info[ax]['w'] = bbox['w']
        axes_info[ax]['h'] = bbox['h']

    for ax in ['x_axis', 'y_axis']:
        ticks = ocr_output['class'] == cls_dict[f"{ax[0]}ticklabel"]
        # ticks location:
        ticks_boxes = ocr_output['bboxes'][ticks]
        horizontal = True if (ax == 'x_axis' and not axes_boxes['values_are_x']) or (ax == 'y_axis' and axes_boxes['values_are_x']) else False
        c2, c1 = (2, 0) if horizontal else (3, 1)
        ticks_val = (ticks_boxes[:, c2] + ticks_boxes[:, c1]) / 2

        # Normalize the ticks to [0,1] by R^2 coordinates
        # convert x,y ticks to R^2 axes
        if horizontal:
            ticks_val -= axes_info['y_axis']['x']
            ticks_val /= axes_info['x_axis']['w']
        else:
            ticks_val = axes_info['x_axis']['y'] - ticks_val
            ticks_val /= axes_info['y_axis']['h']

        texts = ocr_output['text'][ticks]
        if args.dataset == 'dvqa' and 'mathdefault' in texts[0]:
            y_ticks = []
            p = re.compile(r'(-?[0-9]+)\^{(-?[0-9]+)}')
            for i, val in enumerate(texts):
                b, e = p.findall(val)[0]
                y_ticks.append(f"{b}e{e}")
            texts = y_ticks

        axes_info[ax]['ticks'] = sorted(list(zip(*(texts, ticks_val))), key=lambda x: x[1])
        axes_info[ax]['label'] = ocr_output['text'][ocr_output['class'] == cls_dict[f'{ax[0]}label']]
        if len(axes_info[ax]['label']) > 0:
            axes_info[ax]['label'] = axes_info[ax]['label'][0]
        else:
            axes_info[ax]['label'] = ""
            if 'plotqa' in args.dataset:
                print(f"Warning: no {ax} label was found. {img_path}")

    return axes_info


def get_title_legends(axes_info, ocr_output, cls_dict):
    title_legend = {'title': dict(), 'legend': dict()}
    # title
    title_text = ocr_output['text'][ocr_output['class'] == cls_dict['title']]
    if len(title_text) > 0:
        title_legend['title']['text'] = title_text[0]
        bbox = ocr_output['bboxes'][ocr_output['class'] == cls_dict['title']][0]
        title_legend['title']['bbox'] = normalize_bbox(bbox[np.newaxis, ...], axes_info)[0]
    else:
        del title_legend['title']
    # legend labels:
    if type(cls_dict['legend_label']) is np.ndarray:
        legend_labels = [(c in cls_dict['legend_label']) for c in ocr_output['class']]
    else:
        legend_labels = ocr_output['class'] == cls_dict['legend_label']

    title_legend['legend']['label'] = ocr_output['text'][legend_labels]

    if len(title_legend['legend']['label']) == 0:
        del title_legend['legend']
    else:
        bbox = ocr_output['bboxes'][legend_labels]
        title_legend['legend']['bbox'] = normalize_bbox(bbox, axes_info)

    return title_legend


def normalize_ticks_data(info):
    """ Normalize the ticks to [0,1] by R^2 coordinates """
    # convert x,y ticks to R^2 axes
    info['x_axis']['ticks'][1] -= info['y_axis']['x']
    info['y_axis']['ticks'][1] = info['x_axis']['y'] - info['y_axis']['ticks'][1]
    # normalize to [0,1]
    info['x_axis']['ticks'][1] /= info['x_axis']['w']
    info['y_axis']['ticks'][1] /= info['y_axis']['h']
    # re-compose
    info['x_axis']['ticks'] = list(zip(info['x_axis']['ticks'][0], info['x_axis']['ticks'][1]))
    info['y_axis']['ticks'] = list(zip(info['y_axis']['ticks'][0], info['y_axis']['ticks'][1]))

    return info


def normalize_bbox(bbox, info):
    """
    Normalizing the coordinates to a scale of R^2 axes, when the graph in the image is between 0,0 to 1,1
    """
    # bbox structure: (x1, y1, x2, y2)
    bbox = bbox.astype(np.float32)
    # x1, x2:
    bbox[:, 0] = (bbox[:, 0] - info['y_axis']['x']) / info['x_axis']['w']
    bbox[:, 2] = (bbox[:, 2] - info['y_axis']['x']) / info['x_axis']['w']
    # y1, y2:
    bbox[:, 1] = (info['x_axis']['y'] - bbox[:, 1]) / info['y_axis']['h']
    bbox[:, 3] = (info['x_axis']['y'] - bbox[:, 3]) / info['y_axis']['h']
    return bbox.astype(np.float32)


def get_nonvis_ids(args):
    # dicts with cls number - 1 (starts from 0)
    if args.dataset == "plotqa":
        cls_dict = {'legend_label': 0, 'title': 1, 'xlabel': 2, 'xticklabel': 3,
               'ylabel': 4, 'yticklabel': 5, 'x_axis': 6, 'y_axis': 7}
        return np.arange(8), cls_dict

    elif args.dataset == "plotqa_colorless":
        cls_dict = {'legend_label': 2, 'title': 5, 'xlabel': 6, 'xticklabel': 7,
                    'ylabel': 8, 'yticklabel': 9, 'x_axis': 10, 'y_axis': 11}
        return np.array([2, 5, 6, 7, 8, 9, 10, 11]), cls_dict

    elif args.dataset == 'figure_qa':
        cls_dict = {'legend_label': 6, 'title': None, 'xlabel': 2, 'xticklabel': 1,
               'ylabel': 5, 'yticklabel': 4, 'x_axis': 0, 'y_axis': 3}
        return np.arange(8), cls_dict

    elif args.dataset == 'dvqa':
        cls_dict = {'legend_label': np.arange(4, 62), 'title': 2, 'xlabel': None, 'xticklabel': 0,
               'ylabel': 1, 'yticklabel': 3, 'x_axis': None, 'y_axis': None}
        return np.arange(62), cls_dict


def dvqa_axes(cls_dict, oracle_ocr, vis_boxes):
    values = oracle_ocr['text'][oracle_ocr['class'] == cls_dict['yticklabel']]
    zero = [0, 0, 0, 0]
    is_value = np.array([True] * len(values))

    if 'mathdefault' in values[0]:
        y_ticks = []
        p = re.compile(r'(-?[0-9]+)\^{(-?[0-9]+)}')
        for i, val in enumerate(values):
            b, e = p.findall(val)[0]
            v = float(f"{b}e{e}")
            y_ticks.append(v)
            if v == 0:
                zero = oracle_ocr['bboxes'][oracle_ocr['class'] == cls_dict['yticklabel']][i]
    else:
        y_ticks = []
        for i, t in enumerate(values):
            try:
                v = float(t.replace("−", "-"))
                y_ticks.append(v)
                if v == 0:
                    zero = oracle_ocr['bboxes'][oracle_ocr['class'] == cls_dict['yticklabel']][i]
            except:
                # not a number for some reason
                is_value[i] = False
                print(f"\t   failed to float() this: {t}")

    if len(y_ticks) == 0:
        return None

    low, high = np.argmin(y_ticks), np.argmax(y_ticks)
    low = oracle_ocr['bboxes'][oracle_ocr['class'] == cls_dict['yticklabel']][is_value][low]
    high = oracle_ocr['bboxes'][oracle_ocr['class'] == cls_dict['yticklabel']][is_value][high]
    if high[0] - low[0] >= 50:
        # values are x_axis:
        h = (low[3] - low[1]) / 2
        zero = (zero[0] + zero[2]) / 2
        y_start = np.max(oracle_ocr['bboxes'][oracle_ocr['class'] == cls_dict['xticklabel']][:, 2]) if zero == 0 else zero
        a = np.min(vis_boxes[1:, 1]) if len(vis_boxes) > 1 else np.min(oracle_ocr['bboxes'][oracle_ocr['class'] == cls_dict['xticklabel']][:, 1])
        y_axis = np.array([[y_start - 5, a, y_start + 5, low[1] - h]])
        x_axis = np.array([[y_start, low[1] - h, (high[0] + high[2]) / 2, high[3] - h]])
        return zero, {'x_axis': x_axis, 'y_axis': y_axis, 'values_are_x': True}
    else:
        # values are y_axis:
        w = (high[2] - high[0]) / 2
        if (zero[1] + zero[2]) / 2 > 0:
            low = zero
        zero = (zero[1] + zero[2]) / 2
        a = np.max(vis_boxes[1:, 3]) if len(vis_boxes) > 1 else np.max(oracle_ocr['bboxes'][oracle_ocr['class'] == cls_dict['xticklabel']][:, 3])
        b = np.max(vis_boxes[1:, 2]) if len(vis_boxes) > 1 else np.max(oracle_ocr['bboxes'][oracle_ocr['class'] == cls_dict['xticklabel']][:, 3])
        x_axis = np.array([[low[2], low[1] if min(y_ticks) <= 0 else (a - 5), b, low[3] if min(y_ticks) <= 0 else (a + 5)]])
        y_axis = np.array([[high[0] + w, (high[1] + high[3]) / 2, high[2] + w, (x_axis[0][1] + x_axis[0][3]) / 2]])
        return zero, {'x_axis': x_axis, 'y_axis': y_axis, 'values_are_x': False}


def process_chunk(args, image_files):
    start = time.time()
    step_start = time.time()
    feature_lst = list()
    for i, img_path in enumerate(image_files):
        PRINT_EVERY = 10 if args.OCR else 100
        if i > 0 and i % PRINT_EVERY == 0:
            print("     i:{}/{}".format(i, len(image_files)), "step_time:", sec_to_hhmmss(time.time() - step_start),
                  "est for chunk:", est(i + 1, len(image_files), time.time() - start), flush=True)
            step_start = time.time()
        # build input
        inputs = get_input(args, img_path)
        # calculate features:
        orig_shape = (inputs[0]['height'], inputs[0]['width'])
        # print(img_path)
        feats, classes, boxes, img_shape = get_features(args, inputs, orig_shape, visual_only=False)
        # separate feature kind
        txt_cls_ids, cls_dict = get_nonvis_ids(args)
        if args.dataset == 'figure_qa':
            # exclude "pie"
            non_vis = np.array([i for i in range(len(boxes)) if (classes[i] in txt_cls_ids) and (classes[i] != cls_dict['x_axis'] and classes[i] != cls_dict['y_axis'] and classes[i] != 7)])
        else:
            non_vis = np.array([i for i in range(len(boxes)) if (classes[i] in txt_cls_ids) and (classes[i] != cls_dict['x_axis'] and classes[i] != cls_dict['y_axis'])])
        vis = np.array([i for i in range(len(boxes)) if (classes[i] not in txt_cls_ids) and (classes[i] != cls_dict['x_axis'] and classes[i] != cls_dict['y_axis'])])

        if args.dataset == 'dvqa':
            # legend label in this dataset are vis feature too (preview + label):
            legend_labels = np.array([i for i in range(len(boxes)) if classes[i] in cls_dict['legend_label']], dtype=np.int32)
            vis = np.concatenate((vis, legend_labels), axis=0)

        # Oracle OCR:
        if len(non_vis) == 0 and not args.ocr_gt:
            print("Error: ", img_path)
            feature_lst.append({"image_id": path_to_img_id(img_path),
                                "vis_feat": None,
                                "vis_bbox": None,
                                # "class_prob": scores.cpu().numpy(),
                                "class": None,
                                "text_feat": None,
                                "width": None,
                                "height": None
                                })
            continue
        if args.OCR:
            preds_feats = {'bboxes': boxes[non_vis], 'class': classes[non_vis]}
            oracle_ocr = preds_feats
            oracle_ocr['text'] = np.array(apply_OCR(img_path, boxes[non_vis], classes[non_vis], cls_dict))
        else:
            gt_feats = get_GT_texts(args, img_path, cls_dict)
            oracle_ocr = gt_feats
            if not args.ocr_gt:
                preds_feats = {'bboxes': boxes[non_vis], 'class': classes[non_vis]}
                oracle_ocr = OCR_oracle(preds_feats, gt_feats)

        # get axes information:
        zero_loc = 0
        if args.dataset == 'dvqa':
            try:
                zero_loc, axes_boxes = dvqa_axes(cls_dict, oracle_ocr, boxes[vis])
            except Exception as e:
                print(e)
                print(f"img_path: {img_path}")
                assert False
            if axes_boxes is None:
                print("Error: ", img_path)
                feature_lst.append({"image_id": path_to_img_id(img_path),
                                    "vis_feat": None,
                                    "vis_bbox": None,
                                    # "class_prob": scores.cpu().numpy(),
                                    "class": None,
                                    "text_feat": None,
                                    "width": None,
                                    "height": None
                                    })
                continue
        else:
            axes_boxes = {'x_axis': boxes[classes == cls_dict['x_axis']],
                          'y_axis': boxes[classes == cls_dict['y_axis']],
                          'values_are_x': False}

        axes = validate_both_axes(axes_boxes, boxes, img_path)
        if axes is None and args.dataset in ['figure_qa']:
            # Probably Pie-chart

            areas, center, r = get_pie_areas(boxes[vis][1:])
            if r is None:
                r = (boxes[classes == 7][0, 2] - boxes[classes == 7][0, 0])
                r += (boxes[classes == 7][0, 3] - boxes[classes == 7][0, 1])
                r /= 4
                center = Point((boxes[classes == 7][0, 2] + boxes[classes == 7][0, 0]) / 2,
                               (boxes[classes == 7][0, 3] + boxes[classes == 7][0, 1]) / 2
                               )
            axes_by_radius = {'x_axis': {'y': center.y, 'w': r},
                              'y_axis': {'x': center.x, 'h': r}}
            # title & legend
            textual_features = get_title_legends(axes_by_radius, oracle_ocr, cls_dict)
            # normalize by center and radius
            feats, classes, boxes = feats[vis], classes[vis], boxes[vis]
            # x1, x2:
            boxes[:, 0] = (boxes[:, 0] - center.x) / r
            boxes[:, 2] = (boxes[:, 2] - center.x) / r
            # y1, y2:
            boxes[:, 1] = (center.y - boxes[:, 1]) / r
            boxes[:, 3] = (center.y - boxes[:, 3]) / r
            # add pie charts details:
            textual_features['pie'] = {'areas': areas, 'radius': r}
        else:
            axes_info = get_axes_info(axes_boxes, oracle_ocr, cls_dict, img_path)
            title_legend = get_title_legends(axes_info, oracle_ocr, cls_dict)
            # merge
            textual_features = {**axes_info, **title_legend}
            # # get GT textual:
            # textual_features = get_GT_textual_features(args, img_pat
            #
            feats, classes, boxes = feats[vis], classes[vis], boxes[vis]

            # normalize visual features bboxes:
            boxes = normalize_bbox(boxes, textual_features)

        # add info
        feature_lst.append({"image_id": path_to_img_id(img_path),
                            "vis_feat": feats.cpu().numpy(),
                            "vis_bbox": boxes,
                            # "class_prob": scores.cpu().numpy(),
                            "class": classes,
                            "text_feat": textual_features,
                            "width": img_shape[0],
                            "height": img_shape[1]
                            })
    return feature_lst


def apply_OCR(img_path, boxes, classes, cls_dict):
    img = cv2.imread(img_path)
    c = boxes.astype(int)
    xticks = boxes[classes == cls_dict['xticklabel']]

    c[:, 1] = np.maximum(c[:, 1] - 5, 0)
    c[:, 0] = np.maximum(c[:, 0] - 5, 0)

    c[:, 2] = np.minimum(c[:, 2] + 5, img.shape[1] - 1)
    c[:, 3] = np.minimum(c[:, 3] + 5, img.shape[0] - 1)

    prop = 1
    if len(xticks) > 0:
        prop = np.median((xticks[:, 3] - xticks[:, 1]) / (xticks[:, 2] - xticks[:, 0]))

    texts = []
    for k in range(len(boxes)):
        if classes[k] == cls_dict['xticklabel'] and prop >= 3:
            # check if rotation is needed
            crop = cv2.rotate(img[c[k][1]:c[k][3], c[k][0]:c[k][2], :], 2)
        else:
            crop = img[c[k][1]:c[k][3], c[k][0]:c[k][2], :]
        try:
            OCR = pytesseract.image_to_string(crop).split("\n")
        except:
            print("FUCKKKK", crop.shape)
            print(img_path)
            assert False

        if OCR[0] in ['\x0c', '\n\x0c']:
            texts.append("0")
            continue

        closest = difflib.get_close_matches(OCR[0], COLORS_FIGUREQA)
        if len(closest) > 0:
            texts.append(closest[0])
            continue

        for num in [OCR[0], OCR[0][1:], OCR[0][:-1], OCR[0][1:-1]]:
            try:
                float(num)
                texts.append(num)
                break
            except:
                pass
        else:
            texts.append(OCR[0])

    return texts


def chunk_gen(lst, chunk_size):
    """ chunk generator for lst """
    chunks = None
    if args.chunk is not None:
        chunks = [int(n) for n in args.chunk.split(":")]
    for c_id, k in enumerate(range(0, len(lst), chunk_size)):
        if chunks and not (chunks[0] <= c_id < chunks[1]):
            # print("======= passing chunk k:", k, "========")
            continue
        yield c_id, lst[k: k + chunk_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from PlotQA dataset')
    parser.add_argument('--dir-path', type=str, default="")
    parser.add_argument('--load-weights', type=str, required=True)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--cuda-num', type=str, default=None)
    parser.add_argument('--ticks-class', type=str, default="7,9")
    parser.add_argument('--output', type=str, default="../CRCT/data/fig_features/train/")
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument("--no-axes", action="store_true", help="flag if you don't want to detect axes")
    parser.add_argument('--chunk', type=str, default=None)
    parser.add_argument("--on_cpu", action="store_true", help="if to run on cpu")
    parser.add_argument('--ocr-gt', action="store_true", help="completly oracle OCR")
    parser.add_argument('--OCR', action="store_true", help="extract features with OCR usage")
    parser.add_argument('--dataset', type=str, help="dataset name", choices=['figure_qa', 'plotqa', 'dvqa'], default='plotqa')
    args = parser.parse_args()

    print(f"Reading ground [{args.split}] annotations...")
    ann_path = args.dir_path + args.split + "/annotations.json"
    if os.path.isfile(ann_path):
        with open(ann_path, "r") as f:
            args.gt_ann = json.load(f)
    else:
        args.gt_ann = None

    # configs
    cfg = get_plotqa_cfg(args)
    if args.on_cpu:
        cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = args.load_weights
    if args.cuda_num is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_num

    # prepare model:
    predictor = DefaultPredictor(cfg)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    args.model = model
    args.predictor = predictor
    # ----------
    os.makedirs(args.output, exist_ok=True)
    input_samples = []
    # get sorted file names
    img_files_lst = glob.glob(os.path.join(args.dir_path + args.split + "/png/", "*.png"))
    img_files_lst = sorted(img_files_lst, key=lambda x: float(re.findall("(\d+)", x)[-1]))
    #
    num_of_chunks = len(range(0, len(img_files_lst), args.batch_size))
    print("Dataset images path:", args.dir_path + args.split + "/png/")
    print("Start to calculate. Total chunks: " + str(num_of_chunks), flush=True)
    start = time.time()
    step_start = time.time()

    for k, (c_id, image_files) in enumerate(chunk_gen(img_files_lst, args.batch_size)):
        info = process_chunk(args, image_files)
        np.save(os.path.join(args.output, "{}.npy".format(c_id)), info)
        print("Chunk saved: {}/{}.".format(c_id, num_of_chunks), "chunk_time:", sec_to_hhmmss(time.time() - step_start),
              "est:", est(k + 1, num_of_chunks, time.time() - start), flush=True)
        step_start = time.time()
