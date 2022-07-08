import os
import torch
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
# ----- trainer --
import time
import contextlib
try:
    _nullcontext = contextlib.nullcontext  # python 3.7+
except AttributeError:

    @contextlib.contextmanager
    def _nullcontext(enter_result=None):
        yield enter_result
# -----
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, SimpleTrainer, default_argument_parser, launch
from detectron2.evaluation import COCOEvaluator


class BigBatchTrainer(SimpleTrainer):

    def __init__(self, model, data_loader, optimizer, batch_multiply):
        super(BigBatchTrainer, self).__init__(model, data_loader, optimizer)
        self.batch_multiply = batch_multiply

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        # print(self.iter)
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        if self.iter % self.batch_multiply == 0:
            # print("*"*20, "BOOM", "*"*20)
            start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        if self.iter % self.batch_multiply == 0:
            data_time = time.perf_counter() - start
        """
        If you want to do something with the losses, you can wrap the model.
        """

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if self.iter % self.batch_multiply == 0:
            self.optimizer.zero_grad()
        losses.backward()
        if self.iter % self.batch_multiply == 0:
            self._write_metrics(loss_dict, data_time)

            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method. But it is
            suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
            """
            self.optimizer.step()


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = "plotqa_val_output"
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, tasks=('bbox',),
                             distributed=True,
                             output_dir=output_folder,
                             use_fast_impl=True)


def get_data_lst(dataset='plotqa'):
    if dataset == 'plotqa':
        png_paths = "../CRCT/Data/PlotQA/"
        coco_format_path = 'PlotQA/coco_format_plotqa'

        return [("plotqa_train1", f"{coco_format_path}/train_50k_annotations_inc_axes_colors.json", png_paths + "train/png"),
                ("plotqa_train2", f"{coco_format_path}/train_50k_1l_annotations_inc_axes_colors.json", png_paths + "train/png"),
                ("plotqa_train3", f"{coco_format_path}/train_1l_end_annotations_inc_axes_colors.json", png_paths + "train/png"),
                ("plotqa_val", f"{coco_format_path}/val_annotations_inc_axes_colors.json", png_paths + "val/png"),
                ("plotqa_test", f"{coco_format_path}/test_annotations_inc_axes_colors.json", png_paths + "test/png")]

    elif dataset == 'figure_qa':
        base = "FigureQA/"
        return [("figurqa_train1", f"{base}/figureqa_coco/train1/new_figureqa.json", f"{base}/figureqa/train1/png/")]

    elif dataset == 'dvqa':
        base = "DVQA/"
        return [("dvqa_train", f"{base}/coco/train_metadata_new.json", f"{base}/train/png/"),
                ("dvqa_val_easy", f"{base}/coco/val_easy_metadata_new.json", f"{base}/val_easy/png/")]
    else:
        raise RuntimeError(f"dataset was not found: {dataset}")


def get_class_list(args=None):
    import json
    # LABELS = ['legend_label', 'title', 'xlabel', 'xticklabel', 'ylabel', 'yticklabel', 'x_axis', 'y_axis', 'bar_0', 'bar_1', 'bar_2', 'bar_3', 'bar_4', 'bar_5', 'bar_6', 'bar_7', 'bar_8', 'bar_9', 'bar_10', 'bar_11', 'bar_12', 'bar_13', 'bar_14', 'bar_15', 'bar_16', 'bar_17', 'bar_18', 'bar_19', 'bar_20', 'bar_21', 'bar_22', 'bar_23', 'bar_24', 'bar_25', 'bar_26', 'bar_27', 'bar_28', 'bar_29', 'bar_30', 'bar_31', 'bar_32', 'bar_33', 'bar_34', 'bar_35', 'bar_36', 'bar_37', 'bar_38', 'bar_39', 'bar_40', 'bar_41', 'bar_42', 'bar_43', 'bar_44', 'bar_45', 'bar_46', 'bar_47', 'bar_48', 'bar_49', 'bar_50', 'bar_51', 'bar_52', 'bar_53', 'bar_54', 'bar_55', 'bar_56', 'bar_57', 'bar_58', 'bar_59', 'bar_60', 'bar_61', 'bar_62', 'bar_63', 'bar_64', 'bar_65', 'bar_66', 'bar_67', 'bar_68', 'bar_69', 'bar_70', 'bar_71', 'bar_72', 'dot_line_0', 'dot_line_1', 'dot_line_2', 'dot_line_3', 'dot_line_4', 'dot_line_5', 'dot_line_6', 'dot_line_7', 'dot_line_8', 'dot_line_9', 'dot_line_10', 'dot_line_11', 'dot_line_12', 'dot_line_13', 'dot_line_14', 'dot_line_15', 'dot_line_16', 'dot_line_17', 'dot_line_18', 'dot_line_19', 'dot_line_20', 'dot_line_21', 'dot_line_22', 'dot_line_23', 'dot_line_24', 'dot_line_25', 'dot_line_26', 'dot_line_27', 'dot_line_28', 'dot_line_29', 'dot_line_30', 'dot_line_31', 'dot_line_32', 'dot_line_33', 'dot_line_34', 'dot_line_35', 'dot_line_36', 'dot_line_37', 'dot_line_38', 'dot_line_39', 'dot_line_40', 'dot_line_41', 'dot_line_42', 'dot_line_43', 'dot_line_44', 'dot_line_45', 'dot_line_46', 'dot_line_47', 'dot_line_48', 'dot_line_49', 'dot_line_50', 'dot_line_51', 'dot_line_52', 'dot_line_53', 'dot_line_54', 'dot_line_55', 'dot_line_56', 'dot_line_57', 'dot_line_58', 'dot_line_59', 'dot_line_60', 'dot_line_61', 'dot_line_62', 'dot_line_63', 'dot_line_64', 'dot_line_65', 'dot_line_66', 'dot_line_67', 'dot_line_68', 'dot_line_69', 'dot_line_70', 'dot_line_71', 'dot_line_72', 'line_0', 'line_1', 'line_2', 'line_3', 'line_4', 'line_5', 'line_6', 'line_7', 'line_8', 'line_9', 'line_10', 'line_11', 'line_12', 'line_13', 'line_14', 'line_15', 'line_16', 'line_17', 'line_18', 'line_19', 'line_20', 'line_21', 'line_22', 'line_23', 'line_24', 'line_25', 'line_26', 'line_27', 'line_28', 'line_29', 'line_30', 'line_31', 'line_32', 'line_33', 'line_34', 'line_35', 'line_36', 'line_37', 'line_38', 'line_39', 'line_40', 'line_41', 'line_42', 'line_43', 'line_44', 'line_45', 'line_46', 'line_47', 'line_48', 'line_49', 'line_50', 'line_51', 'line_52', 'line_53', 'line_54', 'line_55', 'line_56', 'line_57', 'line_58', 'line_59', 'line_60', 'line_61', 'line_62', 'line_63', 'line_64', 'line_65', 'line_66', 'line_67', 'line_68', 'line_69', 'line_70', 'line_71', 'line_72', 'preview_0', 'preview_1', 'preview_2', 'preview_3', 'preview_4', 'preview_5', 'preview_6', 'preview_7', 'preview_8', 'preview_9', 'preview_10', 'preview_11', 'preview_12', 'preview_13', 'preview_14', 'preview_15', 'preview_16', 'preview_17', 'preview_18', 'preview_19', 'preview_20', 'preview_21', 'preview_22', 'preview_23', 'preview_24', 'preview_25', 'preview_26', 'preview_27', 'preview_28', 'preview_29', 'preview_30', 'preview_31', 'preview_32', 'preview_33', 'preview_34', 'preview_35', 'preview_36', 'preview_37', 'preview_38', 'preview_39', 'preview_40', 'preview_41', 'preview_42', 'preview_43', 'preview_44', 'preview_45', 'preview_46', 'preview_47', 'preview_48', 'preview_49', 'preview_50', 'preview_51', 'preview_52', 'preview_53', 'preview_54', 'preview_55', 'preview_56', 'preview_57', 'preview_58', 'preview_59', 'preview_60', 'preview_61', 'preview_62', 'preview_63', 'preview_64', 'preview_65', 'preview_66', 'preview_67', 'preview_68', 'preview_69', 'preview_70', 'preview_71', 'preview_72']
    data_lst = get_data_lst() if args is None else get_data_lst(dataset=args.dataset)
    annotations_val = data_lst[-1][1]
    with open(annotations_val, "r") as f:
        gt_ann = json.load(f)

    return [cat['name'] for cat in gt_ann['categories']]


def get_plotqa_cfg(args):

    # batch_size = args.batch_size
    for (name, annotations, img_dir) in get_data_lst(dataset=args.dataset):
        # print("===== REGISTERING ======")
        # print(f"name: {name}")
        # print(f"annotations: {annotations}")
        # print(f"img_dir: {img_dir}")
        register_coco_instances(name, {}, annotations, img_dir)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # cfg.merge_from_file("detectron2/configs/e2e_faster_rcnn_R-50-FPN_1x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))

    if args.dataset == 'plotqa':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        cfg.DATASETS.TRAIN = ("plotqa_train1", "plotqa_train2", "plotqa_train3",)
        cfg.DATASETS.TEST = ["plotqa_test"]
    elif args.dataset == 'plotqa_colorless':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.DATASETS.TRAIN = ("plotqa_train1", "plotqa_train2", "plotqa_train3",)
        cfg.DATASETS.TEST = ["plotqa_test"]
    elif args.dataset == 'figure_qa':
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ["figurqa_train1"]
        cfg.DATASETS.TEST = ()  #("plotqa_val", "plotqa_test") #todo: check why did it happend, test failed to generate annots
    elif args.dataset == 'dvqa':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.DATASETS.TRAIN = ["dvqa_train"]
        cfg.DATASETS.TEST = ["dvqa_val_easy"]
    else:
        raise RuntimeError(f"dataset was not found: {args.dataset}")

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    try:
        cfg.MODEL.WEIGHTS = args.model_path
    except:
        cfg.MODEL.WEIGHTS = args.load_weights

    print(f"cfg.MODEL.WEIGHTS: {cfg.MODEL.WEIGHTS}")

    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")

    cfg.NUM_WORKERS = 10
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = (10000, 20000, 30000, 40000, 50000, 60000, 70000)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(get_class_list(args))

    # cfg.SOLVER.IMS_PER_BATCH = batch_size  # 4
    cfg.SOLVER.BASE_LR = 0.00025
    # cfg.SOLVER.BASE_LR = 0.000025
    # -------- mixed precision --------------
    # cfg.SOLVER.AMP.ENABLED = True
    return cfg


def main(args):
    # --------------- configuration ---------------
    print("2: config...")
    cfg = get_plotqa_cfg(args)
    if args.output_dir is not None:
        cfg.OUTPUT_DIR = "./output/" + args.output_dir

    # cfg.MODEL.WEIGHTS = args.load_weights
    # ---------------------------------
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # batch_multiply = int(128 // cfg.SOLVER.IMS_PER_BATCH)
    # cfg.SOLVER.MAX_ITER *= batch_multiply

    trainer = CocoTrainer(cfg)
    # trainer._trainer = BigBatchTrainer(trainer.model, trainer.data_loader, trainer.optimizer, batch_multiply)
    trainer.resume_or_load(resume=args.resume)
    # ----
    if args.test:
        print("3: testing...")
        return trainer.test(cfg, trainer.model)

    print("3: training...")
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--cuda-num", default=None, type=str)
    parser.add_argument("--no-axes", action="store_true", help="flag if you don't want to detect axes")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--load-weights",
                        default=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"), # Let training initialize from model zoo,
                        type=str
    )
    parser.add_argument('--dataset', type=str, help="dataset name", choices=['figure_qa', 'plotqa', 'dvqa', 'plotqa_colorless'], default='plotqa')
    parser.add_argument("--test", action="store_true", help="perform evaluation only")

    args = parser.parse_args()
    if args.cuda_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
