import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import io
import torch
import sys
import pickle
import itertools
import math
import collections

from PIL import Image, ImageDraw

from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor

# from detectron2_git.demo.predictor import VisualizationDemo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog

def get_map_labels_img(instances, visualized_output, metadata):
    def box_center(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def box_distance(b1, b2):
        b1_center = box_center(b1)
        b2_center = box_center(b2)
        return math.sqrt((b1_center[0] - b2_center[0])**2 + (b1_center[1] - b2_center[1])**2)

    def intersect_box_fraction(b1, b2):
        dx = min(b1[2], b2[2]) - max(b1[0], b2[0])
        dy = min(b1[3], b2[3]) - max(b1[1], b2[1])
        if (dx>=0) and (dy>=0):
            b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
            if b1_area == 0 or b2_area == 0:
                return 0
            intersect_area = dx*dy
            return max(intersect_area / b1_area,
                       intersect_area / b2_area)
        return 0

    thing_classes = metadata.get("thing_classes", None)

    classes = instances.get("pred_classes")
    boxes = instances.get("pred_boxes")
    scores = instances.get("scores")

    element_boxes = []
    label_boxes = []
    for box, cl in zip(boxes, classes):
        cl_name = thing_classes[cl.item()]
        box_as_tuple = tuple(box.tolist())
        if cl_name in ["horiz_label", "vert_label"]:
            label_boxes.append(box_as_tuple)
        else:
            element_boxes.append(box_as_tuple)

    rest_labels = set(label_boxes)

    MAX_DISTANCE = 500

    pairs_candidates = []
    for el_box in element_boxes:
        for lbl_box in label_boxes:
            d = box_distance(el_box, lbl_box)
            if d <= MAX_DISTANCE:
                pairs_candidates.append((el_box, lbl_box, d))
    pairs_candidates.sort(key=lambda e: e[2])

    img = Image.fromarray(visualized_output.img)
    draw = ImageDraw.Draw(img)

    bad_boxes = set()

    # calc bad boxes
    MAX_INTERSECT_THRESHOLD = 0.7
    for b1, b2 in itertools.combinations(element_boxes, 2):
        if intersect_box_fraction(b1, b2) > MAX_INTERSECT_THRESHOLD:
            b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])

            if b1_area > b2_area:
                bad_boxes.add(b1)
            else:
                bad_boxes.add(b2)

    for box in element_boxes:
        if box in bad_boxes:
            continue
        draw.rectangle(box, outline="blue")
    for box in label_boxes:
        if box in bad_boxes:
            continue
        draw.rectangle(box, outline="green")

    used_lbls = set()
    used_els = set()
    for el_box, lbl_box, d in pairs_candidates:
        if el_box in bad_boxes or lbl_box in bad_boxes:
            continue

        if lbl_box in used_lbls:
            continue
        if el_box in used_els:
            continue

        used_lbls.add(lbl_box)
        used_els.add(el_box)

        draw.line([box_center(el_box), box_center(lbl_box)], fill="red")
    return img


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file("configs/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_list(["MODEL.WEIGHTS", "model_final.pth"])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.33
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.33
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.33
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.33
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.33
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.33
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=10000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=48000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST=8000

    cfg.freeze()
    return cfg


mp.set_start_method("spawn", force=True)
cfg = setup_cfg()
predictor = DefaultPredictor(cfg)
# demo = VisualizationDemo(cfg)

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

TOKEN = "1477429159:AAGynG19D6k6O6I4soNdku7hvmKeLe_Wrlc"


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Вышлите фотку, а я поищу объекты')


def answer(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f"Вышлите фотку, а я поищу объекты")


def photo_answer(update: Update, context: CallbackContext) -> None:
    global metadata
    global predictor
    print("Got photo")
    content = update.message.photo[-1].get_file(timeout=10).download_as_bytearray()

    img = convert_PIL_to_numpy(Image.open(io.BytesIO(content)), format="BGR")

    predictions = predictor(img)
    img = img[:, :, ::-1]

    visualizer = Visualizer(img, metadata, instance_mode=ColorMode.IMAGE)
    visualizer._default_font_size = 6
    instances = predictions["instances"].to(torch.device("cpu"))

    # predictions, visualized_output = demo.run_on_image(img)

    print(predictions)

    # out_image = visualized_output.get_image()
    instances = predictions["instances"].to(torch.device("cpu"))
    visualized_output = visualizer.draw_instance_predictions(predictions=instances)

    out_img_io = io.BytesIO()
    visualized_output.save(out_img_io)

    out_img_io_read = io.BytesIO(out_img_io.getvalue())
    out_img_io_read.name = "out.jpg"

    update.message.reply_photo(out_img_io_read)
    print(f"MSG photo {update.message.photo}", file=sys.stderr)

    img2 = get_map_labels_img(instances, visualized_output, metadata)

    out_img2_io = io.BytesIO()
    out_img2_io.name = "out.png"
    img2.save(out_img2_io)

    out_img2_io_read = io.BytesIO(out_img2_io.getvalue())
    out_img2_io_read.name = "out.png"

    update.message.reply_photo(out_img2_io_read)
    print(f"MSG tg2 sent")


DATASET_NAME = "test"
DATASET_DIR = "dataset"

register_coco_instances(DATASET_NAME, {"dataset_name": DATASET_NAME}, os.path.join(DATASET_DIR, "coco-1608066283.6559396.json"), DATASET_DIR)
metadata = MetadataCatalog.get(DATASET_NAME)

# print(dir(metadata))

coco_json = load_coco_json(os.path.join(DATASET_DIR, "coco-1608066283.6559396.json"), DATASET_DIR, DATASET_NAME)
print(metadata.get("thing_classes", None))


updater = Updater(TOKEN, use_context=True)
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler("start", start))

dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, answer))
dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo_answer))
print("starting")
updater.start_polling()
updater.idle()
