import json
import multiprocessing
import os
import io
import threading

import torch

from PIL import Image

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

lock = threading.Lock()

def init_predictor_and_thing_classes():
    DATASET_NAME = "test"
    DATASET_DIR = "dataset"

    register_coco_instances(DATASET_NAME, {}, os.path.join(DATASET_DIR, "coco-1608066283.6559396.json"), DATASET_DIR)
    metadata = MetadataCatalog.get(DATASET_NAME)
    coco_json = load_coco_json(os.path.join(DATASET_DIR, "coco-1608066283.6559396.json"), DATASET_DIR, DATASET_NAME)

    cfg = get_cfg()
    cfg.merge_from_file("./configs/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set the testing threshold for this model
    cfg.DATASETS.TEST = (DATASET_NAME, )
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=10000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=24000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST=4000
    predictor = DefaultPredictor(cfg)

    return predictor, metadata.get("thing_classes", None)


def make_obj(thing_classes, predictions):
    obj = {
        "images": [{}],
        # "categories": [{
        #     "id": 10,
        #     "name": "horiz_label",
        # }],
        # "annotations": [{
        #     "category_id": 10,
        #     "segmentation": [
        #         [1578.4, 792.3, 1578.4, 880.5, 1467.8, 880.5, 1467.8, 792.3]
        #     ],
        #     "isbbox": True,
        # }, {
        #     "category_id": 10,
        #     "segmentation": [
        #         [1092.0, 676.5, 1092.0, 557.2, 1231.6, 557.2, 1231.6, 676.5]
        #     ],
        #     "isbbox": True,
        # }]
    }

    categories = []
    for i in range(len(thing_classes)):
        categories.append({"id": i, "name": thing_classes[i]})

    obj["categories"] = categories

    annotations = []
    instances = predictions["instances"].to(torch.device("cpu"))

    for box, cat in zip(instances.pred_boxes, instances.pred_classes):
        # print(dir(box))
        # print(box.tolist(), cat.tolist())
        x1, y1, x2, y2 = box.tolist()
        annotations.append({
            "category_id": cat.tolist(),
            "segmentation": [
                [x1, y1, x1, y2, x2, y2, x2, y1]
            ],
            "isbbox": True
        })
    obj["annotations"] = annotations

    print(obj)
    return {"coco": obj}

class HttpProcessor(BaseHTTPRequestHandler):
    def do_POST(self):
        global predictor, thing_classes

        with lock:
            data = self.rfile.read(int(self.headers["Content-Length"]))
            data = data.split(b"\r\n\r\n", 1)[1]

            data_io = io.BytesIO(data)

            img = convert_PIL_to_numpy(Image.open(data_io), format="BGR")

            predictions = predictor(img)

            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()

            self.wfile.write(json.dumps(make_obj(thing_classes, predictions)).encode())
            self.close_connection = True

multiprocessing.set_start_method("spawn", force=True)

predictor, thing_classes = init_predictor_and_thing_classes()

print("Serving")
serv = ThreadingHTTPServer(("0.0.0.0",80), HttpProcessor)
serv.serve_forever()

