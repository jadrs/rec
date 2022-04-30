import numpy as np

import torch

import gzip

import json

from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm

import time

from datetime import datetime

from colorama import Fore, Style, init
init()


__color_table__ = {
    None: Style.RESET_ALL,
    "red": Fore.LIGHTRED_EX,
    "green": Fore.LIGHTGREEN_EX,
    "blue": Fore.LIGHTBLUE_EX,
}


def cprint(*parg, **kwargs):
    color = kwargs["color"] if "color" in kwargs else None
    print(__color_table__[color], end="")
    print(*parg, end="")
    print(Style.RESET_ALL)


def timestamp():
    fmt = "%Y%m%d_%H%M%S"
    return datetime.now().strftime(fmt)


def hms():
    return time.strftime("%H:%M:%S", time.gmtime(time.time()))


def progressbar(x, **kwargs):
    return tqdm(x, ascii=True, **kwargs)


def draw_bounding_boxes(img, bboxes, labels=None, fmt="xywh",
                        color=(223, 223, 0)):
    assert fmt in ("xywh", "xyxy")
    line_width = 1
    fnt_size = 8
    # fnt = ImageFont.truetype("arial.ttf", fnt_size)
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', fnt_size)

    draw = ImageDraw.Draw(img)
    for i, bbox in enumerate(bboxes):
        if fmt == "xywh":
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1]
        draw.rectangle(bbox, fill=None, outline=color, width=line_width)
        if labels is None:
            continue
        lbl = labels[i]
        x, y = bbox[0]+1, bbox[1]+1
        w, h = fnt.getsize(lbl)
        draw.rectangle((x, y, x + w, y + h), fill=color)
        draw.text((x, y), lbl, font=fnt, fill="black")
    del draw

    return img


def load_data(jsonl_file):
    data = []
    with gzip.open(jsonl_file, "rb") as fin:
        for line in fin:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            data.append(game)
    return data


def save_data(data, jsonl_file):
    with gzip.open(jsonl_file, "wb") as fout:
        for x in data:
            json_bytes = (json.dumps(x) + "\n").encode("utf-8")
            fout.write(json_bytes)


def game2image(data, game_id):
    return [game["image"]["id"] for game in data if game["id"] == game_id][0]


def image2game(data, image_id):
    return [game["id"] for game in data if game["image"]["id"] == image_id][0]


def xyxy2xywh(boxes, inplace=False, as_int=False):
    """Convert boxes format: (x1, y1, x2, y2) -> (x, y, w, h)

    Args:
      boxes: input boxes in (x1, y1, x2, y2) format
      inplace: if True, replace input boxes with their converted versions
      as_int: if True, interpret the input as integer coordinates (takes into
              account the +1 offset when computing the box width and height)
    Returns:
      boxes in (x, y, w, h) format
    """
    assert (
        (isinstance(boxes, np.ndarray) or torch.is_tensor(boxes))
        and boxes.ndim == 2
        and boxes.shape[1] == 4
    )
    if not inplace:
        boxes = boxes.clone() if torch.is_tensor(boxes) else boxes.copy()
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0] + int(as_int)
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1] + int(as_int)
    return boxes


def xywh2xyxy(boxes, inplace=False, as_int=False):
    """convert boxes format: (x, y, w, h) -> (x1, y1, x2, y2)

    Args:
      boxes: input boxes in (x, y, w, h) format
      inplace: if True, replace input boxes with their converted versions
      as_int: if True, interpret the input as integer coordinates (takes into
              account the -1 offset when computing the box x2 and y2 coords)
    Returns:
      boxes in (x1, y1, x2, y2) format
    """
    assert (
        (isinstance(boxes, np.ndarray) or torch.is_tensor(boxes))
        and boxes.ndim == 2
        and boxes.shape[1] == 4
    )
    if not inplace:
        boxes = boxes.clone() if torch.is_tensor(boxes) else boxes.copy()
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - int(as_int)
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - int(as_int)
    return boxes
