
import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.general import user_config_dir, is_ascii, is_chinese, xywh2xyxy, xyxy2xywh, poly2hbb
from utils.metrics import fitness

CONFIG_DIR = user_config_dir()
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')

class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

def check_font(font='Arial.ttf', size=10):
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception as e:
        url = "https://ultralytics.com/assets/" + font.name
        print(f'Downloading {url} to {font}...')
        torch.hub.download_url_to_file(url, str(font), progress=False)
        return ImageFont.truetype(str(font), size)

class Annotator:
    if RANK in (-1, 0):
        check_font()

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        if self.pil:
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                   size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:
            self.im = im
            h, w, _ = self.im.shape
            self.pts_1 = np.float32([[0.0 * w, 0], [0.5 * w, 0], [0.6 * w, 1.0 * h], [0, 1.0 * h]])
            self.pts_2 = np.float32([[0, 0], [0.33 * w, 0], [0.33 * w, 1.0 * h], [0, 1.0 * h]])
            self.M = cv2.getPerspectiveTransform(self.pts_1, self.pts_2)
            self.frame_draw = np.zeros((h, int(w + w/2), 3), dtype=np.uint8)
            self.frame_draw[:h, :w, :] = self.im
            mesh_width = 100
            mesh_height = 100
            mesh_width_num = int(w / mesh_width)
            mesh_height_num = int(h / mesh_height)
            for i in range(mesh_width_num):
                cv2.line(self.frame_draw[:, w:, :], (0, mesh_height * i), (w, mesh_height * i), (255, 255, 255), 1)
            for i in range(mesh_height_num):
                cv2.line(self.frame_draw[:, w:, :], (mesh_width * i, 0), (mesh_width * i, h), (255, 255, 255), 1)
            self.mask = np.zeros(self.im.shape, dtype='uint8')

        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = box[1] - h >= 0
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                outside = p1[1] - h - 3 >= 0
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def bev_box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = box[1] - h >= 0
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.frame_draw, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                outside = p1[1] - h - 3 >= 0
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.frame_draw, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.frame_draw, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def polygon_label(self, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        if self.pil or not is_ascii(label):
            polygon = [int(x_tl), int(y_tl), int(x_tr), int(y_tr), int(x_br), int(y_br), int(x_bl), int(y_bl)]
            self.draw.polygon(polygon, outline=color)
        else:
            polygon = np.array([[int(x_tl), int(y_tl)], [int(x_tr), int(y_tr)], [int(x_br), int(y_br)], [int(x_bl), int(y_bl)]], np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            if np.mean(polygon) == 0:
                return 
            cv2.polylines(self.im, [polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
    
    def polygon_label_3d(self, xyxy, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        if self.pil or not is_ascii(label):
            polygon = [x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl]
            self.draw.polygon(polygon, outline=color)
        else:
            bbox_y_lt = xyxy[1]
            height = min(y_tl - bbox_y_lt, y_tr - bbox_y_lt)
            polygon = np.array([[x_tl.cpu(), y_tl.cpu()], [x_tr.cpu(), y_tr.cpu()], [x_br.cpu(), y_br.cpu()], [x_bl.cpu(), y_bl.cpu()]], np.int32)
            if np.mean(polygon) == 0:
                return 
            polygon = polygon.reshape((-1, 1, 2))
            upper_polygon = np.array([[x_tl.cpu(), (y_tl - height).cpu()], [x_tr.cpu(), (y_tr - height).cpu()], [x_br.cpu(), (y_br - height).cpu()], [x_bl.cpu(), (y_bl - height).cpu()]], np.int32)
            upper_polygon = upper_polygon.reshape((-1 , 1, 2))
            cv2.polylines(self.im, [polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.polylines(self.im, [upper_polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.im, np.array([x_tl.cpu(), (y_tl - height).cpu()], np.int32), np.array([x_tl.cpu(), y_tl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.im, np.array([x_tr.cpu(), (y_tr - height).cpu()], np.int32), np.array([x_tr.cpu(), y_tr.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.im, np.array([x_br.cpu(), (y_br - height).cpu()], np.int32), np.array([x_br.cpu(), y_br.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.im, np.array([x_bl.cpu(), (y_bl - height).cpu()], np.int32), np.array([x_bl.cpu(), y_bl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 

    def polygon_label_3d_8points(self, xyxy, points, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7 = points
        if self.pil or not is_ascii(label):
            polygon = points
            self.draw.polygon(polygon, outline=color)
        else:
            polygon = np.array([[x_4.cpu(), y_4.cpu()], [x_5.cpu(), y_5.cpu()], [x_6.cpu(), y_6.cpu()], [x_7.cpu(), y_7.cpu()]], np.int32)
            if np.mean(polygon) == 0:
                return 
            polygon = polygon.reshape((-1, 1, 2))
            upper_polygon = np.array([[x_0.cpu(), y_0.cpu()], [x_1.cpu(), y_1.cpu()], [x_2.cpu(), y_2.cpu()], [x_3.cpu(), y_3.cpu()]], np.int32)
            upper_polygon = upper_polygon.reshape((-1 , 1, 2))
            cv2.putText(self.im, '0', np.array([x_0.cpu(), (y_0 - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.im, '1', np.array([x_1.cpu(), (y_1 - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.im, '2', np.array([x_2.cpu(), (y_2 - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.im, '3', np.array([x_3.cpu(), (y_3 - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.im, '4', np.array([x_4.cpu(), (y_4 - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.im, '5', np.array([x_5.cpu(), (y_5 - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.im, '6', np.array([x_6.cpu(), (y_6 - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.im, '7', np.array([x_7.cpu(), (y_7 - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.polylines(self.im, [polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.polylines(self.im, [upper_polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.im, np.array([x_0.cpu(), y_0.cpu()], np.int32), np.array([x_4.cpu(), y_4.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.im, np.array([x_1.cpu(), y_1.cpu()], np.int32), np.array([x_5.cpu(), y_5.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.im, np.array([x_2.cpu(), y_2.cpu()], np.int32), np.array([x_6.cpu(), y_6.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.im, np.array([x_3.cpu(), y_3.cpu()], np.int32), np.array([x_7.cpu(), y_7.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 

    def polygon_mask_3d_8points(self, xyxy, points, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7 = points
        if self.pil or not is_ascii(label):
            polygon = points
            self.draw.polygon(polygon, outline=color)
        else:
            polygon = np.array([[x_4.cpu(), y_4.cpu()], [x_5.cpu(), y_5.cpu()], [x_6.cpu(), y_6.cpu()], [x_7.cpu(), y_7.cpu()]], np.int32)
            if np.mean(polygon) == 0:
                return 
            polygon = polygon.reshape((-1, 1, 2))
            upper_polygon = np.array([[x_0.cpu(), y_0.cpu()], [x_1.cpu(), y_1.cpu()], [x_2.cpu(), y_2.cpu()], [x_3.cpu(), y_3.cpu()]], np.int32)
            upper_polygon = upper_polygon.reshape((-1 , 1, 2))
            left_polygon = np.array([[x_0.cpu(), y_0.cpu()], [x_3.cpu(), y_3.cpu()], [x_7.cpu(), y_7.cpu()], [x_4.cpu(), y_4.cpu()]], np.int32)
            left_polygon = left_polygon.reshape((-1 , 1, 2))
            right_polygon = np.array([[x_1.cpu(), y_1.cpu()], [x_2.cpu(), y_2.cpu()], [x_6.cpu(), y_6.cpu()], [x_5.cpu(), y_5.cpu()]], np.int32)
            right_polygon = right_polygon.reshape((-1 , 1, 2))
            front_polygon = np.array([[x_3.cpu(), y_3.cpu()], [x_2.cpu(), y_2.cpu()], [x_6.cpu(), y_6.cpu()], [x_7.cpu(), y_7.cpu()]], np.int32)
            front_polygon = front_polygon.reshape((-1, 1, 2))
            back_polygon = np.array([[x_0.cpu(), y_0.cpu()], [x_1.cpu(), y_1.cpu()], [x_5.cpu(), y_5.cpu()], [x_4.cpu(), y_4.cpu()]], np.int32)
            back_polygon = back_polygon.reshape((-1, 1, 2))

            cv2.fillPoly(self.mask, [polygon], color)
            cv2.fillPoly(self.mask, [upper_polygon], color)
            cv2.fillPoly(self.mask, [left_polygon], color)
            cv2.fillPoly(self.mask, [front_polygon], color)
            cv2.fillPoly(self.mask, [back_polygon], color)

    def draw_mask(self):
        self.im =cv2.addWeighted(self.im, 1.0, self.mask, 0.5, 0)

    def polygon_label_4points(self,xyxy,points,color,labelcls,rect=True):
        txt_color = (0, 0, 255)
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for i in range(4):
            cv2.putText(self.im, str(i), (int(points[2 * i])-2, int(points[2 * i + 1])-2), 0, min(self.lw / 3,1), txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.circle(self.im, (int(points[2 * i]), int(points[2 * i + 1])), 3, color,-1)
        if rect:
            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)

    def order_points_old(self, pts):
          rect = np.zeros((4, 2), dtype="int")
          s = pts.sum(axis=1)
          rect[0] = pts[np.argmin(s)]
          rect[2] = pts[np.argmax(s)]
          diff = np.diff(pts, axis=1)
          rect[1] = pts[np.argmin(diff)]
          rect[3] = pts[np.argmax(diff)]
          return rect

    def rotate_bev(self, img, xy, degrees):
        height = img.shape[0]
        width = img.shape[1]
        R = np.eye(3)
        if degrees < -90:
            a = -(90 + degrees)
        else:
            a = -(90 + degrees)
        print(a)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width/2, height/2), scale=1)
        M = R
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
        xy_tmp = np.ones((4, 3))
        xy_tmp[0, 0] = xy[0][0]
        xy_tmp[0, 1] = xy[0][1]
        xy_tmp[1, 0] = xy[1][0]
        xy_tmp[1, 1] = xy[1][1]
        xy_tmp[2, 0] = xy[2][0]
        xy_tmp[2, 1] = xy[2][1]
        xy_tmp[3, 0] = xy[3][0]
        xy_tmp[3, 1] = xy[3][1]
        xy_tmp = xy_tmp @ M.T 
        xy_tmp[0, 0] = np.clip(xy_tmp[0][0], 0, width)
        xy_tmp[0, 1] = np.clip(xy_tmp[0][1], 0, height)
        xy_tmp[1, 0] = np.clip(xy_tmp[1][0], 0, width)
        xy_tmp[1, 1] = np.clip(xy_tmp[1][1], 0, height)
        xy_tmp[2, 0] = np.clip(xy_tmp[2][0], 0, width)
        xy_tmp[2, 1] = np.clip(xy_tmp[2][1], 0, height)
        xy_tmp[3, 0] = np.clip(xy_tmp[3][0], 0, width)
        xy_tmp[3, 1] = np.clip(xy_tmp[3][1], 0, height)
        xy = np.array(xy_tmp[:, :2].reshape(4, 2)).astype(np.float32)
        return xy

    def vector_space_label(self, xyxy, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, bev_angle, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        if self.pil or not is_ascii(label):
            polygon = [x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl]
            self.draw.polygon(polygon, outline=color)
        else:
            bbox_y_lt = xyxy[1]
            height = min(y_tl - bbox_y_lt, y_tr - bbox_y_lt)
            polygon = np.array([[x_tl.cpu(), y_tl.cpu()], [x_tr.cpu(), y_tr.cpu()], [x_br.cpu(), y_br.cpu()], [x_bl.cpu(), y_bl.cpu()]], np.int32)
            if np.mean(polygon) == 0:
                return 0
            polygon = polygon.reshape((-1, 1, 2))
            upper_polygon = np.array([[x_tl.cpu(), (y_tl - height).cpu()], [x_tr.cpu(), (y_tr - height).cpu()], [x_br.cpu(), (y_br - height).cpu()], [x_bl.cpu(), (y_bl - height).cpu()]], np.int32)
            upper_polygon = upper_polygon.reshape((-1 , 1, 2))
            polygon_center = np.mean(polygon.reshape((-1, 2)), 0)
            orientate_point = np.array([((x_tl + x_tr)/ 2).cpu(), ((y_tl + y_tr) /2).cpu()])
            cv2.polylines(self.frame_draw, [polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.polylines(self.frame_draw, [upper_polygon], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.line(self.frame_draw, np.array([x_tl.cpu(), (y_tl - height).cpu()], np.int32), np.array([x_tl.cpu(), y_tl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.frame_draw, np.array([x_tr.cpu(), (y_tr - height).cpu()], np.int32), np.array([x_tr.cpu(), y_tr.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.frame_draw, np.array([x_br.cpu(), (y_br - height).cpu()], np.int32), np.array([x_br.cpu(), y_br.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.frame_draw, np.array([x_bl.cpu(), (y_bl - height).cpu()], np.int32), np.array([x_bl.cpu(), y_bl.cpu()], np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.line(self.frame_draw, np.array(polygon_center, np.int32), np.array(orientate_point, np.int32), color, thickness=self.lw, lineType=cv2.LINE_AA) 
            cv2.putText(self.frame_draw, label, np.array([x_tl.cpu(), (y_tl - height - 2).cpu()], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)
            polygon = polygon.astype(np.float32)

            birds_eye_polygon = cv2.perspectiveTransform(polygon, self.M)
            birds_eye_polygon[:, :, 0] = birds_eye_polygon[:, :, 0] * 0.5 + self.im.shape[1]
            birds_eye_polygon[:, :, 1] = birds_eye_polygon[:, :, 1]
            birds_eye_polygon = birds_eye_polygon.astype(np.int32)
            rect = cv2.minAreaRect(birds_eye_polygon)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = box.reshape((-1, 1, 2))
            cv2.polylines(self.frame_draw, [box], True, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.putText(self.frame_draw, label, np.array([box[0][0][0], box[0][0][1] - 2], np.int32), 0, self.lw / 3, txt_color, thickness=self.lw, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        w, h = self.font.getsize(text)
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        return np.asarray(self.im)
    
    def bev_result(self):
        return np.asarray(self.frame_draw)

def hist2d(x, y, n=100):
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    from scipy.signal import butter, filtfilt

    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

def output_to_target_keypoints(output):
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(np.array([x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl]))])
    return np.array(targets)

def output_to_target_ssod(output):
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls, obj_conf, cls_conf in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, obj_conf, cls_conf])
    return np.array(targets)

def output_to_target(output):
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)

def plot_images_debug(images, targets, score=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255.0
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        if i == max_subplots:
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6
            conf = None if labels else ti[:, 6]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale < 1:
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    if score:
        annotator.text(( 5, 30), text=score, txt_color=(220, 220, 220))

    annotator.im.save(fname)
   
def plot_images_keypoints(images, targets, paths=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255.0
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        if i == max_subplots:
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(poly2hbb(ti[:, -8:])).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 14
            conf = None if labels else ti[:, 6]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale < 1:
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)

def plot_images_ssod(images, targets, paths=None, fname='images.jpg', num_points=0, names=None,  max_size=1920, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255.0
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        if i == max_subplots:
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = False
            conf = ti[:, 6]
            obj_conf = ti[:, 7]
            cls_conf = ti[:, 8]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale < 1:
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y

            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.1:
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f} {obj_conf[j]:.1f} {cls_conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)
    
def plot_images(images, targets, paths=None, fname='images.jpg', num_points=0, names=None,  max_size=1920, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255.0
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        if i == max_subplots:
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == (6 + num_points * 2)
            conf = None if labels else ti[:, 6]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale < 1:
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y

            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.1:
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)

def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    optimizer, scheduler = copy(optimizer), copy(scheduler)
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()

def plot_val_txt():
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)

def plot_targets_txt():
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)

def plot_val_study(file='', dir='', x=None):
    save_dir = Path(file).parent if file else Path(dir)
    plot2 = False
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in sorted(save_dir.glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
                ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[5, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(25, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    f = save_dir / 'study.png'
    print(f'Saving {f}...')
    plt.savefig(f, dpi=300)

def plot_labels(labels, names=(), save_dir=Path('')):
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:5].transpose()
    nc = int(c.max() + 1)
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    matplotlib.use('svg')
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    labels[:, 1:3] = 0.5
    labels[:, 1:5] = xywh2xyxy(labels[:, 1:5]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000, :5]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]
            n = results.shape[1]
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))
    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)

def plot_evolve(evolve_csv='path/to/evolve.csv'):
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    f = evolve_csv.with_suffix('.png')
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')

def plot_results(file='path/to/results.csv', dir=''):
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j]
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()

def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape
        if height > 1 and width > 1:
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)
            n = min(n, channels)
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())
                ax[i].axis('off')

            print(f'Saving {save_dir / f}... ({n}/{channels})')
            plt.savefig(save_dir / f, dpi=300, bbox_inches='tight')
            plt.close()

def plot_single_anchor_single_feature(img_show, feature, feature_index, anchor_index, save_dir):
     (img_c, img_h, img_w) =  img_show.shape
     total_show = np.zeros((img_h, img_w, img_c))
     total_show[:,:, anchor_index] = feature
     plt.figure()
     img_tmp = np.transpose(img_show, (1, 2, 0)) * 0.5 + total_show * 255
     img_tmp = np.minimum(img_tmp, 255).astype(np.uint8)
     plt.imshow(img_tmp)
     plt.savefig(str(save_dir) + '/feature_' + str(feature_index) + '_' + str(anchor_index) + '.png')

def feature_vis(pred, img_show, save_dir):
        pred_debug = np.asarray(pred.cpu(), dtype=np.float32)
        (img_c, img_h, img_w) =  img_show.shape
        total_show = np.zeros((img_h, img_w, img_c))
        w = [int(img_w/8), int(img_w/16), int(img_w/32)]
        h = [int(img_h/8), int(img_h/16), int(img_h/32)]
        index_range  = [w[0] * h[0], w[1] * h[1], w[2]* h[2]]
        index_start = [0, w[0]* h[0] * 3, (w[0] * h[0] + w[1] * h[1]) * 3]
        print('img_show_shape:', img_show.shape)
        print('w', w)
        print('h', h)
        print('index_range', index_range)
        print('index_start', index_start)
        for i in range(3):
            tmp_show = np.zeros((img_h, img_w))
            feature_0 = pred_debug[0, index_start[i]:index_start[i] + index_range[i], 4].reshape(h[i], w[i])
            feature_0 = cv2.resize(feature_0, (img_w, img_h), interpolation = cv2.INTER_NEAREST)
            plot_single_anchor_single_feature(img_show, feature_0, i, 0, save_dir)
            tmp_show = np.maximum(feature_0, tmp_show)
            print('start:', index_start[i] + index_range[i])
            print('end:', index_start[i] + index_range[i] * 2)
            print('pred_debug:',pred_debug[0, (index_start[i] + index_range[i]):(index_start[i] + index_range[i] * 2), 4].shape )
            feature_1 = pred_debug[0, (index_start[i] + index_range[i]):(index_start[i] + index_range[i] * 2), 4].reshape(h[i], w[i])
            feature_1 = cv2.resize(feature_1, (img_w, img_h), interpolation = cv2.INTER_NEAREST)
            plot_single_anchor_single_feature(img_show, feature_1, i, 1, save_dir)
            tmp_show = np.maximum(feature_1, tmp_show)
            feature_2 = pred_debug[0, (index_start[i] + index_range[i] * 2) : (index_start[i] + index_range[i] * 3), 4].reshape(h[i], w[i])
            feature_2 = cv2.resize(feature_2, (img_w, img_h), interpolation = cv2.INTER_NEAREST)
            plot_single_anchor_single_feature(img_show, feature_2, i, 2, save_dir)
            tmp_show = np.maximum(feature_2, tmp_show)
            total_show[:,:, 0] = np.maximum(total_show[:,:,0], tmp_show)

        plt.figure()
        img_show = np.transpose(img_show, (1, 2, 0)) * 0.5 + total_show * 255
        img_show = np.minimum(img_show, 255).astype(np.uint8)
        plt.imshow(img_show) 
        plt.savefig(save_dir + 'total.png')
