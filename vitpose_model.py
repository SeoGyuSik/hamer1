from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result

os.environ["PYOPENGL_PLATFORM"] = "egl"

# project root directory
ROOT_DIR = "./"
VIT_DIR = os.path.join(ROOT_DIR, "third-party/ViTPose")

class ViTPoseModel(object):
    MODEL_DICT = {
        'ViTPose+-G (multi-task train, COCO)': {
            'config': f'{VIT_DIR}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py',
            'model': f'{ROOT_DIR}/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth',
        },
    }

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model_name = 'ViTPose+-G (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model']
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            images: list[np.ndarray],
            det_results: list[list[np.ndarray]],
            box_score_threshold: float = 0.5) -> list[list[dict[str, np.ndarray]]]:
        # Process each image in the batch
        all_person_results = process_mmdet_results_batch(det_results)
        
        # Inference poses for the batch
        pose_results, _ = inference_top_down_pose_model_batch(
            self.model,
            images,
            all_person_results=all_person_results,
            bbox_thr=box_score_threshold,
            format='xyxy')
        
        return pose_results

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB

        self.model_name = 'ViTPose+-G (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model']
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            images: list[np.ndarray],
            det_results: list[list[np.ndarray]],
            box_score_threshold: float = 0.5) -> list[list[dict[str, np.ndarray]]]:
        # Process each image in the batch
        all_person_results = process_mmdet_results(det_results)

        #for idx, person_results in enumerate(all_person_results):
        #    print(f"Image {idx}: Processed person results: {person_results}")
        
        # Inference poses for the batch
        pose_results, _ = inference_top_down_pose_model(
            self.model,
            images,
            all_person_results=all_person_results,
            bbox_thr=box_score_threshold,
            format='xyxy')
        
        return pose_results

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB
