
import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import MetadataCatalog
from omegaconf import OmegaConf
import numpy as np
import cv2

class DefaultPredictor_Lazy:
    def __init__(self, cfg):
        if isinstance(cfg, CfgNode):
            self.cfg = cfg.clone()
            self.model = build_model(self.cfg)  # noqa: F821
            if len(cfg.DATASETS.TEST):
                test_dataset = cfg.DATASETS.TEST[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

            self.input_format = cfg.INPUT.FORMAT
        else:  # new LazyConfig
            self.cfg = cfg
            self.model = instantiate(cfg.model)
            test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
            if isinstance(test_dataset, (list, tuple)):
                test_dataset = test_dataset[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

            mapper = instantiate(cfg.dataloader.test.mapper)
            self.aug = mapper.augmentations
            self.input_format = mapper.image_format

        self.model.eval().cuda()
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_images):
        with torch.no_grad():
            if not isinstance(original_images, list):
                original_images = [original_images]
                
            processed_images = []
            original_sizes = []

            # Determine the maximum height and width in the batch
            max_height = max([img.shape[0] for img in original_images])
            max_width = max([img.shape[1] for img in original_images])

            for original_image in original_images:
                if self.input_format == "RGB":
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                original_sizes.append((height, width))

                # Resize images to the max height and width
                resized_image = cv2.resize(original_image, (max_width, max_height))
                
                # Apply augmentation
                image = self.aug(T.AugInput(resized_image)).apply_image(resized_image)
                
                # Convert to tensor
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                processed_images.append(image)
            
            # Stack images into a single batch tensor
            images_tensor = torch.stack(processed_images).cuda()
            inputs = [{"image": img, "height": h, "width": w} for img, (h, w) in zip(images_tensor, original_sizes)]
            
            # Perform batch inference
            predictions = self.model(inputs)

            #for idx, pred in enumerate(predictions):
            #    print(f"Image {idx}: Detection results (boxes): {pred['instances'].pred_boxes.tensor.cpu().numpy()}")
            #    print(f"Image {idx}: Detection results (scores): {pred['instances'].scores.cpu().numpy()}")
            
            # Adjust predictions back to original sizes
            for pred, (orig_h, orig_w), (new_h, new_w) in zip(predictions, original_sizes, [(max_height, max_width)] * len(original_sizes)):
                scale_x = orig_w / new_w
                scale_y = orig_h / new_h
                pred["instances"].pred_boxes.tensor[:, [0, 2]] *= scale_x
                pred["instances"].pred_boxes.tensor[:, [1, 3]] *= scale_y

            return predictions

# Example usage:
# cfg = ...  # Load or create a Detectron2 config
# predictor = DefaultPredictor_Lazy(cfg)
# batch_images = [cv2.imread("input1.jpg"), cv2.imread("input2.jpg")]
# outputs = predictor(batch_images)  # Process a batch of images