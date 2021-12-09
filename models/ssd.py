#sources 
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/model.py
# Libtorch
import torch
import torch.nn.functional as F
import warnings
import sys

from functools import partial
from collections import OrderedDict
from typing import Any, Optional, Dict, List, Tuple, Callable

from torch import nn, Tensor


import torchvision.models.resnet as resnet
import torchvision.models as models

from torch.hub import load_state_dict_from_url
#from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops import boxes as box_ops
from torchvision.models import mobilenet

from .transform import GeneralizedRCNNTransform
#from . import _utils as det_utils
from torchvision.models.detection import _utils as det_utils

from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
#sys.path.append('torchvision/models/detection')
#from anchor_utils import DefaultBoxGenerator

__all__ = ['SSD', 'ssd300_vgg16', 'ssd300_resnet50', 'ssd512_resnet50']

model_urls = {
    'ssd300_vgg16_coco': 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth',
    'ssd512_resnet50_coco': 'https://download.pytorch.org/models/ssd512_resnet50_coco-d6d7edbb.pth',
}

backbone_urls = {
    # We port the features of a VGG16 backbone trained by amdegroot because unlike the one on TorchVision, it uses the
    # same input standardization method as the paper. Ref: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
    'vgg16_features': 'https://download.pytorch.org/models/vgg16_features-amdegroot.pth'
}


def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class SSDHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'bbox_regression': self.regression_head(x),
            'cls_logits': self.classification_head(x),
        }


class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.module_list[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.module_list:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))
        _xavier_init(cls_logits)
        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 4)


class SSD(nn.Module):
    """
    Implements SSD architecture from `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size before passing it to the backbone.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute with the list of the output channels of
            each feature map. The backbone should return a single Tensor or an OrderedDict[Tensor].
        anchor_generator (DefaultBoxGenerator): module that generates the default boxes for a
            set of feature maps.
        size (Tuple[int, int]): the width and height to which images will be rescaled before feeding them
            to the backbone.
        num_classes (int): number of output classes of the model (excluding the background).
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        head (nn.Module, optional): Module run on top of the backbone features. Defaults to a module containing
            a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        topk_candidates (int): Number of best detections to keep before NMS.
        positive_fraction (float): a number between 0 and 1 which indicates the proportion of positive
            proposals used during the training of the classification head. It is used to estimate the negative to
            positive ratio.
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }

    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator,
                 size: Tuple[int, int], num_classes: int,
                 image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None,
                 head: Optional[nn.Module] = None,
                 score_thresh: float = 0.01,
                 nms_thresh: float = 0.45,
                 detections_per_img: int = 200,
                 iou_thresh: float = 0.5,
                 topk_candidates: int = 400,
                 positive_fraction: float = 0.25):
        super().__init__()

        self.backbone = backbone

        self.anchor_generator = anchor_generator

        self.box_coder = det_utils.BoxCoder(weights=(10., 10., 5., 5.))

        if head is None:
            if hasattr(backbone, 'out_channels'):
                out_channels = backbone.out_channels
            else:
                out_channels = det_utils.retrieve_out_channels(backbone, size)

            assert len(out_channels) == len(anchor_generator.aspect_ratios)

            num_anchors = self.anchor_generator.num_anchors_per_location()
            head = SSDHead(out_channels, num_anchors, num_classes)
        self.head = head

        self.proposal_matcher = det_utils.SSDMatcher(iou_thresh)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min(size), max(size), image_mean, image_std,
                                                  size_divisible=1, fixed_size=size)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor],
                      detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        bbox_regression = head_outputs['bbox_regression']
        cls_logits = head_outputs['cls_logits']

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (targets_per_image, bbox_regression_per_image, cls_logits_per_image, anchors_per_image,
             matched_idxs_per_image) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image['boxes'][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(torch.nn.functional.smooth_l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ))

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros((cls_logits_per_image.size(0), ), dtype=targets_per_image['labels'].dtype,
                                            device=targets_per_image['labels'].device)
            gt_classes_target[foreground_idxs_per_image] = \
                targets_per_image['labels'][foreground_matched_idxs_per_image]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes),
            cls_targets.view(-1),
            reduction='none'
        ).view(cls_targets.size())

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float('inf')  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            'bbox_regression': bbox_loss.sum() / N,
            'classification': (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

    def forward(self, images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image['boxes'].numel() == 0:
                    matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                                   device=anchors_per_image.device))
                    continue

                match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

    def postprocess_detections(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor],
                               image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs['bbox_regression']
        pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })
        return detections

class SSDFeatureExtractorResNet(nn.Module):
    def __init__(self, backbone: resnet.ResNet):
        super().__init__()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )

        # Patch last block's strides to get valid output sizes
        for m in self.features[-1][0].modules():
            if hasattr(m, 'stride'):
                m.stride = 1

        backbone_out_channels = self.features[-1][-1].bn3.num_features

        #print(nn.Sequential(*list(backbone.children())[:7])[-1] == self.features[-1])
        #print(self.features[-1])

        assert backbone_out_channels == 1024

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        ])
        _xavier_init(extra)
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)
        output = [x]
        
        for block in self.extra:
            x = block(x)
            output.append(x)


        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def _resnet_extractor(backbone_name: str, pretrained: bool, trainable_layers: int):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
    
    #is this part really necessary?
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 4:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    


    return SSDFeatureExtractorResNet(backbone)

def ssd300_resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    #model = models.detection.ssd300_vgg16(pretrained=True, progress=True)

    size = (300, 300)

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 4, 4)

    if pretrained:
        pretrained_backbone = False

    backbone = _resnet_extractor("resnet50", pretrained_backbone, trainable_backbone_layers)

    #anchor_generator =  model.anchor_generator
    #head = model.head

    #ssd_resnet = models.detection.SSD(backbone, anchor_generator, size, num_classes, head = head, **kwargs)

    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                            steps=[8, 16, 32, 64, 100, 300])
    model = SSD(backbone, anchor_generator, size, num_classes, **kwargs)

    # if pretrained:
    #     weights_name = 'ssd300_resnet50_coco'
    #     if model_urls.get(weights_name, None) is None:
    #         raise ValueError("No checkpoint is available for model {}".format(weights_name))
    #     state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
    #     model.load_state_dict(state_dict)

    return model

class SSDFeatureExtractorMobileNet(nn.Module):
    def __init__(self, backbone: nn.Module, norm_layer: Callable[..., nn.Module], **kwargs: Any):
        super().__init__()

        self.features = nn.Sequential(*backbone[:14])

        backbone_out_channels = features[-1].conv[-1].num_features

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_out_channels, 256, kernel_size=1), # conv8_2
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), # conv9_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3), # conv10_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3), # conv11_2
                nn.ReLU(inplace=True),
            )
        ])



        # # non-public config parameters
        # min_depth = kwargs.pop('_min_depth', 16)
        # width_mult = kwargs.pop('_width_mult', 1.0)

        # assert not backbone[c4_pos].use_res_connect
        # self.features = nn.Sequential(
        #     nn.Sequential(*backbone[:c4_pos], backbone[c4_pos].block[0]),  # from start until C4 expansion layer
        #     nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1:]),  # from C4 depthwise until end
        # )

        # get_depth = lambda d: max(min_depth, int(d * width_mult))  # noqa: E731
        # extra = nn.ModuleList([
        #     _extra_block(backbone[-1].out_channels, get_depth(512), norm_layer),
        #     _extra_block(get_depth(512), get_depth(256), norm_layer),
        #     _extra_block(get_depth(256), get_depth(256), norm_layer),
        #     _extra_block(get_depth(256), get_depth(128), norm_layer),
        # ])
        # _normal_init(extra)

        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Get feature maps from backbone and extra. Can't be refactored due to JIT limitations.
        output = []
        for block in self.features:
            x = block(x)
            print(x.shape)
            output.append(x)

        for block in self.extra:
            x = block(x)
            print(x.shape)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])




def _mobilenet_extractor(backbone_name: str, progress: bool, pretrained: bool, trainable_layers: int,
                         norm_layer: Callable[..., nn.Module], **kwargs: Any):
    backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, progress=progress,
                                                 norm_layer=norm_layer, **kwargs).features
    # if not pretrained:
    #     # Change the default initialization scheme if not pretrained
    #     _normal_init(backbone)

    # # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    # stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    # num_stages = len(stage_indices)

    # print('stages: ', num_stages)
    # print(stage_indices[-2])
    # # find the index of the layer from which we wont freeze
    # assert 0 <= trainable_layers <= num_stages
    # freeze_before = num_stages if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    # for b in backbone[:freeze_before]:
    #     for parameter in b.parameters():
    #         parameter.requires_grad_(False)


    #return backbone
    return SSDFeatureExtractorMobileNet(backbone, stage_indices[-2], norm_layer, **kwargs)


def ssd300_mobilenet_v2(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                                  pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                                  norm_layer: Optional[Callable[..., nn.Module]] = None,
                                  **kwargs: Any):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6)

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected
    reduce_tail = not pretrained_backbone

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = _mobilenet_extractor("mobilenet_v3_large", progress, pretrained_backbone, trainable_backbone_layers,
                                norm_layer)

    #print(backbone)
    # size = (320, 320)
    # anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    #out_channels = det_utils.retrieve_out_channels(backbone, size)
    # num_anchors = anchor_generator.num_anchors_per_location()
    # assert len(out_channels) == len(anchor_generator.aspect_ratios)


class ResnetAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )

        # Patch last block's strides to get valid output sizes
        for m in self.features[-1][0].modules():
            if hasattr(m, 'stride'):
                m.stride = 1

        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        ])

        _xavier_init(extra)
        self.extra = extra

        _xavier_init(self.reducer)

    def forward(self, x):
        x = self.features(x)
        x = self.reducer(x)
        output = [x]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])

def ssd_resnet50_adapted(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):

    pretrained_model = models.detection.ssd300_vgg16(pretrained=True)
    pretrained_head = pretrained_model.head

    backbone = ResnetAdapter(models.resnet50())
    size = (300, 300)
    anchor_generator = pretrained_model.anchor_generator

    model = SSD(backbone, anchor_generator, size, num_classes, head = pretrained_head, **kwargs)
    return model


class ResnetAdapterV2(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.features = nn.Sequential(
            *[u for v, u in list(backbone.items())[:-1]]
        )

        for m in self.features[-1][0].modules():
            if hasattr(m, 'stride'):
                m.stride = 1


        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        ])

        _xavier_init(extra)
        self.extra = extra

        _xavier_init(self.reducer)


    def forward(self, x):
        x = self.features(x)
        x = self.reducer(x)

        output = [x]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def ssd_resnet50_adapted_v2(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):  

    #grab pre-trained retinanet model (resnet 50 backbone)
    model = models.detection.retinanet_resnet50_fpn(pretrained=True)
    
    #get the resnet backbone from this model
    pretrained_backbone = model.backbone.body

    #freeze all backbone layers except layer 3 and 4
    layers_to_train = ['layer3', 'layer4']

    for name, parameter in pretrained_backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)



    #attach ssd layers to the backbone
    backbone = ResnetAdapterV2(pretrained_backbone)
    pretrained_model = models.detection.ssd300_vgg16(pretrained=True)

    #don't freeze head. at least not yet.  
    pretrained_head = pretrained_model.head

    size = (300, 300)
    anchor_generator = pretrained_model.anchor_generator
    
    model = SSD(backbone, anchor_generator, size, num_classes, head = pretrained_head, **kwargs)

    return model
