from typing import List

from mmseg.models.segmentors.depth_estimator import DepthEstimator
from mmseg.registry import MODELS
from mmseg.utils import SampleList, add_prefix
from torch import Tensor


@MODELS.register_module()
class CoarseAlphaEstimator(DepthEstimator):
    """Depth-estimator adapter with the standard MMEngine loss contract."""

    def _decode_head_forward_train(
        self,
        inputs: List[Tensor],
        data_samples: SampleList,
    ) -> dict:
        loss_decode = self.decode_head.loss(
            inputs, data_samples, self.train_cfg)
        return add_prefix(loss_decode, 'decode')

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        batch_img_metas = (
            [data_sample.metainfo for data_sample in data_samples]
            if data_samples is not None
            else None
        )
        features = self.extract_feat(inputs, batch_img_metas)
        losses = self._decode_head_forward_train(features, data_samples)
        if self.with_auxiliary_head:
            losses.update(
                self._auxiliary_head_forward_train(features, data_samples))
        return losses
