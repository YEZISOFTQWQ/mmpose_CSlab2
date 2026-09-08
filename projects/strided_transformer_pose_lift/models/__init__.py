"""Register custom Strided Transformer modules with MMPose."""

from .full_to_single_head import FullToSingleRegressionHead
from .strided_transformer import StridedTransformerBackbone

__all__ = ['FullToSingleRegressionHead', 'StridedTransformerBackbone']
