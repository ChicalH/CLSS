# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead,DAFormerHead_shareproto
from .cp_head import CPHead_for_hrda
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .hrda_head_cp import HRDAHead_cp
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'SegFormerHead',
    'CPHead_for_hrda',
    'DAFormerHead_shareproto',
    'DAFormerHead',
    'ISAHead',
    'HRDAHead_cp',
]
