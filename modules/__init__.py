'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

from .affine import Biaffine
from .affine import Triaffine
from .mlp import MLP, NonLinear
from .ScaleMix import ScalarMix
from .soft_mask import max_min_scale
# from structs.tree import (DependencyCRF, MatrixTree)
from .lstm import CharLSTM, VariationalLSTM
from .dropout import IndependentDropout, SharedDropout, TokenDropout
from .pretrained import ELMoEmbedding, TransformerEmbedding
from .transformer import (TransformerDecoder, TransformerEncoder,
                          TransformerWordEmbedding)
__all__ = [
    'Biaffine',
    'Triaffine',
    'IndependentDropout',
    'SharedDropout',
    'TokenDropout',
    'CharLSTM',
    'VariationalLSTM',
    'MLP',
    'ELMoEmbedding',
    'TransformerEmbedding',
    'TransformerWordEmbedding',
    'TransformerDecoder',
    'TransformerEncoder',
    'NonLinear',
    'ScalarMix',
    'max_min_scale'
]
