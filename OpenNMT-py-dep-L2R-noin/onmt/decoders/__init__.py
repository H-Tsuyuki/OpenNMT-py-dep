"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase
from onmt.decoders.transformer import TransformerDecoder, TransformerDecoderLayer


str2dec = {"transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "str2dec", "TransformerDecoderLayer"]
