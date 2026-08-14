"""TensorRT-LLM 1.2.1 compatibility wrapper for the fixed AutoReply tokenizer.

The checkpoint was saved by a newer Transformers release with
``tokenizer_class=TokenizersBackend``.  TensorRT-LLM 1.2.1 bundles an older
Transformers build whose AutoTokenizer cannot resolve that class name, even
though its PreTrainedTokenizerFast can read the exact same tokenizer.json.
"""

from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer
from transformers import PreTrainedTokenizerFast


class AutoReplyTokenizer(TransformersTokenizer):
    """Load the unchanged tokenizer.json through the compatible fast class."""

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_dir, **kwargs
        )
        return cls(tokenizer)
