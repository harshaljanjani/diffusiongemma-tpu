import pytest
import torch
from transformers import (
    DiffusionGemmaConfig,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaTextConfig,
    Gemma4VisionConfig,
)


def build_tiny_model(sliding_window: int = 8, canvas_length: int = 16, vocab_size: int = 512, seed: int = 0):
    torch.manual_seed(seed)
    text_config = DiffusionGemmaTextConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=sliding_window,
        max_position_embeddings=1024,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16
    )
    config = DiffusionGemmaConfig(
        text_config=text_config,
        vision_config=vision_config,
        canvas_length=canvas_length,
        boi_token_id=vocab_size - 3,
        eoi_token_id=vocab_size - 2,
        image_token_id=vocab_size - 1,
    )
    return DiffusionGemmaForBlockDiffusion(config).eval()


@pytest.fixture(scope="session")
def tiny_model_factory():
    return build_tiny_model
