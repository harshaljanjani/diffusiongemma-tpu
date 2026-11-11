import pytest
import torch
from transformers import (
    DiffusionGemmaConfig,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaTextConfig,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
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


def build_tiny_gemma4(sliding_window: int = 24, vocab_size: int = 512, seed: int = 0):
    torch.manual_seed(seed)
    text_config = Gemma4TextConfig(
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
        enable_moe_block=True,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=vocab_size,
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16
    )
    config = Gemma4Config(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=None,
        boi_token_id=vocab_size - 3,
        eoi_token_id=vocab_size - 2,
        image_token_id=vocab_size - 1,
    )
    model = Gemma4ForConditionalGeneration(config).eval()
    model.generation_config.eos_token_id = 1
    model.generation_config.pad_token_id = 0
    return model


@pytest.fixture(scope="session")
def tiny_gemma4_factory():
    return build_tiny_gemma4
