#!/usr/bin/env python3
"""Test script to verify model save and load functionality."""

import torch
import tempfile
import shutil
from transformers import GPT2Config
from model import GPT2ForLongshot, GPT2ForLongshotConfig


def test_save_and_load():
    """Test that we can save and load a GPT2ForLongshot model."""
    
    # Create a test configuration
    n = 3
    k = 2
    
    gpt2_config = GPT2Config(
        vocab_size=1,
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    
    config = GPT2ForLongshotConfig(
        num_vars=n,
        width=k,
        n_embed_lit=8,
        ub_q=3.0,
        alpha=1.0,
        beta=20.0,
        gamma=1.0,
        share_semantic=False,
        universal=False,
        gpt2_config=gpt2_config
    )
    
    # Create model
    print("Creating original model...")
    model = GPT2ForLongshot(config)
    model.eval()  # Set to eval mode to disable dropout
    
    # Create some test input
    batch_size = 2
    seq_len = 10
    test_input = torch.randint(-100, 100, (batch_size, seq_len))
    
    # Get output from original model
    print("Running forward pass on original model...")
    with torch.no_grad():
        original_output = model(test_input, return_dict=True)
        original_logits = original_output.logits.clone()
    
    # Save model to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Saving model to {temp_dir}...")
        model.save_pretrained(temp_dir)
        
        # Check that files were created
        import os
        files = os.listdir(temp_dir)
        print(f"Files saved: {files}")
        assert "config.json" in files, "config.json not found"
        assert "model.safetensors" in files or "pytorch_model.bin" in files, "Model weights not found"
        
        # Load model from saved directory
        print(f"Loading model from {temp_dir}...")
        loaded_model = GPT2ForLongshot.from_pretrained(temp_dir)
        loaded_model.eval()  # Set to eval mode to disable dropout
        
        # Compare ALL weights to debug
        print("\nComparing all weights...")
        all_weights_match = True
        for name, param in model.named_parameters():
            loaded_param = dict(loaded_model.named_parameters())[name]
            weight_diff = torch.max(torch.abs(param - loaded_param)).item()
            if weight_diff > 1e-6:
                print(f"  {name}: max diff = {weight_diff} ❌")
                all_weights_match = False
            else:
                print(f"  {name}: max diff = {weight_diff} ✓")
        
        if not all_weights_match:
            print("\n⚠️ Some weights don't match!")
        
        # Run forward pass on loaded model
        print("Running forward pass on loaded model...")
        with torch.no_grad():
            loaded_output = loaded_model(test_input, return_dict=True)
            loaded_logits = loaded_output.logits
        
        # Compare outputs
        print("Comparing outputs...")
        if original_logits is not None and loaded_logits is not None:
            max_diff = torch.max(torch.abs(original_logits - loaded_logits)).item()
            print(f"Maximum difference in logits: {max_diff}")
            assert max_diff < 1e-5, f"Outputs differ too much: {max_diff}"
        
        # Verify config attributes are preserved
        print("Verifying configuration...")
        assert loaded_model.num_vars == model.num_vars
        assert loaded_model.width == model.width
        assert loaded_model.n_embed_lit == model.n_embed_lit
        assert loaded_model.alpha == model.alpha
        assert loaded_model.beta == model.beta
        assert loaded_model.gamma == model.gamma
        assert loaded_model.share_semantic == model.share_semantic
        assert loaded_model.universal == model.universal
        
    print("✅ Test passed! Model can be saved and loaded successfully.")


if __name__ == "__main__":
    test_save_and_load()