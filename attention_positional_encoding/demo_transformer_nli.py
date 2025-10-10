"""
Demo script for the complete Transformer NLI model

This script demonstrates how to use the transformer model for Natural Language Inference tasks.
"""

import torch
from transformer import TransformerForNLI, create_nli_model
from config import NLIConfig, get_config
from embeddings import create_nli_input_format


def demo_model_creation():
    """Demonstrate different ways to create the model."""
    print("=== Model Creation Demo ===")
    
    # Method 1: Using factory function
    print("1. Creating model with factory function...")
    model1 = create_nli_model('nli')
    print(f"Model created with {sum(p.numel() for p in model1.parameters())} parameters")
    
    # Method 2: Using custom config
    print("\n2. Creating model with custom config...")
    config = NLIConfig(
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        vocab_size=5000,
        max_seq_len=256
    )
    model2 = TransformerForNLI(config)
    print(f"Custom model created with {sum(p.numel() for p in model2.parameters())} parameters")
    
    return model2


def demo_input_format():
    """Demonstrate how to format inputs for NLI."""
    print("\n=== Input Format Demo ===")
    
    # Example premise and hypothesis token IDs
    batch_size = 2
    premise_ids = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0],      # "The cat is sleeping" (padded)
        [6, 7, 8, 9, 10, 11, 0]     # "A dog is running fast" (padded)
    ])
    
    hypothesis_ids = torch.tensor([
        [12, 13, 14, 0, 0],         # "An animal rests" (padded)
        [15, 16, 17, 18, 0]         # "The pet is moving" (padded)
    ])
    
    print(f"Premise IDs shape: {premise_ids.shape}")
    print(f"Hypothesis IDs shape: {hypothesis_ids.shape}")
    
    # Create NLI input format
    input_ids, segment_ids, attention_mask = create_nli_input_format(
        premise_ids, hypothesis_ids,
        cls_token_id=101, sep_token_id=102, pad_token_id=0, max_length=20
    )
    
    print(f"\nFormatted input_ids shape: {input_ids.shape}")
    print(f"Segment IDs shape: {segment_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    print(f"\nExample input_ids[0]: {input_ids[0]}")
    print(f"Example segment_ids[0]: {segment_ids[0]}")
    print(f"Example attention_mask[0]: {attention_mask[0]}")
    
    return input_ids, segment_ids, attention_mask


def demo_forward_pass():
    """Demonstrate forward pass through the model."""
    print("\n=== Forward Pass Demo ===")
    
    # Create a small model for demo
    config = NLIConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=64,
        trace_shapes=True  # Enable shape tracing
    )
    
    model = TransformerForNLI(config)
    model.eval()
    
    # Create sample input
    batch_size = 2
    seq_len = 20
    
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    segment_ids = torch.cat([
        torch.zeros(batch_size, seq_len // 2),
        torch.ones(batch_size, seq_len // 2)
    ], dim=1).long()
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Segment shape: {segment_ids.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        logits, hidden_states = model(input_ids, segment_ids)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    
    print(f"\nPredictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample probabilities: {probabilities[0]}")
    
    return model, logits


def demo_training_setup():
    """Demonstrate how to set up the model for training."""
    print("\n=== Training Setup Demo ===")
    
    # Create model
    model = create_nli_model('nli')
    
    # Sample training data
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    segment_ids = torch.randint(0, 2, (batch_size, seq_len))
    labels = torch.randint(0, 3, (batch_size,))  # 3 classes for NLI
    
    print(f"Training batch - Input: {input_ids.shape}, Labels: {labels.shape}")
    
    # Forward pass with labels (training mode)
    model.train()
    loss, logits, hidden_states = model(input_ids, segment_ids, labels=labels)
    
    print(f"Training loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    
    # Backward pass (gradient computation)
    loss.backward()
    
    print("Gradients computed successfully!")
    
    # Show some gradient statistics
    total_params = 0
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
    
    print(f"Parameters with gradients: {params_with_grad}/{total_params}")


def main():
    """Run all demos."""
    print("Transformer NLI Model Demo")
    print("=" * 50)
    
    # Demo 1: Model creation
    model = demo_model_creation()
    
    # Demo 2: Input formatting
    input_ids, segment_ids, attention_mask = demo_input_format()
    
    # Demo 3: Forward pass
    model, logits = demo_forward_pass()
    
    # Demo 4: Training setup
    demo_training_setup()
    
    print("\n" + "=" * 50)
    print("All demos completed successfully!")
    print("\nYour transformer model is ready for NLI tasks!")
    print("Key features implemented:")
    print("- ✅ Multi-head self-attention (encoder mode)")
    print("- ✅ Position-wise feed forward networks")
    print("- ✅ Layer normalization and residual connections")
    print("- ✅ Sinusoidal and learned positional encodings")
    print("- ✅ Token and segment embeddings for NLI")
    print("- ✅ Classification head for 3-class NLI")
    print("- ✅ Configurable model architecture")
    print("- ✅ Training and inference modes")


if __name__ == "__main__":
    main()
