import argparse
import json
import os
import random
import statistics
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from model import GPT2ForLongshot
from dataset import TrajectoryDataset
from collator import TrajectoryCollator


def extract_avgq_from_trajectory(trajectory: List[int], model: GPT2ForLongshot, device: str = "cpu") -> float:
    """
    Extract the avgQ value from a trajectory by running it through the model.

    Args:
        trajectory: List of encoded gate tokens
        model: The GPT2ForLongshot model
        device: Device to run computation on

    Returns:
        The highest avgQ value from the trajectory

    Raises:
        ValueError: If trajectory is empty
        RuntimeError: If avgQ extraction fails
    """
    if not trajectory:
        raise ValueError("Trajectory cannot be empty")

    if len(trajectory) > 512:  # Reasonable limit based on typical model architecture
        print(f"Warning: Trajectory length {len(trajectory)} may exceed model capacity")

    try:
        # Convert to tensor and add batch dimension with proper device placement
        input_ids = torch.tensor(trajectory, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float)

        # Get model output using the full forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # The model's forward method already computes the hidden states
            # We need to extract them from the model's internal computation
            # Using the model's mlp layer to get the final output
            x, _ = model._decode_input_to_embedding(input_ids)
            x = model.proj(x)
            gpt2_outputs = model.gpt2_model(
                inputs_embeds=x,
                attention_mask=attention_mask,
                return_dict=True
            )
            y = model.mlp(gpt2_outputs.last_hidden_state)

            # Split to get q values
            _, _, _, q = torch.split(y, [1, model.num_vars, model.num_vars, 1], dim=-1)
            q = q.squeeze(-1)  # Remove last dimension

            # Apply attention mask before getting max
            q_masked = q * attention_mask

            # Return the maximum avgQ value (ensure non-negative)
            max_q = q_masked.max().item()
            return max(0.0, max_q)  # avgQ values should be non-negative
    except Exception as e:
        raise RuntimeError(f"Failed to extract avgQ from trajectory: {e}")


def generate_sequences(
    model: GPT2ForLongshot,
    prompt: torch.Tensor,
    num_sequences: int = 1,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu"
) -> Tuple[torch.Tensor, List[float]]:
    """
    Generate sequences from a prompt and extract avgQ values.

    Args:
        model: The GPT2ForLongshot model
        prompt: Input prompt tensor
        num_sequences: Number of sequences to generate
        max_length: Maximum length of generated sequences
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        device: Device to run generation on

    Returns:
        Tuple of (generated sequences, list of max avgQ values)
    """
    model.to(device)
    prompt = prompt.to(device)

    # Generate sequences
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=0,
            eos_token_id=None,  # No EOS token for continuous generation
        )

    # Extract avgQ values for each generated sequence
    avgq_values = []
    for seq in generated:
        avgq = extract_avgq_from_trajectory(seq.cpu().tolist(), model, device)
        avgq_values.append(avgq)

    return generated, avgq_values


def extract_avgq_batch(
    trajectories: List[List[int]],
    model: GPT2ForLongshot,
    device: str = "cpu"
) -> List[float]:
    """
    Extract avgQ values for multiple trajectories in a single batch.

    Args:
        trajectories: List of trajectory sequences
        model: The GPT2ForLongshot model
        device: Device to run computation on

    Returns:
        List of maximum avgQ values for each trajectory
    """
    if not trajectories:
        return []

    # Use TrajectoryCollator for proper batching
    collator = TrajectoryCollator(num_vars=model.num_vars)

    # Convert to dataset format
    batch_data = [
        {
            "input_ids": traj,
            "attention_mask": [1] * len(traj),
            "labels": [0] * len(traj)
        }
        for traj in trajectories
    ]

    batch = collator(batch_data)
    batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

    with torch.no_grad():
        # Use the model's internal computation
        x, _ = model._decode_input_to_embedding(batch["input_ids"])
        x = model.proj(x)
        gpt2_outputs = model.gpt2_model(
            inputs_embeds=x,
            attention_mask=batch["attention_mask"],
            return_dict=True
        )
        y = model.mlp(gpt2_outputs.last_hidden_state)

        # Extract Q values for entire batch
        _, _, _, q = torch.split(y, [1, model.num_vars, model.num_vars, 1], dim=-1)
        q = q.squeeze(-1)

        # Apply attention mask and get max for each sequence
        q_masked = q * batch["attention_mask"]
        max_q_values = q_masked.max(dim=1)[0].cpu().tolist()

        # Ensure all avgQ values are non-negative
        max_q_values = [max(0.0, v) for v in max_q_values]

    return max_q_values


def analyze_avgq_improvement(
    prompt_avgq: List[float],
    generated_avgq: List[float]
) -> Dict[str, float]:
    """
    Analyze improvement in avgQ values from prompts to generated sequences.

    Args:
        prompt_avgq: List of avgQ values from prompts
        generated_avgq: List of avgQ values from generated sequences

    Returns:
        Dictionary with analysis metrics
    """
    if not prompt_avgq or not generated_avgq:
        return {}

    # Calculate paired improvements
    min_len = min(len(prompt_avgq), len(generated_avgq))
    improvements = [generated_avgq[i] - prompt_avgq[i % len(prompt_avgq)]
                   for i in range(min_len)]

    return {
        "prompt_mean": statistics.mean(prompt_avgq),
        "prompt_std": statistics.stdev(prompt_avgq) if len(prompt_avgq) > 1 else 0,
        "prompt_max": max(prompt_avgq),
        "prompt_min": min(prompt_avgq),
        "generated_mean": statistics.mean(generated_avgq),
        "generated_std": statistics.stdev(generated_avgq) if len(generated_avgq) > 1 else 0,
        "generated_max": max(generated_avgq),
        "generated_min": min(generated_avgq),
        "improvement_mean": statistics.mean(improvements) if improvements else 0,
        "improvement_std": statistics.stdev(improvements) if len(improvements) > 1 else 0,
        "improvement_max": max(improvements) if improvements else 0,
        "improvement_count": sum(1 for i in improvements if i > 0),
        "improvement_percentage": 100 * sum(1 for i in improvements if i > 0) / len(improvements) if improvements else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT2ForLongshot model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the pre-trained model directory"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset JSON file (if not provided, uses warehouse)"
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=None,
        help="Number of variables (required if not using dataset-path)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width parameter (required if not using dataset-path)"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Number of sequences to generate per prompt"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of random prompts to evaluate"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=5,
        help="Length of the prompt (number of tokens)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum length of generated sequences"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file to save results"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the pre-trained model
    print(f"Loading model from {args.model_path}...")
    model = GPT2ForLongshot.from_pretrained(args.model_path)
    model.eval()
    model.to(args.device)

    # Initialize dataset
    print("Initializing dataset...")
    if args.dataset_path:
        dataset = TrajectoryDataset(local_file=args.dataset_path)
        # Extract num_vars and width from model config
        num_vars = model.num_vars
        width = model.width
    else:
        if args.num_vars is None or args.width is None:
            # Try to get from model config
            num_vars = model.num_vars
            width = model.width
            print(f"Using model's num_vars={num_vars} and width={width}")
        else:
            num_vars = args.num_vars
            width = args.width
        dataset = TrajectoryDataset(num_vars=num_vars, width=width)

    # Initialize data collator
    collator = TrajectoryCollator(num_vars=num_vars)

    # Results storage
    results = {
        "model_path": args.model_path,
        "num_prompts": args.num_prompts,
        "num_sequences_per_prompt": args.num_sequences,
        "prompt_length": args.prompt_length,
        "max_length": args.max_length,
        "evaluations": []
    }

    print(f"\nEvaluating {args.num_prompts} random prompts...")
    print("=" * 60)

    for i in range(args.num_prompts):
        # Select a random trajectory from the dataset
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]

        # Extract prompt from the trajectory
        full_trajectory = sample["input_ids"]
        if len(full_trajectory) < args.prompt_length:
            print(f"Skipping trajectory {idx} (too short: {len(full_trajectory)} < {args.prompt_length})")
            continue

        # Use first prompt_length tokens as prompt
        prompt_tokens = full_trajectory[:args.prompt_length]
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)

        # Get avgQ for the prompt
        prompt_avgq = sample["labels"][:args.prompt_length]
        max_prompt_avgq = max(prompt_avgq) if prompt_avgq else 0.0

        # Generate sequences
        print(f"\nPrompt {i+1}/{args.num_prompts}:")
        print(f"  Prompt tokens: {prompt_tokens[:10]}..." if len(prompt_tokens) > 10 else f"  Prompt tokens: {prompt_tokens}")
        print(f"  Max prompt avgQ: {max_prompt_avgq:.4f}")

        generated_sequences, generated_avgqs = generate_sequences(
            model=model,
            prompt=prompt_tensor,
            num_sequences=args.num_sequences,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )

        # Find highest avgQ among generated sequences
        max_generated_avgq = max(generated_avgqs) if generated_avgqs else 0.0
        avg_generated_avgq = sum(generated_avgqs) / len(generated_avgqs) if generated_avgqs else 0.0

        print(f"  Generated sequences:")
        for j, avgq in enumerate(generated_avgqs):
            print(f"    Sequence {j+1}: avgQ = {avgq:.4f}")
        print(f"  Max generated avgQ: {max_generated_avgq:.4f}")
        print(f"  Avg generated avgQ: {avg_generated_avgq:.4f}")
        print(f"  Improvement: {max_generated_avgq - max_prompt_avgq:.4f}")

        # Store results
        evaluation = {
            "prompt_idx": idx,
            "prompt_tokens": prompt_tokens,
            "prompt_max_avgq": max_prompt_avgq,
            "generated_avgqs": generated_avgqs,
            "generated_max_avgq": max_generated_avgq,
            "generated_avg_avgq": avg_generated_avgq,
            "improvement": max_generated_avgq - max_prompt_avgq
        }
        results["evaluations"].append(evaluation)

    # Calculate overall statistics
    if results["evaluations"]:
        all_prompt_avgqs = [e["prompt_max_avgq"] for e in results["evaluations"]]
        all_generated_max_avgqs = [e["generated_max_avgq"] for e in results["evaluations"]]
        all_generated_avgqs = []
        for e in results["evaluations"]:
            all_generated_avgqs.extend(e["generated_avgqs"])

        # Use the analyze_avgq_improvement function for comprehensive statistics
        analysis = analyze_avgq_improvement(all_prompt_avgqs, all_generated_avgqs)

        print("\n" + "=" * 60)
        print("OVERALL STATISTICS:")
        print(f"  Prompts evaluated: {len(all_prompt_avgqs)}")
        print(f"  Sequences generated: {len(all_generated_avgqs)}")
        print(f"\nPrompt avgQ Statistics:")
        print(f"    Mean: {analysis['prompt_mean']:.4f} ± {analysis['prompt_std']:.4f}")
        print(f"    Range: [{analysis['prompt_min']:.4f}, {analysis['prompt_max']:.4f}]")
        print(f"\nGenerated avgQ Statistics:")
        print(f"    Mean: {analysis['generated_mean']:.4f} ± {analysis['generated_std']:.4f}")
        print(f"    Range: [{analysis['generated_min']:.4f}, {analysis['generated_max']:.4f}]")
        print(f"\nImprovement Statistics:")
        print(f"    Mean improvement: {analysis['improvement_mean']:.4f} ± {analysis['improvement_std']:.4f}")
        print(f"    Max improvement: {analysis['improvement_max']:.4f}")
        print(f"    Improved sequences: {analysis['improvement_count']}/{len(all_generated_avgqs)} ({analysis['improvement_percentage']:.1f}%)")

        results["statistics"] = analysis
        results["timestamp"] = datetime.now().isoformat()

    # Save results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()