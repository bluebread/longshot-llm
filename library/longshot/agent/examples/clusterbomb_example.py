#!/usr/bin/env python3
"""
Example usage of ClusterbombAgent for generating trajectories.

This example demonstrates how to use the ClusterbombAgent to:
1. Connect to the Clusterbomb microservice
2. Execute weapon rollouts to generate trajectories
3. Handle both synchronous and asynchronous operations
"""

import asyncio
from longshot.agent import ClusterbombAgent, AsyncClusterbombAgent


def sync_example():
    """Synchronous example of using ClusterbombAgent."""
    print("=== Synchronous ClusterbombAgent Example ===\n")
    
    # Connect to the Clusterbomb service
    agent = ClusterbombAgent('localhost', 8060)
    
    try:
        # Check service health
        health = agent.health_check()
        print(f"Service health: {health}")
        
        # Execute a weapon rollout with deterministic seed
        print("\nExecuting weapon rollout with seed...")
        result = agent.weapon_rollout(
            num_vars=3,
            width=2,
            size=5,
            steps_per_trajectory=10,
            num_trajectories=3,
            initial_definition=[1, 2, 3, 4, 5],
            initial_node_id="example_formula_001",
            seed=42  # Deterministic for reproducibility
        )
        
        print(f"Rollout completed:")
        print(f"  - Total steps: {result.total_steps}")
        print(f"  - Trajectories generated: {result.num_trajectories}")
        
        # Execute another rollout without seed (non-deterministic)
        print("\nExecuting weapon rollout without seed (random)...")
        result2 = agent.weapon_rollout(
            num_vars=2,
            width=2,
            size=3,
            steps_per_trajectory=5,
            num_trajectories=2,
            initial_definition=[1, 2, 3]
            # No seed - will use random generation
        )
        
        print(f"Random rollout completed:")
        print(f"  - Total steps: {result2.total_steps}")
        print(f"  - Trajectories generated: {result2.num_trajectories}")
        
    finally:
        agent.close()


def context_manager_example():
    """Example using context manager for automatic cleanup."""
    print("\n=== Context Manager Example ===\n")
    
    # Using context manager ensures proper cleanup
    with ClusterbombAgent('localhost', 8060) as agent:
        health = agent.health_check()
        print(f"Service status: {health['status']}")
        
        # Small batch rollout
        result = agent.weapon_rollout(
            num_vars=2,
            width=2,
            size=4,
            steps_per_trajectory=3,
            num_trajectories=1,
            initial_definition=[1, 2, 3, 4],
            seed=123
        )
        
        print(f"Generated {result.num_trajectories} trajectory with {result.total_steps} steps")


async def async_example():
    """Asynchronous example of using AsyncClusterbombAgent."""
    print("\n=== Asynchronous ClusterbombAgent Example ===\n")
    
    # Using async context manager
    async with AsyncClusterbombAgent('localhost', 8060) as agent:
        # Check health asynchronously
        health = await agent.health_check()
        print(f"Async health check: {health}")
        
        # Execute multiple rollouts concurrently
        print("\nExecuting multiple rollouts concurrently...")
        
        rollout_tasks = [
            agent.weapon_rollout(
                num_vars=3,
                width=2,
                size=5,
                steps_per_trajectory=5,
                num_trajectories=2,
                initial_definition=[1, 2, 3, 4, 5],
                initial_node_id=f"async_formula_{i}",
                seed=100 + i
            )
            for i in range(3)
        ]
        
        # Wait for all rollouts to complete
        results = await asyncio.gather(*rollout_tasks)
        
        for i, result in enumerate(results):
            print(f"  Rollout {i+1}: {result.num_trajectories} trajectories, {result.total_steps} steps")


async def batch_processing_example():
    """Example of batch processing with multiple agents."""
    print("\n=== Batch Processing Example ===\n")
    
    # Create multiple agents for parallel processing
    agents = [
        AsyncClusterbombAgent('localhost', 8060)
        for _ in range(2)
    ]
    
    try:
        # Different configurations for each batch
        configs = [
            {
                "num_vars": 2,
                "width": 2,
                "size": 3,
                "steps_per_trajectory": 10,
                "num_trajectories": 5,
                "initial_definition": [1, 2, 3],
            },
            {
                "num_vars": 3,
                "width": 3,
                "size": 6,
                "steps_per_trajectory": 15,
                "num_trajectories": 3,
                "initial_definition": [1, 2, 3, 4, 5, 6],
            }
        ]
        
        # Process batches in parallel
        tasks = [
            agent.weapon_rollout(**config, seed=i*100)
            for i, (agent, config) in enumerate(zip(agents, configs))
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_trajectories = sum(r.num_trajectories for r in results)
        total_steps = sum(r.total_steps for r in results)
        
        print(f"Batch processing complete:")
        print(f"  - Total trajectories: {total_trajectories}")
        print(f"  - Total steps: {total_steps}")
        
    finally:
        # Clean up all agents
        await asyncio.gather(*[agent.close() for agent in agents])


def main():
    """Run all examples."""
    print("ClusterbombAgent Examples")
    print("=" * 50)
    
    # Run synchronous examples
    try:
        sync_example()
        context_manager_example()
    except Exception as e:
        print(f"Sync example error: {e}")
        print("Make sure the Clusterbomb service is running on localhost:8060")
    
    # Run asynchronous examples
    try:
        asyncio.run(async_example())
        asyncio.run(batch_processing_example())
    except Exception as e:
        print(f"Async example error: {e}")
        print("Make sure the Clusterbomb service is running on localhost:8060")


if __name__ == "__main__":
    main()