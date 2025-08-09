# Clusterbomb Service

A microservice for weapon rollout functionality in the gym-longshot system.

## Overview

The Clusterbomb service provides a single endpoint for executing weapon rollout operations. It is designed to integrate with the broader longshot ecosystem and follows the same architectural patterns as other services in the system.

## API Endpoints

### POST /weapon/rollout

Execute a weapon rollout operation that collects trajectories from the environment and pushes them to the trajectory queue.

**Request Body:**
```json
{
  "num_vars": 5,
  "width": 3,
  "steps_per_trajectory": 50,
  "num_trajectories": 1000,
  "initial_definition": [1, 2, -3, 0, 4, -5, 0],
  "initial_formula_id": "formula_abc123",
  "seed": 42
}
```

**Request Fields:**
- `num_vars` (int, required): Number of variables in the formula
- `width` (int, required): Width of the formula
- `steps_per_trajectory` (int, optional): Number of steps per trajectory
- `num_trajectories` (int, optional): Number of trajectories to collect
- `initial_definition` (list[int], required): Initial definition of the formula as list of integers representing gates
- `initial_formula_id` (str, optional): ID of the initial formula used as base for trajectories
- `seed` (int, optional): Random seed for reproducible trajectory generation. If not provided, randomness will be non-deterministic

**Response:**
```json
{
  "total_steps": 50000,
  "num_trajectories": 1000
}
```

**Response Fields:**
- `total_steps`: Total number of steps actually executed across all trajectories
- `num_trajectories`: Number of trajectories actually collected

**Behavior:**
- Parses the initial formula definition and creates a FormulaGame environment
- Runs trajectory simulation with random token generation
- Collects trajectory data with steps containing order, token_type, token_literals, reward, and avgQ
- Pushes all collected trajectories to RabbitMQ queue in batch for efficient processing
- Uses the provided seed for deterministic random number generation if specified

### GET /health

Health check endpoint for service monitoring.

**Response:**
```json
{
  "status": "healthy",
  "service": "clusterbomb"
}
```

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

The service will be available at `http://localhost:8060`.

### Docker

```bash
# Build the image
docker build -t clusterbomb .

# Run the container
docker run -p 8060:8060 clusterbomb
```

## Configuration

The service uses port 8060 by default and follows the same logging and error handling patterns as other services in the longshot ecosystem.

## Integration

This service is designed to be integrated into the docker-compose infrastructure along with other longshot services (warehouse, ranker, trajproc, etc.).