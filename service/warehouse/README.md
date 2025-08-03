# Warehouse Microservice

This directory contains the implementation of the Warehouse microservice using FastAPI.

## Structure

- `main.py` - FastAPI application with all API endpoints
- `models.py` - Pydantic models for request/response schemas
- `__init__.py` - Package initialization

## Running the Service

To run the warehouse service:

```bash
cd rl/service/warehouse
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or run directly:

```bash
python main.py
```

## API Documentation

The API documentation will be automatically available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### Formula Management
- `GET /formula/info` - Get formula information
- `POST /formula/info` - Create formula
- `PUT /formula/info` - Update formula
- `DELETE /formula/info` - Delete formula
- `GET /formula/likely_isomorphic` - Get isomorphic formulas
- `POST /formula/likely_isomorphic` - Add isomorphic formula
- `GET /formula/definition` - Get formula definition
- `POST /formula/add` - Add formula (high-level)

### Trajectory Management
- `GET /trajectory` - Get trajectory
- `POST /trajectory` - Create trajectory
- `PUT /trajectory` - Update trajectory
- `DELETE /trajectory` - Delete trajectory

### Evolution Graph Management
- `GET /evolution_graph/node` - Get graph node
- `POST /evolution_graph/node` - Create graph node
- `PUT /evolution_graph/node` - Update graph node
- `DELETE /evolution_graph/node` - Delete graph node
- `GET /evolution_graph/edge` - Get graph edge
- `POST /evolution_graph/edge` - Create graph edge
- `DELETE /evolution_graph/edge` - Delete graph edge
- `GET /evolution_graph/subgraph` - Get subgraph
- `POST /evolution_graph/subgraph` - Add subgraph
- `POST /evolution_graph/contract_edge` - Contract edge

### Health Check
- `GET /health` - Health check endpoint

## Note

This is a stub implementation. All endpoints return mock data and need to be connected to actual database storage and business logic.
