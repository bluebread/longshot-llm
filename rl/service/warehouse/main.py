"""
FastAPI application for the Warehouse microservice.
"""

from fastapi import FastAPI, HTTPException, Query, Response
from pymongo import MongoClient
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

from models import (
    FormulaInfo,
    CreateFormulaRequest,
    UpdateFormulaRequest,
    FormulaResponse,
    LikelyIsomorphicResponse,
    LikelyIsomorphicRequest,
    TrajectoryInfo,
    CreateTrajectoryRequest,
    UpdateTrajectoryRequest,
    TrajectoryResponse,
    EvolutionGraphNode,
    CreateNodeRequest,
    UpdateNodeRequest,
    NodeResponse,
    EvolutionGraphEdge,
    CreateEdgeRequest,
    EdgeResponse,
    FormulaDefinition,
    SubgraphResponse,
    SubgraphRequest,
    AddFormulaRequest,
    ContractEdgeRequest,
    SuccessResponse
)

logging.basicConfig(
    level=logging.INFO, 
    filename="warehouse.log", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

mongo_client = MongoClient("mongodb://haowei:bread861122@mongo-bread:27017")
mongodb = mongo_client["LongshotWarehouse"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    try:
        # TODO: add validator
       mongodb.create_collection("FormulaTable", check_exists=True) 
       mongodb.create_collection("TrajectoryTable", check_exists=True)
    except Exception:
        pass
    yield
    # Cleanup resources
    mongo_client.close()
    
app = FastAPI(
    title="Warehouse API",
    lifespan=lifespan,
)
formula_table = mongodb["FormulaTable"]
trajectory_table = mongodb["TrajectoryTable"]

# Formula endpoints
@app.get("/formula/info", response_model=FormulaInfo)
async def get_formula_info(id: str = Query(..., description="Formula UUID")):
    """Retrieve information about a formula by its ID."""
    formula_doc = formula_table.find_one({"_id": id})
    if formula_doc:
        return FormulaInfo(**formula_doc)
    raise HTTPException(status_code=404, detail="Formula not found")

@app.post("/formula/info", response_model=FormulaResponse, status_code=201)
async def create_formula_info(request: CreateFormulaRequest):
    """Add a new formula entry to the formula table."""
    new_id = str(uuid.uuid4())
    formula_doc = request.model_dump()
    formula_doc["_id"] = new_id
    formula_doc["timestamp"] = datetime.now().astimezone()

    formula_table.insert_one(formula_doc)
    
    return FormulaResponse(id=formula_doc["_id"])


@app.put("/formula/info", response_model=SuccessResponse)
async def update_formula_info(request: UpdateFormulaRequest):
    """Update an existing formula entry."""
    formula_id = request.id
    update_data = request.model_dump(exclude_unset=True, exclude={"id"})

    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")

    result = formula_table.update_one(
        {"_id": formula_id},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Formula not found")
    
    return SuccessResponse(message="Formula updated successfully")


@app.delete("/formula/info", response_model=SuccessResponse)
async def delete_formula_info(id: str = Query(..., description="Formula UUID")):
    """Delete a formula entry."""
    result = formula_table.delete_one({"_id": id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Formula not found")
        
    return SuccessResponse(message="Formula deleted successfully")


@app.get("/formula/likely_isomorphic", response_model=LikelyIsomorphicResponse)
async def get_likely_isomorphic(wl_hash: str = Query(..., description="Weisfeiler-Lehman hash")):
    """Retrieve IDs of likely isomorphic formulas."""
    # Stub implementation
    logger.info(f"Retrieving likely isomorphic formulas for hash: {wl_hash}")
    return LikelyIsomorphicResponse(isomorphic_ids=["f123", "f124"])


@app.post("/formula/likely_isomorphic", response_model=SuccessResponse, status_code=201)
async def add_likely_isomorphic(request: LikelyIsomorphicRequest):
    """Add a likely isomorphic formula."""
    # Stub implementation
    return SuccessResponse(message="Likely isomorphic formula added successfully")


# Trajectory endpoints
@app.get("/trajectory", response_model=TrajectoryInfo)
async def get_trajectory(id: str = Query(..., description="Trajectory UUID")):
    """Retrieve a trajectory by its ID."""
    trajectory_doc = trajectory_table.find_one({"_id": id})
    if trajectory_doc:
        return TrajectoryInfo(**trajectory_doc)
    raise HTTPException(status_code=404, detail="Trajectory not found")


@app.post("/trajectory", response_model=TrajectoryResponse, status_code=201)
async def create_trajectory(trajectory: CreateTrajectoryRequest):
    """Add a new trajectory."""
    new_id = str(uuid.uuid4())
    trajectory_doc = trajectory.model_dump()
    trajectory_doc["_id"] = new_id
    trajectory_doc["timestamp"] = datetime.now().astimezone()
    
    trajectory_table.insert_one(trajectory_doc)
    
    return TrajectoryResponse(id=trajectory_doc["_id"])


@app.put("/trajectory", response_model=SuccessResponse)
async def update_trajectory(trajectory: UpdateTrajectoryRequest):
    """Update an existing trajectory."""
    trajectory_id = trajectory.id
    data = trajectory.model_dump(exclude_unset=True, exclude={"id"})
    update_data = {}

    if not data:
        raise HTTPException(status_code=400, detail="No update data provided")

    if "steps" in data:
        update_data = {
            f"steps.{step['order']}": {
                "token_type": step["token_type"],
                "token_literals": step["token_literals"],
                "reward": step["reward"],
            }
            for step in data["steps"]
        }
    if "base_formula_id" in data:
        update_data["base_formula_id"] = data["base_formula_id"]
        
    result = trajectory_table.update_one(
        {"_id": trajectory_id},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    
    return SuccessResponse(message="Trajectory updated successfully")


@app.delete("/trajectory", response_model=SuccessResponse)
async def delete_trajectory(id: str = Query(..., description="Trajectory UUID")):
    """Delete a trajectory."""
    result = trajectory_table.delete_one({"_id": id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Trajectory not found")
        
    return SuccessResponse(message="Trajectory deleted successfully")


# Evolution graph node endpoints
@app.get("/evolution_graph/node", response_model=EvolutionGraphNode)
async def get_evolution_graph_node(id: str = Query(..., description="Node UUID")):
    """Retrieve a node in the evolution graph by its ID."""
    # Stub implementation
    return EvolutionGraphNode(
        formula_id="f123",
        avgQ=2.5,
        visited_counter=10,
        inactive=False,
        in_degree=2,
        out_degree=3
    )


@app.post("/evolution_graph/node", response_model=NodeResponse, status_code=201)
async def create_evolution_graph_node(node: CreateNodeRequest):
    """Add a new node to the evolution graph."""
    # Stub implementation
    new_id = str(uuid.uuid4())
    return NodeResponse(node_id=new_id)


@app.put("/evolution_graph/node", response_model=SuccessResponse)
async def update_evolution_graph_node(node: UpdateNodeRequest):
    """Update an existing node."""
    # Stub implementation
    return SuccessResponse(message="Node updated successfully")


@app.delete("/evolution_graph/node", response_model=SuccessResponse)
async def delete_evolution_graph_node(node_id: str = Query(..., description="Node UUID")):
    """Delete a node."""
    # Stub implementation
    return SuccessResponse(message="Node deleted successfully")


# Evolution graph edge endpoints
@app.get("/evolution_graph/edge", response_model=EvolutionGraphEdge)
async def get_evolution_graph_edge(edge_id: str = Query(..., description="Edge UUID")):
    """Retrieve an edge in the evolution graph by its ID."""
    # Stub implementation
    return EvolutionGraphEdge(
        base_formula_id="f123",
        new_formula_id="f124"
    )


@app.post("/evolution_graph/edge", response_model=EdgeResponse, status_code=201)
async def create_evolution_graph_edge(edge: CreateEdgeRequest):
    """Add a new edge."""
    # Stub implementation
    new_id = str(uuid.uuid4())
    return EdgeResponse(edge_id=new_id)


@app.delete("/evolution_graph/edge", response_model=SuccessResponse)
async def delete_evolution_graph_edge(edge_id: str = Query(..., description="Edge UUID")):
    """Delete an edge."""
    # Stub implementation
    return SuccessResponse(message="Edge deleted successfully")


# High-level API endpoints
@app.get("/formula/definition", response_model=FormulaDefinition)
async def get_formula_definition(id: str = Query(..., description="Formula UUID")):
    """Retrieve the full definition of a formula by its ID."""
    # Stub implementation
    return FormulaDefinition(
        id=id,
        definition=[
            ["x1", "x2", "x3"],
            ["x4", "x5"]
        ]
    )


@app.get("/evolution_graph/subgraph", response_model=SubgraphResponse)
async def get_evolution_subgraph(
    num_vars: int = Query(..., description="Number of variables"),
    width: int = Query(..., description="Width of the formula")
):
    """Retrieve the evolution subgraph of active nodes."""
    # Stub implementation
    return SubgraphResponse(
        nodes=[],
        edges=[]
    )


@app.post("/formula/add", response_model=FormulaResponse, status_code=201)
async def add_formula(formula: AddFormulaRequest):
    """Add a new formula to the warehouse, updating the isomorphism hash table and evolution graph."""
    # Stub implementation
    new_id = str(uuid.uuid4())
    return FormulaResponse(id=new_id)


@app.post("/evolution_graph/subgraph", response_model=SuccessResponse, status_code=201)
async def add_subgraph(subgraph: SubgraphRequest):
    """Add a new subgraph to the evolution graph of a formula."""
    # Stub implementation
    return SuccessResponse(message="Subgraph added successfully")


@app.post("/evolution_graph/contract_edge", response_model=SuccessResponse)
async def contract_edge(request: ContractEdgeRequest):
    """Contract an edge in the evolution graph; one node will be deactivated."""
    # Stub implementation
    return SuccessResponse(message="Edge contracted successfully")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
