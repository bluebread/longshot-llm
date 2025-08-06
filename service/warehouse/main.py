"""
FastAPI application for the Warehouse microservice.
"""

from fastapi import FastAPI, HTTPException, Query, Response
from pymongo import MongoClient
from redis import Redis
from neo4j import GraphDatabase
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
import logging

from longshot.models import (
    QueryFormulaInfoResponse,
    CreateFormulaRequest,
    UpdateFormulaRequest,
    FormulaResponse,
    LikelyIsomorphicResponse,
    LikelyIsomorphicRequest,
    QueryTrajectoryInfoResponse,
    CreateTrajectoryRequest,
    UpdateTrajectoryRequest,
    TrajectoryResponse,
    QueryEvolutionGraphNode,
    CreateNodeRequest,
    UpdateNodeRequest,
    NodeResponse,
    QueryFormulaDefinitionResponse,
    CreateNewPathRequest,
    DownloadNodesResponse,
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

redis_config = {
    "host": "redis-bread",
    "port": 6379,
    "username": "default",
    "password": "bread861122",
    "decode_responses": True
}
iso_hash_table = Redis(db=0, **redis_config)

neo4j_config = {
    "uri": "neo4j://neo4j-bread:7687",
    "auth": ("neo4j", "bread861122"),
}
neo4j_driver = GraphDatabase.driver(**neo4j_config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check availability of external services
    try:
        # Test MongoDB connection
        mongodb.command("ping")
        # Test Redis connection
        iso_hash_table.ping()
        # Test Neo4j connection
        neo4j_driver.verify_connectivity()
    except Exception as e:
        raise HTTPException(status_code=500, detail="MongoDB connection failed")
    
    # Initialize MongoDB resources
    try:
        # TODO: add validator
        mongodb.create_collection("FormulaTable", check_exists=True) 
        mongodb.create_collection("TrajectoryTable", check_exists=True)
    except Exception:
        pass

    # Initialize Neo4j resources
    init_statements = [
        """
        CREATE CONSTRAINT formula_id_unique IF NOT EXISTS
        FOR (n:FormulaNode)
        REQUIRE n.formula_id IS UNIQUE
        """,
        """
        CREATE INDEX num_vars_index IF NOT EXISTS
        FOR (n:FormulaNode)
        ON (n.num_vars)
        """,
        """
        CREATE INDEX width_index IF NOT EXISTS
        FOR (n:FormulaNode)
        ON (n.width)
        """,
        """
        CREATE INDEX size_index IF NOT EXISTS
        FOR (n:FormulaNode)
        ON (n.size)
        """
    ]

    with neo4j_driver.session() as session:
        for stmt in init_statements:
            session.run(stmt)
    
    # Yield control to the application
    yield
    
    # Cleanup resources
    mongo_client.close()
    iso_hash_table.close()
    neo4j_driver.close()
    
app = FastAPI(
    title="Warehouse API",
    lifespan=lifespan,
)
formula_table = mongodb["FormulaTable"]
trajectory_table = mongodb["TrajectoryTable"]

# Formula endpoints
@app.get("/formula/info", response_model=QueryFormulaInfoResponse)
async def get_formula_info(id: str = Query(..., description="Formula UUID")):
    """Retrieve information about a formula by its ID."""
    formula_doc = formula_table.find_one({"_id": id})
    if formula_doc:
        return QueryFormulaInfoResponse(**formula_doc)
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


# Likely isomorphic formulas endpoints
@app.get("/formula/likely_isomorphic", response_model=LikelyIsomorphicResponse)
async def get_likely_isomorphic(wl_hash: str = Query(..., description="Weisfeiler-Lehman hash")):
    """Retrieve IDs of likely isomorphic formulas."""
    formula_ids = list(iso_hash_table.smembers(wl_hash))
    return LikelyIsomorphicResponse(wl_hash=wl_hash, isomorphic_ids=formula_ids)


@app.post("/formula/likely_isomorphic", response_model=SuccessResponse, status_code=201)
async def add_likely_isomorphic(request: LikelyIsomorphicRequest):
    """Add a likely isomorphic formula."""
    iso_hash_table.sadd(request.wl_hash, request.formula_id)
    return SuccessResponse(message="Likely isomorphic formula added successfully")

@app.delete("/formula/likely_isomorphic", response_model=SuccessResponse)
async def delete_likely_isomorphic(wl_hash: str = Query(..., description="Weisfeiler-Lehman hash")):
    """Delete a likely isomorphic formula."""
    iso_hash_table.delete(wl_hash)
    return SuccessResponse(message="Likely isomorphic formula deleted successfully")

# Trajectory endpoints
@app.get("/trajectory", response_model=QueryTrajectoryInfoResponse)
async def get_trajectory(id: str = Query(..., description="Trajectory UUID")):
    """Retrieve a trajectory by its ID."""
    trajectory_doc = trajectory_table.find_one({"_id": id})
    if trajectory_doc:
        return QueryTrajectoryInfoResponse(**trajectory_doc)
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
@app.get("/evolution_graph/node", response_model=QueryEvolutionGraphNode)
async def get_evolution_graph_node(id: str = Query(..., description="Node UUID")):
    """Retrieve a node in the evolution graph by its ID."""
    query = """
    MATCH   (n:FormulaNode {formula_id: $id})
    OPTIONAL MATCH (n)--(m:FormulaNode)
    WITH    n, 
            sum(CASE WHEN m.avgQ < n.avgQ THEN 1 ELSE 0 END) AS in_degree,
            sum(CASE WHEN m.avgQ > n.avgQ THEN 1 ELSE 0 END) AS out_degree
    RETURN  n.formula_id AS formula_id,
            n.avgQ AS avgQ,
            n.visited_counter AS visited_counter,
            n.num_vars AS num_vars,
            n.width AS width,
            n.size AS size,
            in_degree,
            out_degree
    """
    with neo4j_driver.session() as session:
        result = session.run(query, id=id).single()
    
    if not result:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return QueryEvolutionGraphNode(**result.data())


@app.post("/evolution_graph/node", response_model=NodeResponse, status_code=201)
async def create_evolution_graph_node(node: CreateNodeRequest):
    """Add a new node to the evolution graph."""
    query = """
    MERGE (n:FormulaNode {formula_id: $formula_id})
    ON CREATE SET n.avgQ = $avgQ,
                  n.num_vars = $num_vars,
                  n.width = $width,
                  n.size = $size,
                  n.visited_counter = 0
    RETURN n.formula_id AS formula_id
    """
    node_data = node.model_dump()
    
    with neo4j_driver.session() as session:
        result = session.run(query, **node_data).single()

    if not result:
        raise HTTPException(status_code=500, detail="Failed to create node")

    return NodeResponse(formula_id=result["formula_id"])


@app.put("/evolution_graph/node", response_model=SuccessResponse)
async def update_evolution_graph_node(node: UpdateNodeRequest):
    """Update an existing node."""
    formula_id = node.formula_id
    update_data = node.model_dump(exclude_unset=True, exclude={"formula_id"})

    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")

    ivc_s = 'inc_visited_counter'
    set_clauses = [f"n.{key} = ${key}" for key in update_data.keys() if key != ivc_s]
    
    if ivc_s in update_data:
        set_clauses.append("n.visited_counter = n.visited_counter + $inc_visited_counter")

    query = f"""
    MATCH   (n:FormulaNode {{formula_id: $formula_id}})
    SET     {', '.join(set_clauses)}
    RETURN  n
    """
    
    params = {"formula_id": formula_id, **update_data}

    with neo4j_driver.session() as session:
        result = session.run(query, params).single()

    if not result:
        raise HTTPException(status_code=404, detail="Node not found")

    return SuccessResponse(message="Node updated successfully")


@app.delete("/evolution_graph/node", response_model=SuccessResponse)
async def delete_evolution_graph_node(formula_id: str = Query(..., description="Node UUID")):
    """Delete a node and its relationships."""
    query = """
    MATCH   (n:FormulaNode {formula_id: $formula_id})
    WITH    n, n.formula_id AS formula_id
    DETACH  DELETE n
    RETURN  formula_id
    """
    with neo4j_driver.session() as session:
        result = session.run(query, formula_id=formula_id).single()

    if not result:
        raise HTTPException(status_code=404, detail="Node not found")

    return SuccessResponse(message="Node deleted successfully")


# # Evolution graph edge endpoints
# @app.get("/evolution_graph/edge", response_model=EvolutionGraphEdge)
# async def get_evolution_graph_edge(edge_id: str = Query(..., description="Edge UUID")):
#     """Retrieve an edge in the evolution graph by its ID."""
#     # Stub implementation
#     return EvolutionGraphEdge(
#         base_formula_id="f123",
#         new_formula_id="f124"
#     )


# @app.post("/evolution_graph/edge", response_model=EdgeResponse, status_code=201)
# async def create_evolution_graph_edge(edge: CreateEdgeRequest):
#     """Add a new edge."""
#     # Stub implementation
#     new_id = str(uuid.uuid4())
#     return EdgeResponse(edge_id=new_id)


# @app.delete("/evolution_graph/edge", response_model=SuccessResponse)
# async def delete_evolution_graph_edge(edge_id: str = Query(..., description="Edge UUID")):
#     """Delete an edge."""
#     # Stub implementation
#     return SuccessResponse(message="Edge deleted successfully")


# High-level API endpoints
@app.get("/formula/definition", response_model=QueryFormulaDefinitionResponse)
async def get_formula_definition(id: str = Query(..., description="Formula UUID")):
    """Retrieve the full definition of a formula by its ID."""
    # Check if the formula exists first
    formula_doc = formula_table.find_one({"_id": id})
    if not formula_doc:
        raise HTTPException(status_code=404, detail="Formula not found")

    # Aggregation pipeline to trace back the formula's history
    pipeline = [
        {
            "$match": {"_id": id}
        },
        {
            "$graphLookup": {
                "from": "FormulaTable",
                "startWith": "$base_formula_id",
                "connectFromField": "base_formula_id",
                "connectToField": "_id",
                "as": "ancestry",
                "depthField": "depth"
            }
        },
    ]

    result = formula_table.aggregate(pipeline).to_list()[0]

    if len(result) == 0:
        # This case handles formulas with no base_formula_id and no corresponding trajectory
        # (i.e., base formulas that were not created via a trajectory).
        # It returns an empty list of steps.
        return QueryFormulaDefinitionResponse(id=id, definition=[])
    
    tid_s = 'trajectory_id'
    dep_s = 'depth'
    traj_ids = sorted([(t[dep_s], t[tid_s]) for t in result.get("ancestry", [])], reverse=True)
    traj_ids = [t for _, t in traj_ids if t is not None]
    
    if tid_s in result and result[tid_s] is not None:
        # If the current formula has a trajectory_id, include it in the list
        traj_ids.append(result[tid_s])
        
    
    # Retrieve the full trajectory information
    trajectory_docs = list(trajectory_table.find({"_id": {"$in": traj_ids}}))
    traj_lookup = {t["_id"]: t["steps"] for t in trajectory_docs}

    definition = set()
    
    for tid in traj_ids:
        steps = traj_lookup.get(tid, [])
        
        for step in steps:
            ttype = step["token_type"]
            literals = step["token_literals"]

            if ttype == 0:
                definition.add(literals)
            elif ttype == 1:
                definition.discard(literals)
            else:
                raise HTTPException(status_code=500, detail=f"Unknown token type: {ttype}")

    return QueryFormulaDefinitionResponse(id=id, definition=list(definition))


@app.post("/evolution_graph/path", response_model=SuccessResponse, status_code=201)
async def create_new_path(request: CreateNewPathRequest):
    """Create a new path in the evolution graph."""
    query = """
    WITH    $path AS nodes
    UNWIND  range(0, size(nodes) - 2) AS i
    MERGE   (n:FormulaNode {formula_id: nodes[i]})
    MERGE   (m:FormulaNode {formula_id: nodes[i + 1]})
    MERGE   (n)-[:EVOLVED_TO]->(m)
    """
    
    with neo4j_driver.session() as session:
        session.run(query, path=request.path)
        
    return SuccessResponse(message="Path created successfully")


@app.get("/evolution_graph/download_nodes", response_model=DownloadNodesResponse)
async def download_evolution_graph_nodes(
    num_vars: int = Query(..., description="Number of variables in formula"),
    width: int = Query(..., description="Width of formula"),
    size_constraint: int | None = Query(None, description="Maximum size of formula"),
):
    """Download all nodes in the evolution graph."""
    filters = ["n.num_vars = $num_vars", "n.width = $width"]
    params = {"num_vars": num_vars, "width": width}

    if size_constraint is not None:
        filters.append("n.size <= $size_constraint")
        params["size_constraint"] = size_constraint

    filter_str = " AND ".join(filters)
    query = f"""
    MATCH   (n:FormulaNode)
    WHERE   {filter_str}
    OPTIONAL MATCH (n)--(m:FormulaNode)
    WITH    n, 
            sum(CASE WHEN m.avgQ < n.avgQ THEN 1 ELSE 0 END) AS in_degree,
            sum(CASE WHEN m.avgQ > n.avgQ THEN 1 ELSE 0 END) AS out_degree
    RETURN  n.formula_id AS formula_id,
            n.avgQ AS avgQ,
            n.visited_counter AS visited_counter,
            n.num_vars AS num_vars,
            n.width AS width,
            n.size AS size,
            in_degree,
            out_degree
    """
    
    with neo4j_driver.session() as session:
        response = session.run(query, **params)
        records = list(response)

    return {"nodes": [record.data() for record in records]}


# @app.get("/evolution_graph/subgraph", response_model=SubgraphResponse)
# async def get_evolution_subgraph(
#     num_vars: int = Query(..., description="Number of variables"),
#     width: int = Query(..., description="Width of the formula")
# ):
#     """Retrieve the evolution subgraph of active nodes."""
#     # Stub implementation
#     return SubgraphResponse(
#         nodes=[],
#         edges=[]
#     )


# @app.post("/formula/add", response_model=FormulaResponse, status_code=201)
# async def add_formula(formula: AddFormulaRequest):
#     """Add a new formula to the warehouse, updating the isomorphism hash table and evolution graph."""
#     # Stub implementation
#     new_id = str(uuid.uuid4())
#     return FormulaResponse(id=new_id)


# @app.post("/evolution_graph/subgraph", response_model=SuccessResponse, status_code=201)
# async def add_subgraph(subgraph: SubgraphRequest):
#     """Add a new subgraph to the evolution graph of a formula."""
#     # Stub implementation
#     return SuccessResponse(message="Subgraph added successfully")


# @app.post("/evolution_graph/contract_edge", response_model=SuccessResponse)
# async def contract_edge(request: ContractEdgeRequest):
#     """Contract an edge in the evolution graph; one node will be deactivated."""
#     # Stub implementation
#     return SuccessResponse(message="Edge contracted successfully")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
