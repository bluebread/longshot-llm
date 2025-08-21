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
    DownloadHyperNodesResponse,
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
isohash_table = Redis(db=0, **redis_config)

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
        isohash_table.ping()
        # Test Neo4j connection
        neo4j_driver.verify_connectivity()
    except Exception as e:
        raise HTTPException(status_code=500, detail="MongoDB connection failed")
    
    # Initialize MongoDB resources
    try:
        # TODO: add validator
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
    isohash_table.close()
    neo4j_driver.close()
    
app = FastAPI(
    title="Warehouse API",
    lifespan=lifespan,
)
trajectory_table = mongodb["TrajectoryTable"]

# Likely isomorphic formulas endpoints
@app.get("/formula/likely_isomorphic", response_model=LikelyIsomorphicResponse)
async def get_likely_isomorphic(wl_hash: str = Query(..., description="Weisfeiler-Lehman hash")):
    """Retrieve IDs of likely isomorphic formulas."""
    formula_ids = list(isohash_table.smembers(wl_hash))
    return LikelyIsomorphicResponse(wl_hash=wl_hash, isomorphic_ids=formula_ids)


@app.post("/formula/likely_isomorphic", response_model=SuccessResponse, status_code=201)
async def add_likely_isomorphic(request: LikelyIsomorphicRequest):
    """Add a likely isomorphic formula."""
    isohash_table.sadd(request.wl_hash, request.node_id)
    return SuccessResponse(message="Likely isomorphic formula added successfully")

@app.delete("/formula/likely_isomorphic", response_model=SuccessResponse)
async def delete_likely_isomorphic(wl_hash: str = Query(..., description="Weisfeiler-Lehman hash")):
    """Delete a likely isomorphic formula."""
    isohash_table.delete(wl_hash)
    return SuccessResponse(message="Likely isomorphic formula deleted successfully")

# Trajectory endpoints
@app.get("/trajectory", response_model=QueryTrajectoryInfoResponse)
async def get_trajectory(traj_id: str = Query(..., description="Trajectory UUID")):
    """Retrieve a trajectory by its ID."""
    trajectory_doc = trajectory_table.find_one({"_id": traj_id})
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
    
    return TrajectoryResponse(traj_id=trajectory_doc["_id"])


@app.put("/trajectory", response_model=SuccessResponse)
async def update_trajectory(trajectory: UpdateTrajectoryRequest):
    """Update an existing trajectory."""
    trajectory_id = trajectory.traj_id
    data = trajectory.model_dump(exclude_unset=True, exclude={"traj_id"})
    update_data = {}

    if not data:
        raise HTTPException(status_code=400, detail="No update data provided")

    if "steps" in data:
        update_data = {
            f"steps.{step['order']}": {
                "token_type": step["token_type"],
                "token_literals": step["token_literals"],
                "cur_avgQ": step["cur_avgQ"],
            }
            for step in data["steps"]
        }
        
    result = trajectory_table.update_one(
        {"_id": trajectory_id},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    
    return SuccessResponse(message="Trajectory updated successfully")


@app.delete("/trajectory", response_model=SuccessResponse)
async def delete_trajectory(traj_id: str = Query(..., description="Trajectory UUID")):
    """Delete a trajectory."""
    result = trajectory_table.delete_one({"_id": traj_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Trajectory not found")
        
    return SuccessResponse(message="Trajectory deleted successfully")


# Evolution graph node endpoints
@app.get("/evolution_graph/node", response_model=QueryEvolutionGraphNode)
async def get_evolution_graph_node(node_id: str = Query(..., description="Node UUID")):
    """Retrieve a node in the evolution graph by its ID with integrated formula data."""
    query = """
    MATCH   (n:FormulaNode {formula_id: $node_id})
    OPTIONAL MATCH (n)--(m:FormulaNode)
    WITH    n, 
            sum(CASE WHEN m.avgQ < n.avgQ THEN 1 ELSE 0 END) AS in_degree,
            sum(CASE WHEN m.avgQ > n.avgQ THEN 1 ELSE 0 END) AS out_degree
    RETURN  n.formula_id AS node_id,
            n.avgQ AS avgQ,
            n.num_vars AS num_vars,
            n.width AS width,
            n.size AS size,
            n.wl_hash AS wl_hash,
            datetime(n.timestamp) AS timestamp,
            n.traj_id AS traj_id,
            n.traj_slice AS traj_slice,
            in_degree,
            out_degree
    """
    with neo4j_driver.session() as session:
        result = session.run(query, node_id=node_id).single()
    
    if not result:
        raise HTTPException(status_code=404, detail="Node not found")
    
    data = result.data()
    # Convert Neo4j DateTime to Python datetime
    if data["timestamp"] and hasattr(data["timestamp"], "to_native"):
        data["timestamp"] = data["timestamp"].to_native()
    
    return QueryEvolutionGraphNode(**data)


@app.post("/evolution_graph/node", response_model=NodeResponse, status_code=201)
async def create_evolution_graph_node(node: CreateNodeRequest):
    """Add a new node to the evolution graph with integrated formula data."""
    query = """
    MERGE (n:FormulaNode {formula_id: $node_id})
    ON CREATE SET n.avgQ = $avgQ,
                  n.num_vars = $num_vars,
                  n.width = $width,
                  n.size = $size,
                  n.wl_hash = $wl_hash,
                  n.traj_id = $traj_id,
                  n.traj_slice = $traj_slice,
                  n.timestamp = $timestamp
    RETURN n.formula_id AS node_id
    """
    node_data = node.model_dump()
    node_data["timestamp"] = datetime.now()
    
    with neo4j_driver.session() as session:
        result = session.run(query, **node_data).single()

    if not result:
        raise HTTPException(status_code=500, detail="Failed to create node")

    return NodeResponse(node_id=result["node_id"])


@app.put("/evolution_graph/node", response_model=SuccessResponse)
async def update_evolution_graph_node(node: UpdateNodeRequest):
    """Update an existing node with integrated formula data."""
    node_id = node.node_id
    update_data = node.model_dump(exclude_unset=True, exclude={"node_id"})

    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")

    set_clauses = [f"n.{key} = ${key}" for key in update_data.keys()]

    query = f"""
    MATCH   (n:FormulaNode {{formula_id: $node_id}})
    SET     {', '.join(set_clauses)}
    RETURN  n
    """
    
    params = {"node_id": node_id, **update_data}

    with neo4j_driver.session() as session:
        result = session.run(query, params).single()

    if not result:
        raise HTTPException(status_code=404, detail="Node not found")

    return SuccessResponse(message="Node updated successfully")


@app.delete("/evolution_graph/node", response_model=SuccessResponse)
async def delete_evolution_graph_node(node_id: str = Query(..., description="Node UUID")):
    """Delete a node and its relationships."""
    query = """
    MATCH   (n:FormulaNode {formula_id: $node_id})
    WITH    n, n.formula_id AS node_id
    DETACH  DELETE n
    RETURN  node_id
    """
    with neo4j_driver.session() as session:
        result = session.run(query, node_id=node_id).single()

    if not result:
        raise HTTPException(status_code=404, detail="Node not found")

    return SuccessResponse(message="Node deleted successfully")

# High-level API endpoints
@app.get("/formula/definition", response_model=QueryFormulaDefinitionResponse)
async def get_formula_definition(node_id: str = Query(..., description="Node UUID")):
    """Retrieve the full definition of a formula by its ID using trajectory reconstruction."""
    # Get the formula node from Neo4j
    query = """
    MATCH (n:FormulaNode {formula_id: $node_id})
    RETURN n.traj_id AS traj_id, n.traj_slice AS traj_slice
    """
    
    with neo4j_driver.session() as session:
        result = session.run(query, node_id=node_id).single()
    
    if not result:
        raise HTTPException(status_code=404, detail="Formula not found")
    
    traj_id = result["traj_id"]
    traj_slice = result["traj_slice"]
    
    # Get the trajectory from MongoDB
    trajectory_doc = trajectory_table.find_one({"_id": traj_id})
    if not trajectory_doc:
        # Return empty definition if trajectory not found
        return QueryFormulaDefinitionResponse(id=id, definition=[])
    
    # Reconstruct the formula definition up to the specified slice
    definition = set()
    steps = trajectory_doc.get("steps", [])
    
    # Process steps up to and including the specified slice
    for i, step in enumerate(steps):
        if i > traj_slice:
            break
            
        ttype = step["token_type"]
        literals = step["token_literals"]

        if ttype == 0:  # ADD
            definition.add(literals)
        elif ttype == 1:  # DELETE
            definition.discard(literals)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown token type: {ttype}")

    return QueryFormulaDefinitionResponse(id=node_id, definition=list(definition))


@app.post("/evolution_graph/path", response_model=SuccessResponse, status_code=201)
async def create_new_path(request: CreateNewPathRequest):
    """Create a new path in the evolution graph."""
    query = """
    WITH    $path AS nodes
    UNWIND  range(0, size(nodes) - 2) AS i
    MERGE   (n:FormulaNode {formula_id: nodes[i]})
    MERGE   (m:FormulaNode {formula_id: nodes[i + 1]})
    MERGE   (n)-[:EVOLVED_TO]->(m)
    WITH    n, m
    WHERE   n.avgQ IS NOT NULL AND m.avgQ IS NOT NULL AND n.avgQ = m.avgQ
    MERGE   (n)-[:SAME_Q]->(m)
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
    RETURN  n.formula_id AS node_id,
            n.avgQ AS avgQ,
            n.num_vars AS num_vars,
            n.width AS width,
            n.size AS size,
            n.wl_hash AS wl_hash,
            datetime(n.timestamp) AS timestamp,
            n.traj_id AS traj_id,
            n.traj_slice AS traj_slice,
            in_degree,
            out_degree
    """
    
    with neo4j_driver.session() as session:
        response = session.run(query, **params)
        records = list(response)

    nodes = []
    for record in records:
        data = record.data()
        # Convert Neo4j DateTime to Python datetime
        if data["timestamp"] and hasattr(data["timestamp"], "to_native"):
            data["timestamp"] = data["timestamp"].to_native()
        nodes.append(data)

    return {"nodes": nodes}

@app.get("/evolution_graph/download_hypernodes", response_model=DownloadHyperNodesResponse)
async def download_evolution_graph_hypernodes(
    num_vars: int = Query(..., description="Number of variables in formula"),
    width: int = Query(..., description="Width of formula"),
    size_constraint: int | None = Query(None, description="Maximum size of formula"),
):
    """Download all hypernodes in the evolution graph."""
    predicate = "node.num_vars = $num_vars AND node.width = $width"
    params = {"num_vars": num_vars, "width": width}
    
    if size_constraint is not None:
        predicate += " AND node.size <= $size_constraint"
        params["size_constraint"] = size_constraint
    
    gid = uuid.uuid4()
    gs = f"sameQ_{gid}"
    
    proj_cmd = """
    CALL gds.graph.project(
        $gs,
        'FormulaNode',
        { SAME_Q: { type: 'SAME_Q', orientation: 'UNDIRECTED' } }
    )
    YIELD graphName, nodeCount, relationshipCount
    """
    query = f"""
    CALL    gds.wcc.stream($gs)
    YIELD   nodeId, componentId
    WITH    componentId, 
            gds.util.asNode(nodeId) AS node
    WHERE   {predicate}
    WITH    componentId,
            collect(node.formula_id) AS nodes
    WHERE   size(nodes) > 1
    RETURN  componentId AS hnid,
            nodes
    """
    drop_cmd = """
    CALL    gds.graph.drop($gs)
    YIELD   graphName, nodeCount, relationshipCount
    """
    
    with neo4j_driver.session() as session:
        session.run(proj_cmd, gs=gs)
        response = session.run(query, gs=gs, **params)
        records = list(response)
        session.run(drop_cmd, gs=gs)

    return { "hypernodes": [record.data() for record in records] }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
