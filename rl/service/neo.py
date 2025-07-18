from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://192.168.0.65:7687"
AUTH = ("haowei", "bread861122")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()