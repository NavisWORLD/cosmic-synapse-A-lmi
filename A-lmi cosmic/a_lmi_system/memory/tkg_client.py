"""
Temporal Knowledge Graph Client (Tier 3)
Connects to Neo4j for structured reasoning and gap detection
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver

logger = logging.getLogger(__name__)


class TKGClient:
    """
    Client for Neo4j Temporal Knowledge Graph.
    
    Implements:
    - Temporal edge tracking (valid_from, valid_to)
    - Context-aware relationships (acoustic_context)
    - Autonomous gap detection for hypothesis generation
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize TKG client.
        
        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[AsyncDriver] = None
        
    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            await self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Create indexes
            await self._create_indexes()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to Neo4j."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")
    
    async def _create_indexes(self):
        """Create necessary indexes for performance."""
        async with self.driver.session(database=self.database) as session:
            # Index on entity types
            await session.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
            
            # Index on token_id
            await session.run("CREATE INDEX token_id_index IF NOT EXISTS FOR ()-[r:HAS_CONTEXT]-() ON (r.source_token_id)")
            
        logger.debug("Created TKG indexes")
    
    async def create_entity(self, entity_id: str, entity_type: str, properties: Optional[Dict] = None):
        """
        Create or update an entity node.
        
        Args:
            entity_id: Unique entity identifier
            entity_type: Type of entity (Person, Concept, Location, etc.)
            properties: Additional properties
        """
        async with self.driver.session(database=self.database) as session:
            props = properties or {}
            props["entity_id"] = entity_id
            
            query = f"""
            MERGE (e:Entity {{entity_id: $entity_id}})
            SET e.entity_type = $entity_type,
                e += $properties
            RETURN e
            """
            
            await session.run(query, entity_id=entity_id, entity_type=entity_type, properties=props)
            
        logger.debug(f"Created/updated entity: {entity_id}")
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        valid_from: str,
        valid_to: Optional[str],
        source_token_id: Optional[str] = None,
        acoustic_context: Optional[str] = None,
        properties: Optional[Dict] = None
    ):
        """
        Create temporal relationship between entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type
            valid_from: Start timestamp
            valid_to: End timestamp (None for ongoing)
            source_token_id: Associated LightToken ID
            acoustic_context: Environmental acoustic classification
            properties: Additional properties
        """
        async with self.driver.session(database=self.database) as session:
            props = properties or {}
            props["valid_from"] = valid_from
            if valid_to:
                props["valid_to"] = valid_to
            if source_token_id:
                props["source_token_id"] = source_token_id
            if acoustic_context:
                props["acoustic_context"] = acoustic_context
            
            query = f"""
            MATCH (a:Entity {{entity_id: $source_id}}), (b:Entity {{entity_id: $target_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties
            RETURN r
            """
            
            await session.run(query, source_id=source_id, target_id=target_id, properties=props)
            
        logger.debug(f"Created relationship: {source_id} -[{rel_type}]-> {target_id}")
    
    async def find_knowledge_gaps(self, min_path_length: int = 2, max_path_length: int = 3) -> List[Dict[str, Any]]:
        """
        Find knowledge gaps in the graph.
        
        This implements autonomous hypothesis generation by finding entities
        that are connected via intermediate nodes (bridge) but lack direct relationships.
        
        This is THE CORE of "The Scientist Within" - the system actively seeks
        to discover missing knowledge.
        
        Args:
            min_path_length: Minimum path length to consider
            max_path_length: Maximum path length to consider
            
        Returns:
            List of gap patterns (potential hypotheses)
        """
        async with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (e1:Entity)-[r1*{min_path_length}..{max_path_length}]-(e2:Entity)
            WHERE id(e1) < id(e2)  // Avoid duplicates
            AND NOT (e1)-[*1..2]-(e2)  // No direct or 2-hop path exists
            WITH e1, e2, r1
            LIMIT 100
            RETURN DISTINCT
                e1.entity_id AS source_entity,
                e1.entity_type AS source_type,
                e2.entity_id AS target_entity,
                e2.entity_type AS target_type,
                size(r1) AS path_length
            """
            
            result = await session.run(query)
            gaps = []
            
            async for record in result:
                gaps.append({
                    "source_entity": record["source_entity"],
                    "source_type": record["source_type"],
                    "target_entity": record["target_entity"],
                    "target_type": record["target_type"],
                    "path_length": record["path_length"]
                })
            
            logger.info(f"Found {len(gaps)} knowledge gaps")
            return gaps
    
    async def query_graph(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute arbitrary Cypher query.
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher_query, parameters or {})
            records = []
            
            async for record in result:
                records.append(dict(record))
            
            return records

