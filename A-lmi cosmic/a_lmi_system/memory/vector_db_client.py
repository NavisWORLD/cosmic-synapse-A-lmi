"""
Vector Database Client (Tier 2)
Connects to Milvus for semantic and spectral similarity search
"""

import numpy as np
import logging
from typing import Optional, List
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

logger = logging.getLogger(__name__)


class VectorDBClient:
    """
    Client for Milvus vector database.
    
    Manages two collections:
    - semantic_embeddings: Layer 1 semantic vectors (1536D)
    - spectral_signatures: Layer 3 spectral magnitudes (1536D)
    """
    
    def __init__(self, host: str = "localhost", port: int = 19530):
        """
        Initialize vector database client.
        
        Args:
            host: Milvus server host
            port: Milvus server port
        """
        self.host = host
        self.port = port
        self.connected = False
        
        self.semantic_collection: Optional[Collection] = None
        self.spectral_collection: Optional[Collection] = None
        
    def connect(self):
        """Establish connection to Milvus."""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            self.connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
            # Initialize collections
            self._setup_collections()
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self):
        """Close connection to Milvus."""
        if self.connected:
            connections.disconnect(alias="default")
            self.connected = False
            logger.info("Disconnected from Milvus")
    
    def _setup_collections(self):
        """Create collections if they don't exist."""
        dimension = 1536
        
        # Semantic Embeddings Collection
        semantic_fields = [
            FieldSchema(name="token_id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
            FieldSchema(name="source_uri", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        
        semantic_schema = CollectionSchema(
            fields=semantic_fields,
            description="Semantic embeddings (Layer 1)"
        )
        
        if not utility.has_collection("semantic_embeddings"):
            semantic_col = Collection("semantic_embeddings", semantic_schema)
            semantic_col.create_index(
                field_name="embedding",
                index_params={"metric_type": "IP", "index_type": "HNSW"}
            )
            logger.info("Created semantic_embeddings collection")
        else:
            semantic_col = Collection("semantic_embeddings")
            
        self.semantic_collection = semantic_col
        
        # Spectral Signatures Collection
        spectral_fields = [
            FieldSchema(name="token_id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
            FieldSchema(name="source_uri", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="spectral_magnitude", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        
        spectral_schema = CollectionSchema(
            fields=spectral_fields,
            description="Spectral signatures (Layer 3)"
        )
        
        if not utility.has_collection("spectral_signatures"):
            spectral_col = Collection("spectral_signatures", spectral_schema)
            spectral_col.create_index(
                field_name="spectral_magnitude",
                index_params={"metric_type": "L2", "index_type": "HNSW"}
            )
            logger.info("Created spectral_signatures collection")
        else:
            spectral_col = Collection("spectral_signatures")
            
        self.spectral_collection = spectral_col
    
    def insert_semantic_embedding(self, token_id: str, source_uri: str, modality: str, embedding: np.ndarray):
        """Insert semantic embedding (Layer 1)."""
        if not self.connected:
            raise RuntimeError("Not connected to Milvus")
        
        data = [{
            "token_id": token_id,
            "source_uri": source_uri,
            "modality": modality,
            "embedding": embedding.tolist()
        }]
        
        self.semantic_collection.insert(data)
        self.semantic_collection.flush()
        
        logger.debug(f"Inserted semantic embedding for {token_id}")
    
    def insert_spectral_signature(self, token_id: str, source_uri: str, modality: str, spectral_magnitude: np.ndarray):
        """Insert spectral signature (Layer 3)."""
        if not self.connected:
            raise RuntimeError("Not connected to Milvus")
        
        data = [{
            "token_id": token_id,
            "source_uri": source_uri,
            "modality": modality,
            "spectral_magnitude": spectral_magnitude.tolist()
        }]
        
        self.spectral_collection.insert(data)
        self.spectral_collection.flush()
        
        logger.debug(f"Inserted spectral signature for {token_id}")
    
    def search_semantic(self, query_embedding: np.ndarray, top_k: int = 10) -> List[dict]:
        """
        Search for similar tokens by semantic embedding.
        
        Args:
            query_embedding: Query vector (1536D)
            top_k: Number of results to return
            
        Returns:
            List of matching tokens with scores
        """
        if not self.connected:
            raise RuntimeError("Not connected to Milvus")
        
        search_params = {"metric_type": "IP", "params": {"ef": 128}}
        
        results = self.semantic_collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["token_id", "source_uri", "modality"]
        )
        
        matches = []
        for hits in results:
            for hit in hits:
                matches.append({
                    "token_id": hit.entity.get("token_id"),
                    "source_uri": hit.entity.get("source_uri"),
                    "modality": hit.entity.get("modality"),
                    "score": float(hit.score)
                })
        
        return matches
    
    def search_spectral(self, query_spectral: np.ndarray, top_k: int = 10) -> List[dict]:
        """
        Search for similar tokens by spectral signature (Layer 3).
        
        This is THE INNOVATION - finding tokens that have similar
        "spectral texture" even if their semantic content differs.
        
        Args:
            query_spectral: Query spectral magnitude (1536D)
            top_k: Number of results to return
            
        Returns:
            List of matching tokens with scores
        """
        if not self.connected:
            raise RuntimeError("Not connected to Milvus")
        
        search_params = {"metric_type": "L2", "params": {"ef": 128}}
        
        results = self.spectral_collection.search(
            data=[query_spectral.tolist()],
            anns_field="spectral_magnitude",
            param=search_params,
            limit=top_k,
            output_fields=["token_id", "source_uri", "modality"]
        )
        
        matches = []
        for hits in results:
            for hit in hits:
                matches.append({
                    "token_id": hit.entity.get("token_id"),
                    "source_uri": hit.entity.get("source_uri"),
                    "modality": hit.entity.get("modality"),
                    "score": float(hit.score)
                })
        
        return matches

