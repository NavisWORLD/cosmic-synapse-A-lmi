"""
Integration Tests for A-LMI System
Tests complete data flow: crawler → LightToken → storage → query
"""

import pytest
import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import LightToken
from core.event_bus import EventBus
from memory.vector_db_client import VectorDBClient
from memory.tkg_client import TKGClient
from memory.minio_client import MinIOClient
# Note: ProcessingCore and GlobalCrawler not directly imported for testing
# as they require system-level setup (CLIP, Kafka, etc.)

logger = logging.getLogger(__name__)


class TestIntegrationE2E:
    """End-to-end integration tests."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.test_url = "https://example.com"
        self.test_text = "This is a test document about artificial intelligence and cosmic patterns."
        
    @pytest.mark.asyncio
    async def test_complete_data_flow(self):
        """
        Test: Crawler → Processing → Storage → Query
        
        NOTE: This test requires Docker services running (Kafka, Milvus, MinIO, Neo4j).
        Skip if services are not available.
        """
        logger.info("Starting E2E integration test...")
        
        # Check if services are available
        import socket
        
        def check_service(host, port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            except:
                return False
        
        if not check_service("localhost", 9092):
            pytest.skip("Kafka not available - skipping integration test")
        
        # Step 1: Setup services
        logger.info("[1/5] Setting up services...")
        event_bus = None
        vector_db = None
        minio_client = None
        
        try:
            event_bus = EventBus(
                bootstrap_servers=["localhost:9092"],
                topics={
                    "raw.web.content": "raw.web.content",
                    "processed.lighttokens": "processed.lighttokens"
                }
            )
            
            event_bus.connect()
            vector_db = VectorDBClient()
            vector_db.connect()
            
            minio_client = MinIOClient(
                endpoint="localhost:9000",
                access_key="minioadmin",
                secret_key="minioadmin",
                bucket="raw-data-archive"
            )
            minio_client.connect()
            
            # Step 2: Create test LightToken
            logger.info("[2/5] Creating test LightToken...")
            token = LightToken(
                source_uri=self.test_url,
                modality="text",
                raw_data_ref=self.test_url,
                content_text=self.test_text
            )
            
            # Simulate embedding (in real test, would use CLIP)
            import numpy as np
            test_embedding = np.random.rand(1536).astype(np.float32)
            token.set_semantic_embedding(test_embedding)
            
            # Set perceptual hash
            token.set_perceptual_hash(self.test_text.encode('utf-8'))
            
            # Compute spectral signature
            token.compute_spectral_signature()
            
            logger.info(f"Created token: {token.token_id}")
            
            # Step 3: Store in all tiers
            logger.info("[3/5] Storing in memory tiers...")
            
            # Tier 1: MinIO
            minio_ref = minio_client.upload_bytes(
                self.test_text.encode('utf-8'),
                f"test_{token.token_id}.txt"
            )
            logger.info(f"Stored in MinIO: {minio_ref}")
            
            # Tier 2: Milvus
            vector_db.insert_semantic_embedding(
                token_id=str(token.token_id),
                source_uri=token.source_uri,
                modality=token.modality,
                embedding=token.joint_embedding
            )
            
            if token.spectral_magnitude is not None:
                vector_db.insert_spectral_signature(
                    token_id=str(token.token_id),
                    source_uri=token.source_uri,
                    modality=token.modality,
                    spectral_magnitude=token.spectral_magnitude
                )
            
            logger.info("Stored in Milvus")
            
            # Tier 3: Neo4j (will be added after we have test entities)
            
            # Step 4: Query/search
            logger.info("[4/5] Testing queries...")
            
            # Semantic search
            semantic_results = vector_db.search_semantic(
                test_embedding,
                top_k=5
            )
            logger.info(f"Semantic search returned {len(semantic_results)} results")
            assert len(semantic_results) > 0
            
            # Spectral search
            if token.spectral_magnitude is not None:
                spectral_results = vector_db.search_spectral(
                    token.spectral_magnitude,
                    top_k=5
                )
                logger.info(f"Spectral search returned {len(spectral_results)} results")
                assert len(spectral_results) > 0
            
            # Step 5: Verify complete flow
            logger.info("[5/5] Verifying complete data flow...")
            assert token.joint_embedding is not None, "Embedding missing"
            assert token.perceptual_hash is not None, "Hash missing"
            assert token.spectral_signature is not None, "Spectral signature missing"
            assert len(semantic_results) > 0, "Retrieval failed"
            
            logger.info("✅ E2E integration test PASSED")
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            raise
        finally:
            # Cleanup - only disconnect if initialized
            if event_bus is not None:
                try:
                    event_bus.disconnect()
                except:
                    pass
            if vector_db is not None:
                try:
                    vector_db.disconnect()
                except:
                    pass
    
    def test_lighttoken_tripartite_structure(self):
        """Test: Verify LightToken has all three layers."""
        logger.info("Testing LightToken tripartite structure...")
        
        token = LightToken(
            source_uri="test://source",
            modality="text",
            raw_data_ref="test://ref",
            content_text="Test content"
        )
        
        # Layer 1: Semantic
        import numpy as np
        embedding = np.random.rand(1536).astype(np.float32)
        token.set_semantic_embedding(embedding)
        
        # Layer 2: Perceptual
        token.set_perceptual_hash(b"test data")
        
        # Layer 3: Spectral
        token.compute_spectral_signature()
        
        # Verify all layers present
        assert token.joint_embedding is not None, "Layer 1 missing"
        assert token.perceptual_hash is not None, "Layer 2 missing"
        assert token.spectral_signature is not None, "Layer 3 missing"
        
        # Verify spectral similarity works
        token2 = LightToken(
            source_uri="test://source2",
            modality="text",
            raw_data_ref="test://ref2",
            content_text="Different content"
        )
        token2.set_semantic_embedding(np.random.rand(1536).astype(np.float32))
        token2.set_perceptual_hash(b"different data")
        token2.compute_spectral_signature()
        
        similarity = token.spectral_similarity(token2)
        assert 0 <= similarity <= 1, "Spectral similarity out of range"
        
        logger.info("✅ Tripartite structure test PASSED")
    
    def test_spectral_signature_integration(self):
        """Test: Spectral signature computation and similarity."""
        logger.info("Testing spectral signature integration...")
        
        # Create two tokens
        import numpy as np
        
        token1 = LightToken(
            source_uri="test://1",
            modality="text",
            raw_data_ref="test://ref1",
            content_text="Test content one"
        )
        embedding1 = np.random.rand(1536).astype(np.float32)
        token1.set_semantic_embedding(embedding1)
        token1.set_perceptual_hash(b"data1")
        token1.compute_spectral_signature()
        
        token2 = LightToken(
            source_uri="test://2",
            modality="text",
            raw_data_ref="test://ref2",
            content_text="Test content two"
        )
        embedding2 = np.random.rand(1536).astype(np.float32)
        token2.set_semantic_embedding(embedding2)
        token2.set_perceptual_hash(b"data2")
        token2.compute_spectral_signature()
        
        # Test spectral similarity
        similarity = token1.spectral_similarity(token2)
        
        # Verify
        assert isinstance(similarity, float), "Similarity should be float"
        assert 0 <= similarity <= 1, "Similarity should be in [0,1]"
        
        logger.info(f"Spectral similarity: {similarity:.4f}")
        logger.info("✅ Spectral signature integration test PASSED")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

