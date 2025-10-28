"""
Entry point for running the conversational UI standalone
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface.conversational_ui import ConversationalUI
from memory.vector_db_client import VectorDBClient
from memory.tkg_client import TKGClient
from utils.logging_config import setup_logging

async def main():
    """Main entry point for conversational UI."""
    logger = setup_logging(level="INFO")
    logger.info("Starting A-LMI Conversational Interface...")
    
    try:
        # Initialize clients
        vector_db = VectorDBClient(host="localhost", port=19530)
        vector_db.connect()
        logger.info("Connected to Milvus")
        
        tkg_client = TKGClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="neo4j_password",
            database="neo4j"
        )
        await tkg_client.connect()
        logger.info("Connected to Neo4j")
        
        # Launch UI
        ui = ConversationalUI(vector_db, tkg_client)
        logger.info("Launching Gradio interface...")
        ui.launch(share=True)
        
    except Exception as e:
        logger.error(f"Error starting UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

