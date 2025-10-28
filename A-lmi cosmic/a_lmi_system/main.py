"""
Main entry point for A-LMI system
Davis Unified Intelligence System
"""

import asyncio
import yaml
import logging
from pathlib import Path

from .core.event_bus import EventBus
from .core.agent import AAgent
from .memory.tkg_client import TKGClient
from .utils.logging_config import setup_logging


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


async def main():
    """Main entry point for A-LMI system."""
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting Davis Unified Intelligence System (A-LMI v2.0)")
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    config = load_config(str(config_path))
    logger.info("Configuration loaded")
    
    # Initialize Event Bus
    event_bus = EventBus(
        bootstrap_servers=config['event_bus']['bootstrap_servers'],
        topics=config['event_bus']['topics']
    )
    event_bus.connect()
    logger.info("Event Bus initialized")
    
    # Initialize Memory Tier 3 (Knowledge Graph)
    tkg_client = TKGClient(
        uri=config['memory']['tier3']['uri'],
        user=config['memory']['tier3']['user'],
        password=config['memory']['tier3']['password']
    )
    await tkg_client.connect()
    logger.info("Knowledge Graph initialized")
    
    # Initialize Autonomous Agent
    agent = AAgent(event_bus, tkg_client)
    await agent.start()
    logger.info("Autonomous Agent started")
    
    # Keep running
    logger.info("A-LMI system is running. Press Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await agent.stop()
        event_bus.disconnect()
        await tkg_client.disconnect()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

