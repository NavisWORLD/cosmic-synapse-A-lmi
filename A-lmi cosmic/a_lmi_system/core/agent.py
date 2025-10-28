"""
The Autonomous Agent: Central orchestrator for A-LMI system
Implements the Perception-Cognition-Action loop
"""

import logging
import asyncio
from typing import Optional
from datetime import datetime

from .event_bus import EventBus
from ..services.reasoning_engine.hypothesis_generator import HypothesisGenerator
from ..services.reasoning_engine.action_planner import ActionPlanner
from ..memory.tkg_client import TKGClient

logger = logging.getLogger(__name__)


class AAgent:
    """
    The Autonomous Agent - The "Scientist Within"
    
    This is the central cognitive module that:
    1. Observes the world (via EventBus)
    2. Generates hypotheses (finds knowledge gaps)
    3. Plans experiments (designs actions)
    4. Executes actions (creates tasks)
    5. Learns from results (updates knowledge)
    
    This creates a closed loop of self-directed learning.
    """
    
    def __init__(self, event_bus: EventBus, tkg_client: TKGClient):
        """
        Initialize the autonomous agent.
        
        Args:
            event_bus: Event bus for communication
            tkg_client: Neo4j client for knowledge graph access
        """
        self.event_bus = event_bus
        self.tkg_client = tkg_client
        self.hypothesis_generator = HypothesisGenerator(tkg_client)
        self.action_planner = ActionPlanner(event_bus)
        
        self.running = False
        self.last_hypothesis_time = datetime.now()
        
    async def start(self):
        """Start the autonomous agent loop."""
        logger.info("Starting A-LMI Autonomous Agent...")
        self.running = True
        
        # Subscribe to relevant events
        self.event_bus.subscribe("processed_lighttokens", self._on_new_token)
        
        # Start autonomous reasoning loop
        asyncio.create_task(self._autonomous_loop())
        
    async def stop(self):
        """Stop the autonomous agent."""
        logger.info("Stopping A-LMI Autonomous Agent...")
        self.running = False
        
    async def _autonomous_loop(self):
        """
        Main autonomous reasoning loop.
        
        This implements "The Scientist Within" - the system actively seeks
        to expand its knowledge by finding gaps and conducting experiments.
        """
        while self.running:
            try:
                # Check if it's time to generate new hypotheses
                # (Throttle to avoid excessive queries)
                from datetime import timedelta
                if datetime.now() - self.last_hypothesis_time > timedelta(minutes=5):
                    hypotheses = await self.hypothesis_generator.find_knowledge_gaps()
                    
                    for hypothesis in hypotheses:
                        logger.info(f"Generated hypothesis: {hypothesis}")
                        
                        # Design experiment to test hypothesis
                        experiment = await self.action_planner.design_experiment(hypothesis)
                        logger.info(f"Designed experiment: {experiment}")
                        
                        # Execute experiment (queue crawler tasks, etc.)
                        await self.action_planner.execute_experiment(experiment)
                        logger.info("Experiment queued for execution")
                    
                    self.last_hypothesis_time = datetime.now()
                
                # Sleep to avoid tight loop
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(5)
    
    def _on_new_token(self, event: dict):
        """
        Handle new LightToken event.
        
        Args:
            event: Event containing new token data
        """
        logger.debug(f"Agent observed new token: {event.get('token_id')}")
        # Future: Analyze token for opportunities, contradictions, etc.

