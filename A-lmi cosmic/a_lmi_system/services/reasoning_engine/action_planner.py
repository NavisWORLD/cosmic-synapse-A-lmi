"""
Action Planner: Design and execute experiments
Translates hypotheses into concrete actions (crawler tasks, etc.)
"""

import logging
from typing import Dict, Any, List
from a_lmi_system.core.event_bus import EventBus

logger = logging.getLogger(__name__)


class ActionPlanner:
    """
    Autonomous action planning for hypothesis testing.
    
    This component:
    1. Designs experiments to test hypotheses
    2. Generates targeted crawler queries
    3. Publishes actions to the event bus
    4. Creates closed loop: hypothesis → action → data → knowledge
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize action planner.
        
        Args:
            event_bus: Event bus for publishing actions
        """
        self.event_bus = event_bus
        
    async def design_experiment(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design an experiment to test a hypothesis.
        
        Args:
            hypothesis: Hypothesis dictionary
            
        Returns:
            Experiment plan dictionary
        """
        source = hypothesis['source_entity']
        target = hypothesis['target_entity']
        source_type = hypothesis['source_type']
        target_type = hypothesis['target_type']
        
        # Generate targeted search queries
        queries = self._generate_search_queries(source, target, source_type, target_type)
        
        experiment = {
            "experiment_id": f"EXP_{hypothesis['hypothesis_id']}",
            "hypothesis_id": hypothesis['hypothesis_id'],
            "question": hypothesis['question'],
            "search_queries": queries,
            "priority": hypothesis['combined_score'],
            "expected_outcome": f"Find documents containing both {source} and {target}"
        }
        
        logger.info(f"Designed experiment: {experiment['experiment_id']}")
        return experiment
    
    def _generate_search_queries(
        self,
        source: str,
        target: str,
        source_type: str,
        target_type: str
    ) -> List[str]:
        """
        Generate targeted search queries to test hypothesis.
        
        Args:
            source: Source entity
            target: Target entity
            source_type: Source entity type
            target_type: Target entity type
            
        Returns:
            List of search query strings
        """
        queries = [
            f"{source} AND {target}",
            f"{source} relationship {target}",
            f"{source_type} {source} {target_type} {target}",
            f"{source} connected to {target}"
        ]
        
        return queries
    
    async def execute_experiment(self, experiment: Dict[str, Any]):
        """
        Execute an experiment by publishing crawler tasks.
        
        Args:
            experiment: Experiment plan dictionary
        """
        # Publish high-priority crawler tasks to event bus
        for query in experiment['search_queries']:
            task = {
                "event_type": "crawler_task",
                "experiment_id": experiment['experiment_id'],
                "hypothesis_id": experiment['hypothesis_id'],
                "query": query,
                "priority": "high",
                "search_engine": "google",
                "max_results": 10
            }
            
            self.event_bus.publish("ai_commands", task)
            logger.info(f"Published crawler task: {query}")
        
        logger.info(f"Executed experiment: {experiment['experiment_id']}")

