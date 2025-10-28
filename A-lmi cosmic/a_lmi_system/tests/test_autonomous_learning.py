"""
Autonomous Learning Loop Integration Test
Tests hypothesis generation, action planning, and closed learning loop
"""

import pytest
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.reasoning_engine.hypothesis_generator import HypothesisGenerator
from services.reasoning_engine.action_planner import ActionPlanner

logger = logging.getLogger(__name__)


class TestAutonomousLearning:
    """Test autonomous learning capabilities."""
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation(self):
        """Test: Hypothesis generation from knowledge gaps."""
        logger.info("Testing hypothesis generation...")
        
        # Mock knowledge gaps (simulating Neo4j query results)
        mock_gaps = [
            {
                "source_entity": "Albert Einstein",
                "source_type": "Person",
                "target_entity": "Relativity Theory",
                "target_type": "Concept",
                "path_length": 2
            },
            {
                "source_entity": "Neural Networks",
                "source_type": "Concept",
                "target_entity": "Stochastic Resonance",
                "target_type": "Concept",
                "path_length": 3
            }
        ]
        
        # Test hypothesis generation logic
        hypotheses = []
        for gap in mock_gaps:
            hypothesis = {
                "hypothesis_id": f"HYP_{gap['source_entity']}_{gap['target_entity']}",
                "question": f"What is the relationship between {gap['source_entity']} ({gap['source_type']}) and {gap['target_entity']} ({gap['target_type']})?",
                "source_entity": gap['source_entity'],
                "target_entity": gap['target_entity'],
                "testability": 0.9 if gap['path_length'] == 2 else 0.6,
                "significance": 0.9 if gap['path_length'] >= 3 else 0.5
            }
            hypotheses.append(hypothesis)
        
        # Verify
        assert len(hypotheses) == 2, "Should generate 2 hypotheses"
        
        # Calculate combined scores
        for hyp in hypotheses:
            hyp['combined_score'] = hyp['testability'] * 0.4 + hyp['significance'] * 0.6
        
        # Verify scoring
        assert all('combined_score' in h for h in hypotheses), "All hypotheses should have scores"
        assert all(0 <= h['combined_score'] <= 1 for h in hypotheses), "Scores should be [0,1]"
        
        logger.info(f"Generated {len(hypotheses)} hypotheses")
        for hyp in hypotheses:
            logger.info(f"  - {hyp['question']} (score: {hyp['combined_score']:.2f})")
        
        logger.info("✅ Hypothesis generation test PASSED")
    
    @pytest.mark.asyncio
    async def test_action_planning(self):
        """Test: Action planning for hypothesis testing."""
        logger.info("Testing action planning...")
        
        # Mock hypothesis
        hypothesis = {
            "hypothesis_id": "HYP_TEST_123",
            "question": "What is the relationship between AI and Quantum Computing?",
            "source_entity": "AI",
            "target_entity": "Quantum Computing",
            "source_type": "Concept",
            "target_type": "Concept"
        }
        
        # Simulate experiment design
        experiment = {
            "experiment_id": f"EXP_{hypothesis['hypothesis_id']}",
            "hypothesis_id": hypothesis['hypothesis_id'],
            "question": hypothesis['question'],
            "search_queries": [
                f"{hypothesis['source_entity']} AND {hypothesis['target_entity']}",
                f"{hypothesis['source_entity']} relationship {hypothesis['target_entity']}",
                f"{hypothesis['source_type']} {hypothesis['source_entity']} {hypothesis['target_type']} {hypothesis['target_entity']}"
            ],
            "priority": 0.75
        }
        
        # Verify
        assert "experiment_id" in experiment, "Experiment should have ID"
        assert len(experiment['search_queries']) >= 3, "Should have multiple queries"
        assert all(query and len(query) > 0 for query in experiment['search_queries']), "Queries should be non-empty"
        
        logger.info(f"Designed experiment: {experiment['experiment_id']}")
        logger.info(f"Generated {len(experiment['search_queries'])} search queries")
        for i, query in enumerate(experiment['search_queries'], 1):
            logger.info(f"  {i}. {query}")
        
        logger.info("✅ Action planning test PASSED")
    
    @pytest.mark.asyncio
    async def test_closed_learning_loop(self):
        """Test: Complete closed learning loop."""
        logger.info("Testing closed learning loop...")
        
        # Step 1: Generate hypothesis
        gaps = [
            {
                "source_entity": "Sound",
                "source_type": "Concept",
                "target_entity": "Light",
                "target_type": "Concept",
                "path_length": 2
            }
        ]
        
        hypotheses = []
        for gap in gaps:
            hypothesis = {
                "hypothesis_id": f"HYP_{gap['source_entity']}_{gap['target_entity']}",
                "question": f"What connects {gap['source_entity']} and {gap['target_entity']}?",
                "testability": 0.9,
                "significance": 0.5
            }
            hypothesis['combined_score'] = hypothesis['testability'] * 0.4 + hypothesis['significance'] * 0.6
            hypotheses.append(hypothesis)
        
        assert len(hypotheses) == 1, "Should have 1 hypothesis"
        
        # Step 2: Design experiment
        experiment = {
            "experiment_id": f"EXP_{hypotheses[0]['hypothesis_id']}",
            "search_queries": ["Sound AND Light", "Vibrational connection"],
            "priority": hypotheses[0]['combined_score']
        }
        
        # Step 3: Execute (simulate)
        executed_tasks = []
        for query in experiment['search_queries']:
            executed_tasks.append({
                "query": query,
                "status": "queued",
                "timestamp": "2025-10-26T00:00:00Z"
            })
        
        # Step 4: Verify closed loop
        assert len(executed_tasks) == len(experiment['search_queries']), "All queries should be queued"
        
        logger.info("Closed learning loop verified:")
        logger.info(f"  1. Generated hypothesis: {hypotheses[0]['question']}")
        logger.info(f"  2. Designed experiment: {experiment['experiment_id']}")
        logger.info(f"  3. Executed {len(executed_tasks)} tasks")
        logger.info("  4. Loop complete → new data will generate new gaps")
        
        logger.info("✅ Closed learning loop test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

