"""
Hypothesis Generator: The "Scientist Within"
Autonomously generates testable hypotheses by finding knowledge gaps
"""

import logging
from typing import List, Dict, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from memory.tkg_client import TKGClient

logger = logging.getLogger(__name__)


class HypothesisGenerator:
    """
    Autonomous hypothesis generation based on knowledge gap detection.
    
    This component embodies "The Scientist Within" by:
    1. Querying the knowledge graph for structural gaps
    2. Formalizing gaps into testable hypotheses
    3. Prioritizing hypotheses by significance and testability
    """
    
    def __init__(self, tkg_client: TKGClient):
        """
        Initialize hypothesis generator.
        
        Args:
            tkg_client: Temporal Knowledge Graph client
        """
        self.tkg_client = tkg_client
        
    async def find_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """
        Find knowledge gaps and generate hypotheses.
        
        Returns:
            List of hypothesis dictionaries
        """
        # Find gaps: entities connected via bridge but no direct link
        gaps = await self.tkg_client.find_knowledge_gaps(min_path_length=2, max_path_length=3)
        
        # Convert gaps into hypotheses
        hypotheses = []
        for gap in gaps:
            hypothesis = {
                "hypothesis_id": self._generate_hypothesis_id(gap),
                "question": f"What is the relationship between {gap['source_entity']} ({gap['source_type']}) and {gap['target_entity']} ({gap['target_type']})?",
                "source_entity": gap['source_entity'],
                "target_entity": gap['target_entity'],
                "source_type": gap['source_type'],
                "target_type": gap['target_type'],
                "path_length": gap['path_length'],
                "testability": self._assess_testability(gap),
                "significance": self._assess_significance(gap)
            }
            hypotheses.append(hypothesis)
            
            logger.info(f"Generated hypothesis: {hypothesis['question']}")
        
        return hypotheses
    
    def _generate_hypothesis_id(self, gap: Dict[str, Any]) -> str:
        """Generate unique hypothesis ID."""
        return f"HYP_{gap['source_entity']}_{gap['target_entity']}"
    
    def _assess_testability(self, gap: Dict[str, Any]) -> float:
        """
        Assess how testable a hypothesis is.
        
        Args:
            gap: Knowledge gap pattern
            
        Returns:
            Testability score [0, 1]
        """
        # Criteria: longer paths are less testable
        # But also more interesting (potentially novel connections)
        path_length = gap.get('path_length', 3)
        
        if path_length == 2:
            return 0.9  # Highly testable
        elif path_length == 3:
            return 0.6  # Moderately testable
        else:
            return 0.3  # Less testable
    
    def _assess_significance(self, gap: Dict[str, Any]) -> float:
        """
        Assess the significance of a hypothesis.
        
        Args:
            gap: Knowledge gap pattern
            
        Returns:
            Significance score [0, 1]
        """
        # Longer paths might indicate more significant gaps
        # (potential for novel discoveries)
        path_length = gap.get('path_length', 3)
        
        if path_length >= 3:
            return 0.9  # High significance
        else:
            return 0.5  # Moderate significance
    
    async def rank_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank hypotheses by combined score (testability + significance).
        
        Args:
            hypotheses: List of hypothesis dictionaries
            
        Returns:
            Sorted list of hypotheses (highest score first)
        """
        for hyp in hypotheses:
            hyp['combined_score'] = hyp['testability'] * 0.4 + hyp['significance'] * 0.6
        
        ranked = sorted(hypotheses, key=lambda x: x['combined_score'], reverse=True)
        return ranked

