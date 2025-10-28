"""
Mathematical Reasoning Module
Integration with OpenAI o1-mini for complex math reasoning
"""

import logging
from typing import Optional
import openai

logger = logging.getLogger(__name__)


class MathReasoner:
    """
    Mathematical reasoning module for A-LMI.
    
    Uses OpenAI o1-mini for:
    - Multi-step mathematical derivations
    - Equation solving and validation
    - Mathematical proof checking
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize math reasoner.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def reason(self, problem: str) -> str:
        """
        Perform mathematical reasoning.
        
        Args:
            problem: Mathematical problem or equation
            
        Returns:
            Reasoning result
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a mathematical reasoning assistant. Provide step-by-step solutions with derivations."},
                    {"role": "user", "content": problem}
                ],
                temperature=0,
            )
            
            result = response.choices[0].message.content
            logger.info(f"Mathematical reasoning completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in mathematical reasoning: {e}")
            raise
    
    def validate_proof(self, theorem: str, proof: str) -> bool:
        """
        Validate mathematical proof.
        
        Args:
            theorem: Theorem statement
            proof: Proof to validate
            
        Returns:
            True if proof is valid
        """
        try:
            prompt = f"Theorem: {theorem}\n\nProof: {proof}\n\nIs this proof valid? Explain why or why not."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a mathematical proof checker."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            
            result = response.choices[0].message.content
            is_valid = "valid" in result.lower()
            
            logger.info(f"Proof validation: {'valid' if is_valid else 'invalid'}")
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating proof: {e}")
            return False

