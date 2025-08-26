from typing import List, Dict, Any, Optional
from loguru import logger
import json
import re

from llm.factory import LLMFactory
from llm.prompts import SUBQUESTION_DECOMPOSITION_SYSTEM_PROMPT, SUBQUESTION_DECOMPOSITION_PROMPT
from config import config


class SubQuestionPlanner:
    """Sub-question decomposition planner for multi-hop complex queries.
    
    This class uses LLM to decompose complex questions into multiple independent
    sub-questions that can be processed in parallel.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the SubQuestionPlanner.
        
        Args:
            llm_client: Optional LLM client. If not provided, will create one using factory.
        """
        self.llm_client = llm_client or LLMFactory.create_provider()
        
        # Get configuration
        self.max_subquestions = config.get('query.subquestion.max_subquestions', 5)
        self.min_subquestions = config.get('query.subquestion.min_subquestions', 2)
        
        logger.info(f"SubQuestionPlanner initialized with max_subquestions={self.max_subquestions}, min_subquestions={self.min_subquestions}")
    
    def decompose(self, query: str) -> List[str]:
        """Decompose a complex query into multiple sub-questions.
        
        Args:
            query: The original complex query
            
        Returns:
            List of sub-questions
        """
        try:
            # Check if query is complex enough to decompose
            if not self._is_complex_query(query):
                logger.info(f"Query is not complex enough for decomposition: {query}")
                return [query]
            
            # Generate sub-questions using LLM
            sub_questions = self._generate_subquestions(query)
            
            # Validate and filter sub-questions
            sub_questions = self._validate_subquestions(sub_questions, query)
            
            logger.info(f"Decomposed query into {len(sub_questions)} sub-questions: {sub_questions}")
            return sub_questions
            
        except Exception as e:
            logger.error(f"Error in query decomposition: {e}")
            # Fallback to original query
            return [query]
    
    def _is_complex_query(self, query: str) -> bool:
        """Check if a query is complex enough to warrant decomposition.
        
        Args:
            query: The query to check
            
        Returns:
            True if query is complex, False otherwise
        """
        # Simple heuristics to determine query complexity
        complexity_indicators = [
            len(query.split()) > 10,  # Long queries
            'and' in query.lower(),   # Multiple conditions
            'or' in query.lower(),    # Alternative conditions
            '?' in query and query.count('?') > 1,  # Multiple questions
            any(word in query.lower() for word in ['compare', 'difference', 'relationship', 'between']),  # Comparative queries
            any(word in query.lower() for word in ['why', 'how', 'what', 'when', 'where']) and len(query.split()) > 8,  # Complex analytical queries
        ]
        
        return sum(complexity_indicators) >= 2
    
    def _generate_subquestions(self, query: str) -> List[str]:
        """Generate sub-questions using LLM.
        
        Args:
            query: The original query
            
        Returns:
            List of generated sub-questions
        """
        try:
            # Prepare prompt
            prompt = SUBQUESTION_DECOMPOSITION_PROMPT.format(query=query)
            
            # Generate response using LLM
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=SUBQUESTION_DECOMPOSITION_SYSTEM_PROMPT,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=1024
            )
            
            # Parse JSON response
            sub_questions = self._parse_subquestions_response(response)
            
            return sub_questions
            
        except Exception as e:
            logger.error(f"Error generating sub-questions: {e}")
            raise
    
    def _parse_subquestions_response(self, response: str) -> List[str]:
        """Parse LLM response to extract sub-questions.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of parsed sub-questions
        """
        try:
            # Clean response - remove markdown code blocks if present
            cleaned_response = re.sub(r'```json\s*|```\s*', '', response.strip())
            
            # Parse JSON
            parsed = json.loads(cleaned_response)
            
            # Extract sub-questions
            if isinstance(parsed, dict) and 'sub_questions' in parsed:
                sub_questions = parsed['sub_questions']
            elif isinstance(parsed, list):
                sub_questions = parsed
            else:
                raise ValueError(f"Unexpected response format: {parsed}")
            
            # Ensure all items are strings
            sub_questions = [str(q).strip() for q in sub_questions if q and str(q).strip()]
            
            return sub_questions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}. Response: {response}")
            # Try to extract questions using regex as fallback
            return self._extract_questions_fallback(response)
        except Exception as e:
            logger.error(f"Error parsing sub-questions response: {e}")
            raise
    
    def _extract_questions_fallback(self, response: str) -> List[str]:
        """Fallback method to extract questions using regex.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of extracted questions
        """
        # Look for question patterns
        question_patterns = [
            r'"([^"]*\?)"',  # Questions in quotes
            r'\d+\.\s*([^\n]*\?)',  # Numbered questions
            r'-\s*([^\n]*\?)',  # Bulleted questions
        ]
        
        questions = []
        for pattern in question_patterns:
            matches = re.findall(pattern, response)
            questions.extend([q.strip() for q in matches if q.strip()])
        
        return questions[:self.max_subquestions] if questions else []
    
    def _validate_subquestions(self, sub_questions: List[str], original_query: str) -> List[str]:
        """Validate and filter sub-questions.
        
        Args:
            sub_questions: List of generated sub-questions
            original_query: The original query for reference
            
        Returns:
            List of validated sub-questions
        """
        if not sub_questions:
            logger.warning("No sub-questions generated, using original query")
            return [original_query]
        
        # Filter out invalid questions
        valid_questions = []
        for q in sub_questions:
            if self._is_valid_question(q):
                valid_questions.append(q)
        
        # Ensure we have the right number of questions
        if len(valid_questions) < self.min_subquestions:
            logger.warning(f"Too few valid sub-questions ({len(valid_questions)}), using original query")
            return [original_query]
        
        if len(valid_questions) > self.max_subquestions:
            valid_questions = valid_questions[:self.max_subquestions]
            logger.info(f"Truncated to {self.max_subquestions} sub-questions")
        
        return valid_questions
    
    def _is_valid_question(self, question: str) -> bool:
        """Check if a question is valid.
        
        Args:
            question: The question to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not question or len(question.strip()) < 5:
            return False
        
        # Should end with question mark or be a clear question
        question = question.strip()
        if not (question.endswith('?') or any(question.lower().startswith(word) for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which'])):
            return False
        
        # Should not be too similar to avoid redundancy
        return True