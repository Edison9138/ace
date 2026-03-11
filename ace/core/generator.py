"""
Generator agent for ACE system.
Generates answers to questions using playbook and reflection.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from ..prompts.generator import GENERATOR_PROMPT
from llm import timed_llm_call

class Generator:
    """
    Generator agent that produces answers to questions using knowledge
    from a playbook and previous reflections.
    """
    
    def __init__(self, api_client, api_provider, model: str, max_tokens: int = 4096):
        """
        Initialize the Generator agent.
        
        Args:
            api_client: OpenAI client for LLM calls
            api_provider: API provider for LLM calls
            model: Model name to use for generation
            max_tokens: Maximum tokens for generation
        """
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
    
    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate an answer to a question using the playbook.
        
        Args:
            question: The question to answer
            playbook: The current playbook content
            context: Additional context for the question
            reflection: Previous reflection content
            use_json_mode: Whether to use JSON mode
            call_id: Unique identifier for this call
            log_dir: Directory for logging
            
        Returns:
            Tuple of (full_response, bullet_ids_used, call_info)
        """
        # Format the prompt
        prompt = GENERATOR_PROMPT.format(playbook, reflection, question, context)
        
        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="generator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode
        )
        
        # Extract bullet IDs if using retrieval and reason mode
        bullet_ids = []
        bullet_ids = self._extract_bullet_ids(response, use_json_mode)
        
        return response, bullet_ids, call_info

    def get_bullets_for_reflector(self, playbook: str, bullet_ids: List[str]) -> str:
        """Return the bullet content to pass as ``bullets_used`` to the reflector.

        The default implementation extracts only the specific bullets the generator
        cited by ID — appropriate for Q&A tasks (finance, mind2web) where the same
        LLM writes both the answer and the citation list in one JSON response.

        Override in subclasses for tasks where inline citation is impossible.
        For example, SWEBenchProGenerator overrides this to return the full playbook
        so the reflector has context for its analysis even though the SWE agent never
        cites bullet IDs inside a git-diff patch.
        """
        from playbook_utils import extract_playbook_bullets
        return extract_playbook_bullets(playbook, bullet_ids)

    def get_playbook_stats_for_curator(self, playbook: str) -> Optional[dict]:
        """Return playbook statistics to pass to the curator.

        The default implementation calls ``get_playbook_stats`` which includes
        helpful/harmful/unused breakdowns — useful for Q&A tasks (finance,
        mind2web) where bullet feedback is tracked per-citation.

        Override in subclasses where helpful/harmful counts are always zero
        (e.g. SWEBenchProGenerator) to avoid sending misleading statistics.
        Subclasses may return ``None`` to omit the stats block from the
        curator prompt entirely.
        """
        from playbook_utils import get_playbook_stats
        return get_playbook_stats(playbook)

    def get_ground_truth_for_reflector(self, target: str) -> str:
        """Return the ground-truth text to expose to the reflector.

        Default behavior is task-agnostic passthrough. Override in subclasses
        when ``target`` is a transport blob that contains extra evaluation
        metadata in addition to the human-meaningful answer.
        """
        return target

    def get_question_context_for_curator(self, question: str, context: str) -> str:
        """Return the question/context text to expose to the curator.

        Default behavior prefers explicit context when present and otherwise
        falls back to the question text.
        """
        return context or question

    def should_learn_from_initially_correct_samples(self) -> bool:
        """Whether correct-on-first-try samples should still trigger reflection/curation.

        The default matches the historical ACE behavior. Tasks with non-unique
        correct outputs (for example SWE-bench patches that are judged by tests
        rather than text equality) can override this to ``False`` so a correct
        first-pass answer is not "corrected" against a different ground-truth
        artifact and turned into noisy playbook updates.
        """
        return True

    def _extract_bullet_ids(self, response: str, use_json_mode: bool) -> List[str]:
        """
        Extract bullet IDs from generator response.
        
        Args:
            response: The generator's response
            use_json_mode: Whether JSON mode was used
            
        Returns:
            List of bullet IDs
        """
        bullet_ids = []
        
        if use_json_mode:
            try:
                response_json = json.loads(response)
                bullet_ids = response_json.get("bullet_ids", [])
            except (json.JSONDecodeError, KeyError):
                # If parsing fails, try regex extraction
                bullet_ids = self._extract_bullet_ids_regex(response)
        else:
            bullet_ids = self._extract_bullet_ids_regex(response)
        
        return bullet_ids
    
    def _extract_bullet_ids_regex(self, text: str) -> List[str]:
        """
        Extract bullet IDs using regex pattern matching.
        
        Args:
            text: Text to extract bullet IDs from
            
        Returns:
            List of bullet IDs
        """
        # Pattern matches: [xxx-00001], [abc-00042], etc.
        pattern = r'\[([a-z]{3,}-\d{5})\]'
        matches = re.findall(pattern, text)
        return matches