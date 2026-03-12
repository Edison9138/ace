"""
SWE-bench Pro specific curator prompts.

These differ from the standard ACE curator prompts in two ways:

1. They expose the generator's agent trajectory as extra curator context.
   For SWE, the generator is a coding agent whose trajectory carries useful
   process-level information that can complement the reflector summary.

2. They ask each ADD operation to include a short ``reason``. This is useful
   audit metadata for SWE playbook curation, but is not required by the
   shared ACE curator schema for other tasks.
"""

SWE_CURATOR_PROMPT_WITH_GT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions. 
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth. 

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- Use ONLY one of these exact section names for ADD operations:
  - strategies_and_hard_rules
  - useful_code_snippets_and_templates
  - common_mistakes_and_correct_strategies
  - problem_solving_heuristics_and_workflows
  - others
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- Every ADD.content value must be a single-line plain-text sentence. Do not include newlines, bullet lists, markdown, or code fences in content.


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Generator's Reasoning Trace (agent trajectory):**
{reasoning_trace}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. It must be a single-line plain-text sentence with no newlines, bullet lists, markdown, or code fences. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.
    - reason: a short explanation of why this bullet is being added based on the reflection

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "ADD", 
      "section": "common_mistakes_and_correct_strategies",
      "content": "[If a repo uses generated files or startup-time validation, verify the exact version-parsing helper before changing higher-level control flow.]",
      "reason": "[Explanation of why this is needed...]"
    }}
  ]
}}

---
"""


SWE_CURATOR_PROMPT_NO_GT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions. 
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- Use ONLY one of these exact section names for ADD operations:
  - strategies_and_hard_rules
  - useful_code_snippets_and_templates
  - common_mistakes_and_correct_strategies
  - problem_solving_heuristics_and_workflows
  - others
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- Every ADD.content value must be a single-line plain-text sentence. Do not include newlines, bullet lists, markdown, or code fences in content.


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Generator's Reasoning Trace (agent trajectory):**
{reasoning_trace}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. It must be a single-line plain-text sentence with no newlines, bullet lists, markdown, or code fences. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.
    - reason: a short explanation of why this bullet is being added based on the reflection

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "ADD", 
      "section": "common_mistakes_and_correct_strategies",
      "content": "[If a repo uses generated files or startup-time validation, verify the exact version-parsing helper before changing higher-level control flow.]",
      "reason": "[Explanation of why this is needed...]"
    }}
  ]
}}

---
"""
