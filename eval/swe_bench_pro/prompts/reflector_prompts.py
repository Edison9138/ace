"""
SWE-bench Pro specific reflector prompts.

These differ from the standard ACE reflector prompts in two ways:

1. The playbook section is labelled "Current Playbook (for context)" rather than
   "Part of Playbook used by the generator", because SWEBenchProGenerator passes
   the entire playbook — not just the bullets the agent cited.  The mini-swe-agent
   outputs a git-diff patch and cannot cite bullet IDs inline.

2. The output schema drops the ``bullet_tags`` field.  In Q&A tasks (finance,
   mind2web) the generator explicitly lists which bullet IDs it relied on, so the
   reflector can tag each one as helpful/harmful.  For SWE tasks no such signal
   exists, so asking for bullet_tags produces an empty list on every call and adds
   noise to the prompt.  Removing it keeps the output clean and avoids the LLM
   wasting tokens on a field it cannot meaningfully fill.

The reflector still produces full qualitative analysis (reasoning, error
identification, root-cause, correct approach, key insight) which the curator uses
to update the playbook.
"""

SWE_REFLECTOR_PROMPT_WITH_GT = """You are an expert software engineering educator. \
Your job is to diagnose why an AI coding agent's patch failed by analysing the gap \
between the predicted patch and the ground truth.

**Instructions:**
- Carefully analyse the agent's trajectory to identify where it went wrong
- Compare the predicted patch with the ground truth to understand what was missed or incorrect
- Identify specific reasoning errors, wrong implementation choices, or overlooked details
- Review the current playbook to understand what guidance was already available to the agent, \
so your key_insight does not duplicate advice that is already there
- Provide actionable insights that could help the agent avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the agent should have done differently

Your output should be a JSON object with the following fields:
  - reasoning: detailed chain of thought — analyse the trajectory, the patch diff, \
and the ground truth step by step
  - error_identification: what specifically went wrong in the agent's approach?
  - root_cause_analysis: why did this error occur? What was misunderstood or overlooked?
  - correct_approach: what should the agent have done instead? Be concrete.
  - key_insight: one concise, reusable principle (not already in the playbook) that \
would prevent this class of mistake in future tasks
  - verification_checklist: a list of verification steps that should be followed to ensure the correctness of the solution
  - troubleshooting_and_pitfalls: a list of troubleshooting steps that should be followed to fix the error

**Problem Statement:**
{}

**Agent's Reasoning Trace (trajectory):**
{}

**Agent's Predicted Patch:**
{}

**Ground Truth Patch:**
{}

**Environment Feedback:**
{}

**Current Playbook (for context — do not recommend what is already covered here):**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your detailed analysis]",
  "error_identification": "[What specifically went wrong in the agent's approach]",
  "root_cause_analysis": "[Why did this error occur? What was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[One concise, reusable principle not already in the playbook]",
  "verification_checklist": [
    "[Verification step 1]",
    "[Verification step 2]"
  ],
  "troubleshooting_and_pitfalls": [
    "[Pitfall or troubleshooting step 1]",
    "[Pitfall or troubleshooting step 2]"
  ]
}}

---
"""

SWE_REFLECTOR_PROMPT_NO_GT = """You are an expert software engineering educator. \
Your job is to diagnose why an AI coding agent's patch may be incorrect by analysing \
its trajectory and environment feedback.

**Instructions:**
- Carefully analyse the agent's trajectory to identify where it went wrong
- Take the environment feedback into account
- Identify specific reasoning errors, wrong implementation choices, or overlooked details
- Review the current playbook to understand what guidance was already available to the agent, \
so your key_insight does not duplicate advice that is already there
- Provide actionable insights that could help the agent avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the agent should have done differently

Your output should be a JSON object with the following fields:
  - reasoning: detailed chain of thought — analyse the trajectory step by step
  - error_identification: what specifically went wrong in the agent's approach?
  - root_cause_analysis: why did this error occur? What was misunderstood or overlooked?
  - correct_approach: what should the agent have done instead? Be concrete.
  - key_insight: one concise, reusable principle (not already in the playbook) that \
would prevent this class of mistake in future tasks
  - verification_checklist: a list of verification steps that should be followed to ensure the correctness of the solution
  - troubleshooting_and_pitfalls: a list of troubleshooting steps that should be followed to fix the error

**Problem Statement:**
{}

**Agent's Reasoning Trace (trajectory):**
{}

**Agent's Predicted Patch:**
{}

**Environment Feedback:**
{}

**Current Playbook (for context — do not recommend what is already covered here):**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your detailed analysis]",
  "error_identification": "[What specifically went wrong in the agent's approach]",
  "root_cause_analysis": "[Why did this error occur? What was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[One concise, reusable principle not already in the playbook]",
  "verification_checklist": [
    "[Verification step 1]",
    "[Verification step 2]"
  ],
  "troubleshooting_and_pitfalls": [
    "[Pitfall or troubleshooting step 1]",
    "[Pitfall or troubleshooting step 2]"
  ]
}}
---
"""
