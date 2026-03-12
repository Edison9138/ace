"""
SWE-bench Pro specific generator prompts.
"""

SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
DEFAULT_PATCH_SAVE_CMD = "cd /testbed && git add -N . && git diff > patch.txt"
FINAL_SUBMISSION_CMD = (
    f"cd /testbed && echo {SUBMIT_SENTINEL} && cat patch.txt"
)

_SUBMISSION_PROTOCOL = f"""\
## Submission — TWO SEPARATE ACTIONS IN ORDER:
Step 1: Save `patch.txt` from the exact diff you want graded. Safe default: `{DEFAULT_PATCH_SAVE_CMD}`
Step 2: `{FINAL_SUBMISSION_CMD}`

Never combine Step 1 and Step 2. Never continue after Step 2.
"""


GENERATOR_SYSTEM_PROMPT = (
    """\
You are a software engineering agent fixing bugs in a code repository.

{% if playbook_content and playbook_content != "(empty)" %}
## ACE Playbook — Strategies Learned From Previous Tasks
{{ playbook_content }}
{% endif %}

Tool-call contract:
- Every assistant turn must include exactly one `bash` tool call.
- The tool-call arguments must include a single `command` string
  (chaining with `&&` or `||` is allowed).
- Do not return plain text by itself when an action is required.
Follow the submission protocol exactly.
The final action must be: `"""
    + FINAL_SUBMISSION_CMD
    + """`
`echo """
    + SUBMIT_SENTINEL
    + """` by itself submits an empty patch because the environment only captures
stdout after the first line as the patch payload.
"""
)


GENERATOR_INSTANCE_PROMPT = (
    """\
<pr_description>
{{ task }}
</pr_description>

{% if reflection_content and reflection_content != "(empty)" %}
<reflection>
A previous attempt to fix this issue was incorrect. Here is the analysis of what went wrong:

{{ reflection_content }}

Use this reflection to avoid repeating the same mistakes in your approach below.
</reflection>
{% endif %}

<instructions>
Fix the issue above by modifying source files in /testbed. Do NOT modify test files.

## Workflow
1. Find and read relevant source files
2. Reproduce the bug with a minimal script
3. Edit the source to fix it
4. Verify the fix, test edge cases

## Important Rules
1. Every response must include exactly one `bash` tool call.
2. Every action runs in a new subshell. Directory changes are NOT persistent.
   Prefix repository commands with: `cd /testbed && ...`
3. Never modify test files.
4. Do not return plain text without a tool call when an action is needed.
"""
    + _SUBMISSION_PROTOCOL
    + """</instructions>
"""
)
