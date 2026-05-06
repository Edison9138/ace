import dataclasses
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

# mini-swe-agent is a pip package
from minisweagent import __version__ as _MINISWEAGENT_VERSION
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
try:
    from minisweagent.run.utils.save import save_traj as _save_traj
except ImportError:
    _save_traj = None
try:
    from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
except ImportError:
    SwerexModalEnvironment = None

from ace import Generator
from .eval_harness import get_dockerhub_image_uri
from .prompts.generator_prompts import GENERATOR_INSTANCE_PROMPT, GENERATOR_SYSTEM_PROMPT, SUBMIT_SENTINEL

logger = logging.getLogger(__name__)

# ── SWE task environment parameters — from swebench.yaml ──────────────────────
DOCKER_CWD = "/testbed"
# Per-command timeout inside agent environments. 60s is too tight for some
# repository-wide searches in SWE-bench Pro and can abort otherwise recoverable
# trajectories.
DOCKER_TIMEOUT = 300
# Keep these output-noise controls as additive image-level ENVs.
# Passing command-level env through swerex replaces inherited runtime env,
# which can drop PATH/HOME/Go-related variables.
ADDITIVE_IMAGE_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}
STARTUP_TIMEOUT = 1800
RUNTIME_TIMEOUT = 1800
DEPLOYMENT_TIMEOUT = 3600  # STARTUP_TIMEOUT + RUNTIME_TIMEOUT

# Modal sandbox configuration
SANDBOX_CPU = (0.25, 12.0)
SANDBOX_MEMORY = (128, 4096)
SANDBOX_IDLE_TIMEOUT = 600


def _build_additive_image_env_commands() -> list[str]:
    """Render additive ENV lines applied to the Modal clean image."""
    return [f"ENV {key}={value}" for key, value in ADDITIVE_IMAGE_ENV.items()]


def _build_modal_env_kwargs(*, clean_image_obj: Any, docker_image_uri: str) -> dict[str, Any]:
    """Build kwargs for SwerexModalEnvironment without command-level env override."""
    return {
        "clean_image_obj": clean_image_obj,
        "image": docker_image_uri,
        "cwd": DOCKER_CWD,
        "timeout": DOCKER_TIMEOUT,
        # Do not pass runtime `env`: swerex forwards that dict directly to
        # subprocess.run(..., env=...), replacing inherited image/container env.
        "startup_timeout": STARTUP_TIMEOUT,
        "runtime_timeout": RUNTIME_TIMEOUT,
        "deployment_timeout": DEPLOYMENT_TIMEOUT,
        "modal_sandbox_kwargs": {
            "cpu": SANDBOX_CPU,
            "memory": SANDBOX_MEMORY,
            # Auto-terminate the sandbox if the swerex tunnel TCP connection
            # becomes idle for this many seconds.
            "idle_timeout": SANDBOX_IDLE_TIMEOUT,
        },
    }


# Keep per-step observations tight; large tool outputs can quickly exceed
# model context windows over long trajectories.
MAX_OBSERVATION_CHARS = 8000
OBS_HEAD_CHARS = 1800
OBS_TAIL_CHARS = 1800
TRANSPORT_RETRY_ATTEMPTS = 3
TRANSPORT_RETRY_BACKOFF_BASE_S = 0.5

# Retry the entire sandbox lifecycle (creation + agent run) on infra failures.
# Covers cases where Modal fails to schedule the sandbox or the first HTTP
# request to a freshly created sandbox is refused/disconnected.
SANDBOX_LIFECYCLE_RETRIES = 2
SANDBOX_LIFECYCLE_RETRY_BACKOFF_S = 5

# Some SWE-bench Pro images only ship Python 3.9; swe-rex>=1.0 requires >=3.10.
# Prefer system python when available (>=3.10), and fall back to pyenv only
# on older images.
SWEREX_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/98/0d/"
    "d06ab2aa78138055c297490762cd7b4d8ac58a544783f874c869cdb7b534/"
    "swe_rex-1.4.0-py3-none-any.whl"
)
SWEREX_BOOTSTRAP_CMD = (
    "set -e && "
    "(apt update && apt install -y curl) || (apk update && apk add --no-cache curl bash) && "
    "if command -v python3 >/dev/null 2>&1 && "
    'python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; '
    "then "
    "PYBIN=$(command -v python3); "
    "else "
    "(apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && "
    "apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev "
    "libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev "
    "libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev) || "
    "(apk add --no-cache make build-base openssl-dev zlib-dev bzip2-dev readline-dev "
    "sqlite-dev git ncurses-dev xz tk-dev libxml2-dev xmlsec-dev libffi-dev xz-dev) && "
    "curl -fsSL https://pyenv.run | bash && "
    "~/.pyenv/bin/pyenv install -s 3.11.13 && "
    "PYBIN=~/.pyenv/versions/3.11.13/bin/python3.11; "
    "fi && "
    "$PYBIN -m pip --version >/dev/null 2>&1 || $PYBIN -m ensurepip --upgrade || true && "
    # Debian/Ubuntu Python 3.11 images can enforce PEP 668 ("externally managed").
    # Use PIP_BREAK_SYSTEM_PACKAGES for compatibility, and don't fail if pip
    # self-upgrade is blocked in base images.
    "PIP_BREAK_SYSTEM_PACKAGES=1 $PYBIN -m pip install --disable-pip-version-check --upgrade pip || true && "
    f"PIP_BREAK_SYSTEM_PACKAGES=1 $PYBIN -m pip install --disable-pip-version-check --target /opt/swerex '{SWEREX_WHEEL_URL}' && "
    "$PYBIN -c \"import sys; sys.path.insert(0, '/opt/swerex'); import swerex\" && "
    "printf '#!/bin/sh\\nexec env PYTHONPATH=/opt/swerex${PYTHONPATH:+:$PYTHONPATH} %s -m swerex \"$@\"\\n' "
    '"$PYBIN" > '
    "/usr/bin/swerex-remote && "
    "chmod +x /usr/bin/swerex-remote"
)

# ── Agent defaults ─────────────────────────────────────────────────────────────
# A lower default keeps multi-step debugging viable without letting trajectories
# sprawl in cost and prompt size.
DEFAULT_STEP_LIMIT = 80
DEFAULT_COST_LIMIT = 10.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_API_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.2"


def _get_class_name_with_module(obj: Any) -> str | None:
    if obj is None:
        return None
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def _asdict_or_raw(obj: Any) -> Any:
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj


def _save_traj_fallback(
    agent: DefaultAgent | None,
    path: Path,
    *,
    print_path: bool = True,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
    print_fct=print,
    **kwargs,
) -> None:
    """Persist a mini-swe-agent-compatible trajectory JSON without upstream helper support."""
    data = {
        "info": {
            "exit_status": exit_status,
            "submission": result,
            "model_stats": {
                "instance_cost": 0.0,
                "api_calls": 0,
            },
            "mini_version": _MINISWEAGENT_VERSION,
        },
        "messages": [],
        "trajectory_format": "mini-swe-agent-1",
    } | kwargs

    if agent is not None:
        model = getattr(agent, "model", None)
        env = getattr(agent, "env", None)
        data["info"]["model_stats"]["instance_cost"] = float(
            getattr(model, "cost", 0.0) or 0.0
        )
        data["info"]["model_stats"]["api_calls"] = int(
            getattr(model, "n_calls", 0) or 0
        )
        data["messages"] = list(getattr(agent, "messages", []) or [])
        data["info"]["config"] = {
            "agent": _asdict_or_raw(getattr(agent, "config", None)),
            "model": _asdict_or_raw(getattr(model, "config", None)),
            "environment": _asdict_or_raw(getattr(env, "config", None)),
            "agent_type": _get_class_name_with_module(agent),
            "model_type": _get_class_name_with_module(model),
            "environment_type": _get_class_name_with_module(env),
        }

    if extra_info:
        data["info"].update(extra_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    if print_path:
        print_fct(f"Saved trajectory to '{path}'")


def _save_agent_trajectory(
    agent: DefaultAgent | None,
    path: Path,
    *,
    exit_status: str,
    result: str,
) -> None:
    if _save_traj is not None:
        try:
            _save_traj(
                agent,
                path,
                print_path=False,
                exit_status=exit_status,
                result=result,
            )
            return
        except Exception as e:
            logger.warning(
                "Upstream save_traj failed for %s, using local fallback: %s", path, e
            )

    _save_traj_fallback(
        agent,
        path,
        print_path=False,
        exit_status=exit_status,
        result=result,
    )


def _truncate_observation_text(text: str) -> str:
    """Limit observation size to avoid context-window blowups."""
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    omitted = len(text) - (OBS_HEAD_CHARS + OBS_TAIL_CHARS)
    marker = f"\n\n...[output truncated: {omitted} chars omitted]...\n\n"
    return text[:OBS_HEAD_CHARS] + marker + text[-OBS_TAIL_CHARS:]


def _looks_like_transport_error_text(msg: str) -> bool:
    text = (msg or "").lower()
    markers = [
        "infra transport failure",
        "error making request",
        "all connection attempts failed",
        "failed to connect",
        "connection refused",
        "connection reset",
        "server disconnected",
        "remote protocol error",
        "network is unreachable",
        "temporarily unavailable",
        "503 service unavailable",
        "502 bad gateway",
        "504 gateway timeout",
        "http 500",
        "image build",
    ]
    return any(marker in text for marker in markers)


def _is_command_timeout_exception(exc: Exception, _depth: int = 0) -> bool:
    if _depth > 5:
        return False
    name = exc.__class__.__name__
    if name in {"CommandTimeoutError", "TimeoutExpired"}:
        return True
    msg = str(exc).lower()
    # "timed out while running command" — BashSession (pexpect) path in local.py
    # "exceeded while running command" — LocalRuntime.execute() subprocess path
    # Both come from swerex's CommandTimeoutError messages.
    if (
        "timed out while running command" in msg
        or "exceeded while running command" in msg
    ):
        return True
    # Walk the explicit and implicit exception cause chain.  Handles the fallback
    # case in swerex's _handle_transfer_exception where it raises SwerexException
    # wrapping the original timeout message when the exception class cannot be
    # reconstructed on the client (e.g. due to an ImportError).
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return _is_command_timeout_exception(cause, _depth + 1)
    return False


def _is_transport_retryable_exception(exc: Exception) -> bool:
    # Command execution timeouts are behavior-level outcomes and should not be
    # retried as infrastructure flakiness.
    if _is_command_timeout_exception(exc):
        return False

    name = exc.__class__.__name__
    if name in {
        "ClientConnectionError",
        "ClientConnectorError",
        "ServerDisconnectedError",
        "RemoteProtocolError",
        "ConnectError",
        "ReadTimeout",
        "TimeoutError",
        "ConnectionError",
    }:
        return True

    return _looks_like_transport_error_text(str(exc))


def _classify_generation_failure(exc: Exception) -> str:
    # Check command timeout before transport: with cause-chain detection in
    # _is_command_timeout_exception, a timeout wrapped inside a transport-like
    # exception is still correctly classified as agent behaviour.
    if _is_command_timeout_exception(exc):
        return "agent"
    if _is_transport_retryable_exception(exc):
        return "infra"
    if _looks_like_transport_error_text(str(exc)):
        return "infra"
    return "agent"


def _failure_type_from_exit_status(exit_status: str) -> str:
    status = (exit_status or "").lower()
    if status.startswith("infra_error"):
        return "infra"
    if status.startswith("agent_error"):
        return "agent"
    if status in {"submitted", "unknown"}:
        return "none"
    if status == "limitsexceeded":
        return "agent"
    if status.startswith("error:"):
        return "infra" if _looks_like_transport_error_text(status) else "agent"
    return "none"


# Same pattern ACE core uses in generator.py to match playbook IDs such as
# [vc-00001], [api-00042], [misc-00007], etc.
_BULLET_ID_RE = re.compile(r"\[([a-z]{2,}-\d{5})\]")

# Strips "helpful=X harmful=Y :: " metadata from playbook lines so the agent
# sees clean "[id] content" entries without misleading count annotations.
_BULLET_COUNTS_RE = re.compile(r"(\[[a-z]{2,}-\d{5}\])\s*helpful=\d+\s*harmful=\d+\s*::\s*")


def _strip_bullet_counts(playbook: str) -> str:
    """Remove helpful/harmful counts from playbook lines before showing to agent."""
    return _BULLET_COUNTS_RE.sub(r"\1 ", playbook)


def _load_transport_payload(raw: str) -> dict:
    """Best-effort parse of SWE transport JSON stored in ACE samples."""
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


# Per-message cap applied when formatting the trajectory string for the
# reflector / curator.  Matches ace-appworld's truncate_output limit.
_TRAJ_PER_MESSAGE_CHARS = 20_000
_REFLECTOR_TRAJECTORY_MAX_CHARS = 60_000
_CURATOR_TRAJECTORY_MAX_CHARS = 20_000
_REFLECTOR_PATCH_MAX_CHARS = 40_000
_CURATOR_CONTEXT_MAX_CHARS = 60_000
_PR_DESCRIPTION_BLOCK_RE = re.compile(
    r"(?s)<pr_description>\s*.*?\s*</pr_description>"
)
_REFLECTION_BLOCK_RE = re.compile(r"(?s)<reflection>\s*.*?\s*</reflection>")


def _truncate_middle_text(text: str, max_chars: int, label: str) -> str:
    """Limit prompt field size while preserving both the beginning and end."""
    if max_chars <= 0 or not isinstance(text, str) or len(text) <= max_chars:
        return text
    marker = (
        f"\n\n...[{label} truncated: "
        f"{len(text) - max_chars} chars omitted]...\n\n"
    )
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _extract_commands_from_actions(msg: dict[str, Any]) -> list[str]:
    """Return normalized assistant commands from mini-swe-agent extra.actions."""
    extra = msg.get("extra")
    if not isinstance(extra, dict):
        return []

    actions = extra.get("actions")
    if not isinstance(actions, list):
        return []

    commands: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        command = action.get("command")
        if isinstance(command, str) and command.strip():
            commands.append(command.strip())
    return commands


def _render_tool_call(tool_call: dict[str, Any]) -> str:
    """Format a tool call into readable trace text."""
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return str(tool_call)

    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        name = "tool"
    name = name.strip()

    raw_arguments = function.get("arguments")
    parsed_arguments = _load_transport_payload(raw_arguments)

    if name == "bash":
        command = parsed_arguments.get("command")
        if isinstance(command, str) and command.strip():
            return command.strip()
        if isinstance(raw_arguments, str) and raw_arguments.strip():
            return f"bash: {raw_arguments.strip()}"
        return "bash"

    rendered_args = ""
    if parsed_arguments:
        rendered_args = json.dumps(parsed_arguments, sort_keys=True)
    elif raw_arguments is not None:
        rendered_args = str(raw_arguments).strip()

    return f"{name}: {rendered_args}" if rendered_args else name


def _format_assistant_trace_content(msg: dict[str, Any]) -> str:
    """Render assistant text plus tool-invocation details for reflector traces."""
    parts: list[str] = []

    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            parts.append(content.strip())
    elif content not in (None, ""):
        parts.append(str(content))

    commands = _extract_commands_from_actions(msg)
    if commands:
        parts.extend(commands)
    else:
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    parts.append(str(tool_call))
                    continue
                parts.append(_render_tool_call(tool_call))

    return "\n\n".join(part for part in parts if part)


def _format_initial_user_trace_content_for_curator(content: Any) -> str:
    """Redact duplicated task/reflection blocks while preserving instructions."""
    if isinstance(content, str):
        text = content
    elif content in (None, ""):
        return ""
    else:
        text = str(content)

    text = _PR_DESCRIPTION_BLOCK_RE.sub(
        "<pr_description omitted; see Question Context>", text
    )
    text = _REFLECTION_BLOCK_RE.sub(
        "<reflection omitted; see Recent Reflection>", text
    )
    return text.strip()


def _extract_bullet_ids_from_trajectory(
    agent: "DefaultAgent | None",
) -> list[str]:
    """Extract playbook bullet IDs referenced in the agent's trajectory.

    Scans assistant messages for explicit ``[xxx-00001]`` references.
    mini-swe-agent does not produce structured JSON with bullet_ids, so
    this will typically return an empty list — which is correct: no tags
    means no helpful/harmful counts are updated, keeping the playbook clean.
    """
    bullet_ids: list[str] = []

    if agent is not None and hasattr(agent, "messages"):
        for msg in agent.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content") or ""
                bullet_ids.extend(_BULLET_ID_RE.findall(content))

    # Deduplicate while preserving first-seen order.
    return list(dict.fromkeys(bullet_ids))


def _format_trajectory(
    agent: "DefaultAgent | None",
    exit_status: str,
    instance_id: str,
    *,
    redact_initial_user_for_curator: bool = False,
    max_total_chars: int | None = None,
) -> str:
    """Build a rich reasoning trace from the agent's conversation history.

    Formats the stored agent messages into a single text block. The curator can
    optionally receive a redacted version of the initial user message so
    duplicated task/reflection context is omitted while preserving the agent
    instructions.

    The system message body (playbook + instructions) is omitted — the
    reflector already receives the playbook via ``bullets_used``.
    """
    header = f"mini-swe-agent [{exit_status}] on {instance_id}"

    if agent is None or not hasattr(agent, "messages") or not agent.messages:
        return header

    formatted: list[str] = []
    for i, msg in enumerate(agent.messages):
        role = msg.get("role", "unknown").upper()

        # Skip system message body — it's the playbook + instructions that
        # the reflector already knows about.
        if role == "SYSTEM":
            formatted.append(f"[{i}] SYSTEM: [playbook and agent instructions]")
            continue

        if role == "ASSISTANT":
            content = _format_assistant_trace_content(msg)
        elif role == "USER" and redact_initial_user_for_curator and i == 1:
            content = _format_initial_user_trace_content_for_curator(
                msg.get("content")
            )
        else:
            content = msg.get("content") or ""

        # Cap individual messages (like ace-appworld's truncate_output).
        if len(content) > _TRAJ_PER_MESSAGE_CHARS:
            half = _TRAJ_PER_MESSAGE_CHARS // 2
            content = (
                content[:half]
                + f"\n\n...[truncated {len(content) - _TRAJ_PER_MESSAGE_CHARS} chars]...\n\n"
                + content[-half:]
            )

        formatted.append(f"[{i}] {role}: {content}")

    trajectory = "\n\n".join(formatted)

    formatted_trace = header + "\n\n=== FULL AGENT TRAJECTORY ===\n\n" + trajectory
    if max_total_chars is not None:
        formatted_trace = _truncate_middle_text(
            formatted_trace,
            max_total_chars,
            "agent trajectory",
        )
    return formatted_trace


def _format_trajectory_for_reflector(
    agent: "DefaultAgent | None",
    exit_status: str,
    instance_id: str,
) -> str:
    """Build the full reasoning trace shown to the reflector."""
    return _format_trajectory(
        agent,
        exit_status,
        instance_id,
        max_total_chars=_REFLECTOR_TRAJECTORY_MAX_CHARS,
    )


def _format_trajectory_for_curator(
    agent: "DefaultAgent | None",
    exit_status: str,
    instance_id: str,
) -> str:
    """Build the curator trace with duplicated setup blocks redacted."""
    return _format_trajectory(
        agent,
        exit_status,
        instance_id,
        redact_initial_user_for_curator=True,
        max_total_chars=_CURATOR_TRAJECTORY_MAX_CHARS,
    )


def _coerce_agent_run_result(result: Any) -> tuple[str, str]:
    """Normalize DefaultAgent.run() outputs across mini-swe-agent versions."""
    if isinstance(result, dict):
        return result.get("submission", "") or "", result.get("exit_status", "unknown")

    if isinstance(result, tuple):
        if len(result) >= 3:
            exit_status, _result_text, patch = result[:3]
            return patch or "", str(exit_status or "unknown")
        if len(result) == 2:
            exit_status, submission = result
            return submission or "", str(exit_status or "unknown")

    raise TypeError(
        f"Unsupported DefaultAgent.run() result type: {type(result).__name__}"
    )


def _normalize_submission_result(
    patch: str, exit_status: str, instance_id: Optional[str] = None
) -> tuple[str, str]:
    """Reclassify empty successful submissions as explicit agent failures."""
    if exit_status == "Submitted" and not patch.strip():
        if instance_id:
            logger.warning(
                "Empty submission for %s: command emitted %s without patch payload",
                instance_id,
                SUBMIT_SENTINEL,
            )
        else:
            logger.warning(
                "Empty submission: command emitted %s without patch payload",
                SUBMIT_SENTINEL,
            )
        return "", "agent_error:empty_submission"
    return patch, exit_status


class SWEBenchProGenerator(Generator):
    """
    Replaces ACE's single-LLM-call Generator with mini-swe-agent's multi-step bash loop.
    Each generate() call: spins up a Modal sandbox → runs DefaultAgent → returns
    the final patch.
    """

    def __init__(
        self,
        api_client,
        api_provider: str,
        model: str,
        coding_model_name: str,
        dockerhub_username: str,
        step_limit: int = DEFAULT_STEP_LIMIT,
        cost_limit: float = DEFAULT_COST_LIMIT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        traj_output_dir: str = None,
    ):
        """
        Args:
            api_client:        OpenAI client (from ace_system.generator_client)
            api_provider:      API provider string (e.g. "openai")
            model:             Bookkeeping label for the generator slot in ACE logs
            coding_model_name: litellm model string for the coding agent (e.g. "gpt-5-mini")
            dockerhub_username: Docker Hub username for the coding agent
            step_limit:        max bash steps before agent is stopped (0 = unlimited)
            cost_limit:        max USD cost per instance
            max_tokens:        max completion tokens for the SWE coding agent
            traj_output_dir:   optional dir to save .traj.json trajectory files
        """

        super().__init__(api_client, api_provider, coding_model_name, max_tokens)
        self.ace_generator_label = model
        self.coding_model_name = coding_model_name
        self.dockerhub_username = dockerhub_username
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.traj_output_dir = Path(traj_output_dir) if traj_output_dir else None

    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None,
    ) -> tuple:

        ctx = _load_transport_payload(context)
        instance_id = ctx.get("instance_id", "unknown")
        repo = ctx.get("repo", "")
        problem_statement = ctx.get("problem_statement", question)

        # Resolve the image using the same logic as eval_harness
        docker_image_uri = get_dockerhub_image_uri(
            instance_id, self.dockerhub_username, repo
        )

        if not docker_image_uri:
            # missing or malformed docker_image_uri
            return (
                self._empty("no docker_image_uri in context"),
                [],
                {
                    "exit_status": "infra_error:missing_docker_image_uri",
                    "instance_id": instance_id,
                    "failure_type": "infra",
                },
            )

        traj_path = None
        if self.traj_output_dir:
            traj_path = self.traj_output_dir / f"{instance_id}_{call_id}.traj.json"

        # Outer try/finally guarantees the sandbox is always stopped.
        # Inner try/except catches agent-level failures without suppressing cleanup.
        env = None
        agent = None
        patch, exit_status = "", "error: not started"

        try:
            try:
                if SwerexModalEnvironment is None:
                    raise RuntimeError(
                        "minisweagent.environments.extra.swerex_modal is unavailable; "
                        "install mini-swe-agent with Modal support."
                    )

                logger.info(
                    f"Using SwerexModalEnvironment (Remote Sandbox) for {docker_image_uri}"
                )

                import modal

                # Fix: The official SWE-bench OS DockerHub images hardcode `ENTRYPOINT ["/bin/bash"]`.
                # When Modal Sandbox orchestrates using `["/usr/bin/env", "bash", "-c", "..."]`,
                # it natively prepends the ENTRYPOINT causing execution of `/bin/bash /usr/bin/env`
                # which natively crashes with `cannot execute binary file`.
                # We dynamically recompile the image to replace ENTRYPOINT before starting Swerex.
                clean_image = (
                    modal.Image.from_registry(docker_image_uri)
                    .dockerfile_commands(
                        ["ENTRYPOINT []", *_build_additive_image_env_commands()]
                    )
                    .run_commands(
                        [
                            "ln -s /app /testbed || true",
                            SWEREX_BOOTSTRAP_CMD,
                        ]
                    )
                )

                # Subclass SwerexModalEnvironment locally to bypass SwerexModalEnvironmentConfig Pydantic validation
                # which natively requires `image` to be a string. By passing our parsed `modal.Image` directly
                # into `ModalDeployment`, we cleanly enforce the ENTRYPOINT override natively.
                class _CleanSwerexModalEnvironment(SwerexModalEnvironment):
                    def __init__(self, clean_image_obj, **kwargs):
                        from minisweagent.environments.extra.swerex_modal import (
                            SwerexModalEnvironmentConfig,
                        )
                        from swerex.deployment.modal import ModalDeployment
                        import asyncio

                        class _SafeModalDeployment(ModalDeployment):
                            # The base AbstractDeployment.__del__ can emit
                            # "coroutine ... was never awaited" warnings on
                            # interpreter shutdown. We stop explicitly in
                            # generator cleanup, so avoid async work in __del__.
                            def __del__(self):
                                return

                        self.config = SwerexModalEnvironmentConfig(**kwargs)
                        self.deployment = _SafeModalDeployment(
                            image=clean_image_obj,
                            startup_timeout=self.config.startup_timeout,
                            runtime_timeout=self.config.runtime_timeout,
                            deployment_timeout=self.config.deployment_timeout,
                            # We bootstrap /usr/bin/swerex-remote in the custom image,
                            # so we do not rely on pipx inside the task image.
                            install_pipx=False,
                            modal_sandbox_kwargs=self.config.modal_sandbox_kwargs,
                        )
                        asyncio.run(self.deployment.start())

                    def execute(self, action, cwd="", *, timeout=None):
                        # The installed SwerexModalEnvironment.execute() returns only
                        # {"output": ..., "returncode": ...}, missing "exception_info".
                        # The LitellmModel observation template uses StrictUndefined Jinja2,
                        # which raises UndefinedError on `output.exception_info` when the key
                        # is absent. All other mini-swe-agent environments include this key.
                        #
                        # effective_timeout mirrors what SwerexModalEnvironment.execute()
                        # passes to RexCommand (timeout or self.config.timeout), so the
                        # elapsed-time heuristic uses the same budget as the server.
                        effective_timeout = (
                            timeout if timeout is not None else DOCKER_TIMEOUT
                        )
                        start_time = time.monotonic()
                        attempt = 0
                        while True:
                            try:
                                output = super().execute(action, cwd=cwd, timeout=timeout)
                                break
                            except Exception as e:
                                elapsed = time.monotonic() - start_time

                                # ── Direct command timeout (normal path) ─────────────
                                # Server returned HTTP 511; client reconstructed
                                # CommandTimeoutError.  Also catches the SwerexException
                                # fallback (when exception class cannot be imported) via
                                # the message-pattern check in _is_command_timeout_exception.
                                if _is_command_timeout_exception(e):
                                    msg = str(e) or e.__class__.__name__
                                    return {
                                        "output": msg,
                                        "returncode": 124,
                                        "exception_info": msg,
                                    }

                                # ── Indirect command timeout (server-side crash) ──────
                                # swerex's BaseHTTPMiddleware can interact with Python
                                # 3.12's ExceptionGroup unwinding in a way that bypasses
                                # @app.exception_handler(Exception), so the server emits
                                # HTTP 500 (plain-text body) instead of HTTP 511.
                                # The client then sees:
                                #   • ContentTypeError  — response.json() fails on a
                                #     non-JSON "Internal Server Error" body, or
                                #   • ServerDisconnectedError — the TCP connection drops.
                                # Neither carries a CommandTimeoutError class name, so
                                # without this heuristic ServerDisconnectedError would be
                                # retried as an infra flake (potentially 3 × 300 s = 15 min
                                # wasted) and then misclassified as "infra".
                                #
                                # Guard: only apply to HTTP/transport error types so that
                                # application-level exceptions (e.g. Submitted) are never
                                # swallowed here and always propagate via the bare `raise`.
                                _is_http_or_transport = (
                                    _is_transport_retryable_exception(e)
                                    or e.__class__.__name__
                                    in {"ContentTypeError", "ClientResponseError"}
                                )
                                if (
                                    _is_http_or_transport
                                    and elapsed >= effective_timeout * 0.9
                                ):
                                    msg = (
                                        f"Command timeout (inferred: elapsed"
                                        f" {elapsed:.0f}s >="
                                        f" {effective_timeout * 0.9:.0f}s threshold):"
                                        f" {type(e).__name__}: {e}"
                                    )
                                    logger.warning(
                                        "Suspected server-side command timeout"
                                        " (elapsed %.0fs, timeout %ss): %s: %s",
                                        elapsed,
                                        effective_timeout,
                                        type(e).__name__,
                                        e,
                                    )
                                    return {
                                        "output": msg,
                                        "returncode": 124,
                                        "exception_info": msg,
                                    }

                                # ── Infra transport failures (early, before timeout) ──
                                if (
                                    _is_transport_retryable_exception(e)
                                    and attempt < TRANSPORT_RETRY_ATTEMPTS - 1
                                ):
                                    delay_s = TRANSPORT_RETRY_BACKOFF_BASE_S * (
                                        2**attempt
                                    )
                                    logger.warning(
                                        "Transient runtime transport failure "
                                        "(attempt %s/%s), retrying in %.1fs: %s",
                                        attempt + 1,
                                        TRANSPORT_RETRY_ATTEMPTS,
                                        delay_s,
                                        e,
                                    )
                                    time.sleep(delay_s)
                                    attempt += 1
                                    continue

                                if _is_transport_retryable_exception(e):
                                    msg = (
                                        f"infra transport failure after retries: "
                                        f"{type(e).__name__}: {e}"
                                    )
                                    raise RuntimeError(msg) from e

                                raise
                        output.setdefault("exception_info", "")
                        raw_output = output.get("output", "")
                        if (
                            isinstance(raw_output, str)
                            and SUBMIT_SENTINEL not in raw_output
                        ):
                            output["output"] = _truncate_observation_text(raw_output)
                        return output

                    def stop(self):
                        """Override stop() to add a synchronous fallback termination path.

                        ModalDeployment.stop() is async and calls sandbox.terminate()
                        inside a coroutine.  If runtime.close() hangs or raises before
                        reaching the terminate() call, the sandbox can be left alive.

                        This override:
                        1. Runs dep.stop() with a 10 s asyncio timeout so a hanging
                           runtime.close() doesn't block cleanup indefinitely.
                        2. Falls back to dep._sandbox.terminate() synchronously if the
                           async path failed or timed out and the sandbox is still set.

                        Note: Modal-native `idle_timeout` (set in modal_sandbox_kwargs)
                        handles the crash/SIGKILL case where no cleanup code runs at all.
                        """
                        import asyncio
                        import logging

                        dep = self.deployment

                        async def _graceful_stop():
                            try:
                                await asyncio.wait_for(dep.stop(), timeout=10)
                            except Exception as e:
                                logging.getLogger(__name__).debug(
                                    f"Graceful dep.stop() failed or timed out: {e}"
                                )

                        try:
                            asyncio.run(_graceful_stop())
                        except Exception as e:
                            logging.getLogger(__name__).debug(
                                f"asyncio.run(_graceful_stop) failed: {e}"
                            )

                        # Fallback: forcefully terminate the Modal sandbox synchronously
                        if hasattr(dep, "_sandbox") and dep._sandbox is not None:
                            try:
                                dep._sandbox.terminate()
                            except Exception as e:
                                logging.getLogger(__name__).warning(
                                    f"Failed to forcefully terminate Modal sandbox: {e}"
                                )

                # Retry the full sandbox lifecycle on infra failures.
                # The ServerDisconnectedError can happen during agent.run()
                # (not just env creation), so we must retry both together
                # with a fresh sandbox each time.
                for sandbox_attempt in range(SANDBOX_LIFECYCLE_RETRIES):
                    env = None
                    try:
                        env = _CleanSwerexModalEnvironment(
                            **_build_modal_env_kwargs(
                                clean_image_obj=clean_image,
                                docker_image_uri=docker_image_uri,
                            )
                        )

                        agent = DefaultAgent(
                            LitellmModel(
                                model_name=self.coding_model_name,
                                # Match the reference swebp config: deterministic
                                # temperature and an explicit completion budget.
                                model_kwargs={
                                    "drop_params": True,
                                    "temperature": 0.0,
                                    "max_tokens": self.max_tokens,
                                    "max_completion_tokens": self.max_tokens,
                                },
                            ),
                            env,
                            system_template=GENERATOR_SYSTEM_PROMPT,
                            instance_template=GENERATOR_INSTANCE_PROMPT,
                            step_limit=self.step_limit,
                            cost_limit=self.cost_limit,
                        )
                        result = agent.run(
                            problem_statement,
                            playbook_content=_strip_bullet_counts(playbook)
                            if playbook
                            else "(empty)",
                            reflection_content=reflection or "(empty)",
                        )
                        patch, exit_status = _coerce_agent_run_result(result)
                        patch, exit_status = _normalize_submission_result(
                            patch, exit_status, instance_id
                        )
                        break
                    except Exception as e:
                        # Clean up the failed sandbox before retry or re-raise.
                        if env is not None:
                            try:
                                if hasattr(env, "stop"):
                                    env.stop()
                            except Exception:
                                pass
                            env = None

                        is_last_attempt = (
                            sandbox_attempt >= SANDBOX_LIFECYCLE_RETRIES - 1
                        )
                        if (
                            not is_last_attempt
                            and _classify_generation_failure(e) == "infra"
                        ):
                            logger.warning(
                                "Sandbox infra failure (attempt %s/%s) for %s, "
                                "retrying in %ss: %s: %s",
                                sandbox_attempt + 1,
                                SANDBOX_LIFECYCLE_RETRIES,
                                instance_id,
                                SANDBOX_LIFECYCLE_RETRY_BACKOFF_S
                                * (2**sandbox_attempt),
                                type(e).__name__,
                                e,
                            )
                            time.sleep(
                                SANDBOX_LIFECYCLE_RETRY_BACKOFF_S
                                * (2**sandbox_attempt)
                            )
                            continue
                        raise
            except Exception as e:
                logger.error(f"mini-swe-agent failed for {instance_id}: {e}")
                failure_type = _classify_generation_failure(e)
                patch = ""
                exit_status = f"{failure_type}_error:{type(e).__name__}:{e}"
        finally:
            if env is not None:
                if hasattr(env, "stop"):
                    env.stop()

        if traj_path:
            try:
                _save_agent_trajectory(
                    agent,
                    traj_path,
                    exit_status=exit_status,
                    result=patch,
                )
            except Exception as e:
                logger.warning("Failed to save trajectory for %s: %s", instance_id, e)

        # Extract bullet IDs the agent referenced in its trajectory.
        # Scan assistant messages for explicit [xxx-00001] references.
        bullet_ids = _extract_bullet_ids_from_trajectory(agent)
        reasoning = _format_trajectory_for_reflector(agent, exit_status, instance_id)
        curator_reasoning = _format_trajectory_for_curator(
            agent,
            exit_status,
            instance_id,
        )

        response = json.dumps(
            {
                "reasoning": reasoning,
                "curator_reasoning": curator_reasoning,
                "bullet_ids": bullet_ids,
                "final_answer": patch,
            }
        )
        return (
            response,
            bullet_ids,
            {
                "exit_status": exit_status,
                "instance_id": instance_id,
                "failure_type": _failure_type_from_exit_status(exit_status),
            },
        )

    def get_bullets_for_reflector(self, playbook: str, bullet_ids: list) -> str:  # noqa: ARG002
        """Return the full playbook as context for the reflector.

        mini-swe-agent outputs a git-diff patch and cannot cite bullet IDs inline,
        so the base-class default (extract_playbook_bullets) always returns the
        uninformative placeholder "(No bullets used by generator)".

        Instead we pass the entire playbook so the reflector can see what guidance
        was already available to the agent.  This lets it write analysis that
        references existing bullets by name, which in turn helps the curator avoid
        adding redundant bullets.
        """
        if not playbook or not playbook.strip() or playbook.strip() == "(empty)":
            return "(No playbook available)"
        return _strip_bullet_counts(playbook)

    def get_playbook_stats_for_curator(self, playbook: str) -> None:  # noqa: ARG002
        """Omit playbook stats from the curator prompt for SWE-bench Pro.

        AppWorld-style coding adaptation does not rely on helpful/harmful
        counters or playbook-size summaries in the curator prompt.  SWE has the
        same property: the coding agent never cites bullet IDs inline, so the
        counter layer is not meaningful. Passing size stats can also anchor the
        curator on noisy or misleading summaries instead of the full playbook
        text.  We therefore expose only the stripped playbook itself plus the
        existing token-budget line in the prompt.
        """
        return None

    def get_playbook_for_curator(self, playbook: str) -> str:
        """Return the playbook text to expose to the curator.

        SWE keeps helpful/harmful counts in storage for compatibility with the
        root ACE utilities, but those counters are never meaningfully updated in
        this task. Hiding them from the curator avoids suggesting that all-zero
        counts carry signal about bullet quality.
        """
        if not playbook or not playbook.strip() or playbook.strip() == "(empty)":
            return "(empty)"
        return _strip_bullet_counts(playbook)

    def get_predicted_answer_for_reflector(self, predicted_answer: str) -> str:
        """Return a bounded patch copy for reflector prompts only."""
        if not predicted_answer:
            return predicted_answer
        return _truncate_middle_text(
            predicted_answer,
            _REFLECTOR_PATCH_MAX_CHARS,
            "predicted patch",
        )

    def get_ground_truth_for_reflector(self, target: str) -> str:
        """Return only the gold patch, not the full SWE eval metadata blob."""
        meta = _load_transport_payload(target)
        gold_patch = meta.get("gold_patch", "") or "(ground truth patch not available)"
        return _truncate_middle_text(
            gold_patch,
            _REFLECTOR_PATCH_MAX_CHARS,
            "ground truth patch",
        )

    def get_question_context_for_curator(self, question: str, context: str) -> str:
        """Return a readable task summary instead of raw escaped JSON."""
        ctx = _load_transport_payload(context)
        problem_statement = question or ctx.get("problem_statement") or "(not available)"

        lines = ["Problem Statement:", problem_statement]

        metadata = []
        if ctx.get("repo"):
            metadata.append(f"- Repository: {ctx['repo']}")
        if ctx.get("instance_id"):
            metadata.append(f"- Instance ID: {ctx['instance_id']}")

        if metadata:
            lines.extend(["", "Task Metadata:"] + metadata)

        return _truncate_middle_text(
            "\n".join(lines),
            _CURATOR_CONTEXT_MAX_CHARS,
            "question context",
        )

    def should_learn_from_initially_correct_samples(self) -> bool:
        """SWE correctness is test-based and patches are non-unique.

        A first-pass correct patch should not be "improved" by comparing it
        against the benchmark gold patch, because a different-but-valid patch
        can still pass all tests. Skipping reflection/curation in this case
        avoids teaching the playbook that correct alternatives are mistakes.
        """
        return False

    def _empty(self, reason: str) -> str:
        return json.dumps({"reasoning": reason, "bullet_ids": [], "final_answer": ""})
