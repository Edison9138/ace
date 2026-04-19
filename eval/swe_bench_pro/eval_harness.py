"""
Evaluation harness for SWE-bench Pro.
Copied from SWE-bench_Pro-os/swe_bench_pro_eval.py and SWE-bench_Pro-os/helper_code/image_uri.py.
"""

import json
import os
import re
import urllib.request

try:
    import modal
except ImportError:
    modal = None

_ACE_META_KEY = "_ace_meta"

SANDBOX_CPU = (0.25, 12.0)
SANDBOX_MEMORY = (128, 4096)
SANDBOX_IDLE_TIMEOUT = 600
SANDBOX_TIMEOUT = 60 * 60

def _is_modal_sandbox_notfound(exc: Exception) -> bool:
    """Best-effort check for Modal sandbox termination/not-found errors."""
    if exc.__class__.__name__ != "NotFoundError":
        return False
    msg = str(exc)
    return "Sandbox" in msg or "container ID" in msg


# ── Copied from SWE-bench_Pro-os/helper_code/image_uri.py ──────────────────────────────────────


def get_dockerhub_image_uri(uid, dockerhub_username, repo_name=""):
    # repo_name must be in "org/repo" format (e.g. "django/django").
    # Return None for missing or malformed values so callers can fail gracefully.
    if not repo_name or "/" not in repo_name:
        return None
    repo_base, repo_name_only = repo_name.lower().split("/", 1)
    hsh = uid.replace("instance_", "")

    if (
        uid
        == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan"
    ):
        repo_name_only = "element-web"  # Keep full name for this one case
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    # All other repos: strip -vnan suffix
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]

    return f"{dockerhub_username}/sweap-images:{tag}"


# ── Copied from swe_bench_pro_eval.py ─────────────────────────────────────────


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)

    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)

    return "".join(kept)


def load_github_script(instance_id, script_name):
    """Load a script file from the official SWE-bench_Pro-os GitHub repo."""
    url = f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts/{instance_id}/{script_name}"
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to fetch {script_name} from GitHub for {instance_id}: {e}"
        )


def load_local_script(scripts_dir, instance_id, script_name):
    """Load a script file from local scripts directory, falling back to GitHub."""
    if scripts_dir:
        script_path = os.path.join(scripts_dir, instance_id, script_name)
        if os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                return f.read()
    return load_github_script(instance_id, script_name)


def load_github_base_docker(iid):
    url = f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/dockerfiles/base_dockerfile/{iid}/Dockerfile"
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to fetch base Dockerfile from GitHub for {iid}: {e}"
        )


def load_base_docker(iid, dockerfiles_dir):
    if dockerfiles_dir:
        path = os.path.join(dockerfiles_dir, "base_dockerfile", iid, "Dockerfile")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fp:
                return fp.read()
    return load_github_base_docker(iid)


def load_github_instance_docker(iid):
    url = f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/dockerfiles/instance_dockerfile/{iid}/Dockerfile"
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to fetch instance Dockerfile from GitHub for {iid}: {e}"
        )


def instance_docker(iid, dockerfiles_dir):
    if dockerfiles_dir:
        path = os.path.join(dockerfiles_dir, "instance_dockerfile", iid, "Dockerfile")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fp:
                return fp.read()
    return load_github_instance_docker(iid)


def create_entryscript(sample, dockerfiles_dir=None):
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    # selected_test_files_to_run is stored as a JSON-encoded list string in the
    # raw JSONL; use json.loads() instead of eval() to avoid arbitrary code
    # execution.  Guard against pandas silently pre-parsing it into a native list.
    _stf_raw = sample["selected_test_files_to_run"]
    _stf_list = _stf_raw if isinstance(_stf_raw, list) else json.loads(_stf_raw)
    selected_test_files_to_run = ",".join(_stf_list)
    base_commit = sample["base_commit"]
    base_dockerfile = load_base_docker(sample["instance_id"], dockerfiles_dir)
    instance_dockerfile = instance_docker(sample["instance_id"], dockerfiles_dir)

    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                # Convert ENV commands to export statements
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)

    env_cmds = "\n".join(env_cmds)

    entry_script = f"""
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def prepare_run(uid, output_dir, prefix, redo):
    uid_dir = os.path.join(output_dir, uid)
    os.makedirs(uid_dir, exist_ok=True)
    output_path = os.path.join(uid_dir, f"{prefix}_output.json")
    if not redo and os.path.exists(output_path):
        print(f"Skipping {uid} - output already exists")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_output_snapshot(output_dir, uid, prefix, output):
    with open(
        os.path.join(output_dir, uid, f"{prefix}_output.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output, f)


def build_failure_output(
    *,
    output_dir,
    uid,
    prefix,
    failure_type,
    status,
    error="",
    persist=True,
):
    output = {
        "tests": [],
        _ACE_META_KEY: {
            "failure_type": failure_type,
            "status": status,
            "error": error,
        },
    }
    if persist:
        _write_output_snapshot(output_dir, uid, prefix, output)
    return output


def write_files_modal(sandbox, files):
    for rel_path, content in files.items():
        with sandbox.open(f"/workspace/{rel_path}", "w") as f:
            f.write(content)


def collect_outputs_modal(sandbox, output_dir, uid, prefix):
    # Save logs first (best-effort)
    try:
        with sandbox.open("/workspace/stdout.log", "r") as f_in:
            with open(
                os.path.join(output_dir, uid, f"{prefix}_stdout.log"),
                "w",
                encoding="utf-8",
                errors="replace",
            ) as f:
                stdout_content = f_in.read()
                f.write(stdout_content if stdout_content is not None else "")
    except FileNotFoundError:
        pass
    try:
        with sandbox.open("/workspace/stderr.log", "r") as f_in:
            with open(
                os.path.join(output_dir, uid, f"{prefix}_stderr.log"),
                "w",
                encoding="utf-8",
                errors="replace",
            ) as f:
                stderr_content = f_in.read()
                f.write(stderr_content if stderr_content is not None else "")
    except FileNotFoundError:
        pass

    # Then try to read output.json
    try:
        with sandbox.open("/workspace/output.json", "r") as f_in:
            output = json.load(f_in)
            _write_output_snapshot(output_dir, uid, prefix, output)
            return output
    except FileNotFoundError:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return build_failure_output(
            output_dir=output_dir,
            uid=uid,
            prefix=prefix,
            failure_type="agent",
            status="missing_output_json",
            persist=True,
        )
    except json.JSONDecodeError as e:
        print(f"Warning: output.json was invalid JSON for {uid}: {e}")
        return build_failure_output(
            output_dir=output_dir,
            uid=uid,
            prefix=prefix,
            failure_type="agent",
            status="invalid_output_json",
            error=str(e),
            persist=True,
        )


def save_entryscript_copy(output_dir, uid, prefix, entryscript_content):
    with open(
        os.path.join(output_dir, uid, f"{prefix}_entryscript.sh"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(entryscript_content if entryscript_content is not None else "")


def eval_with_modal(
    patch,
    sample,
    output_dir,
    dockerhub_username,
    scripts_dir=None,
    dockerfiles_dir=None,
    prefix="",
    redo=False,
    block_network=False,
):
    if modal is None:
        raise RuntimeError(
            "modal is not installed. Install via 'pip install modal' or run without modal eval backend"
        )
    uid = sample["instance_id"]
    existing_output = prepare_run(uid, output_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    print(f"Running Modal evaluation for {uid}")
    try:
        write_patch_snapshot(output_dir, uid, prefix, patch)

        try:
            files, entryscript_content = assemble_workspace_files(
                uid, scripts_dir, dockerfiles_dir, patch, sample
            )
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return build_failure_output(
                output_dir=output_dir,
                uid=uid,
                prefix=prefix,
                failure_type="infra",
                status="missing_support_files",
                error=str(e),
                persist=False,
            )

        app = modal.App.lookup(name="swe-bench-pro-eval", create_if_missing=True)

        dockerhub_image_uri = get_dockerhub_image_uri(
            uid, dockerhub_username, sample.get("repo", "")
        )
        if not dockerhub_image_uri:
            print(
                f"Cannot construct image URI for {uid}: missing or malformed 'repo' field"
            )
            return build_failure_output(
                output_dir=output_dir,
                uid=uid,
                prefix=prefix,
                failure_type="infra",
                status="missing_docker_image_uri",
                persist=False,
            )
        print(f"Using Docker Hub image: {dockerhub_image_uri}")

        # Try the image as-is first; only fall back to ENTRYPOINT clearing if
        # Modal reports the sandbox disappeared unexpectedly.
        image_attempts = [
            ("default-entrypoint", modal.Image.from_registry(dockerhub_image_uri)),
            (
                "entrypoint-cleared",
                modal.Image.from_registry(dockerhub_image_uri).dockerfile_commands(
                    ["ENTRYPOINT []"]
                ),
            ),
        ]

        last_error = None
        for image_label, image in image_attempts:
            sandbox = None
            try:
                sandbox = modal.Sandbox.create(
                    image=image,
                    app=app,
                    timeout=SANDBOX_TIMEOUT,  # 1 hour: test suites can take 10–30+ min to compile and run
                    idle_timeout=SANDBOX_IDLE_TIMEOUT,
                    cpu=SANDBOX_CPU,
                    memory=SANDBOX_MEMORY,
                    block_network=block_network,
                )

                process = sandbox.exec("mkdir", "-p", "/workspace")
                process.wait()

                write_files_modal(sandbox, files)

                process = sandbox.exec("bash", "/workspace/entryscript.sh")
                process.wait()

                if process.returncode != 0:
                    print(
                        f"Entryscript failed for {uid} with return code: {process.returncode}"
                    )
                    try:
                        stderr_content = getattr(process, "stderr", None)
                        if stderr_content and hasattr(stderr_content, "read"):
                            error_details = stderr_content.read()
                            if error_details:
                                print(f"Error details for {uid}:")
                                print(error_details[:1000])
                    except Exception as e:
                        print(f"Failed to read stderr for {uid}: {e}")

                output = collect_outputs_modal(sandbox, output_dir, uid, prefix)
                save_entryscript_copy(output_dir, uid, prefix, entryscript_content)
                return output
            except Exception as e:
                last_error = e
                # Some images terminate quickly under one ENTRYPOINT strategy.
                # Retry once with the alternate strategy before failing.
                if image_label == "default-entrypoint" and _is_modal_sandbox_notfound(
                    e
                ):
                    print(
                        f"Sandbox vanished with default ENTRYPOINT for {uid}; retrying with cleared ENTRYPOINT"
                    )
                    continue
                raise
            finally:
                if sandbox:
                    try:
                        sandbox.terminate()
                    except Exception:
                        pass

        if last_error is not None:
            raise last_error
        return None
    except Exception as e:
        print(f"Error in eval_with_modal for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return build_failure_output(
            output_dir=output_dir,
            uid=uid,
            prefix=prefix,
            failure_type="infra",
            status="modal_eval_error",
            error=f"{type(e).__name__}: {e}",
            persist=False,
        )


def write_patch_snapshot(output_dir, uid, prefix, patch):
    with open(
        os.path.join(output_dir, uid, f"{prefix}_patch.diff"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(patch)


def assemble_workspace_files(uid, scripts_dir, dockerfiles_dir, patch, sample):
    run_script = load_local_script(scripts_dir, uid, "run_script.sh")
    parser_script = load_local_script(scripts_dir, uid, "parser.py")
    entryscript_content = create_entryscript(sample, dockerfiles_dir)

    cleaned_patch = strip_binary_hunks(patch)
    if cleaned_patch != patch:
        print(f"Stripped binary diff hunks from patch for {uid}")

    files = {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content
