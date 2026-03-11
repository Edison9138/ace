import ast
import json
import os
import hashlib
import threading
from typing import List, Dict, Any, Optional
import pandas as pd
from .eval_harness import eval_with_modal


def _safe_list(value) -> list:
    """Safely parse a list from a JSON string, a Python-repr string, or a real list.

    Tries json.loads() first (covers newly generated JSONL splits where
    prepare_data.py now stores fail_to_pass/pass_to_pass as proper JSON).
    Falls back to ast.literal_eval() for backwards-compatibility with existing
    splits that were produced with str() (Python repr with single quotes).
    Accepts an actual list unchanged.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: safely evaluate Python-repr strings like "['a', 'b']"
        try:
            result = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"Cannot parse list from string: {value!r}")
        if isinstance(result, list):
            return result
        raise ValueError(f"Expected a list after parsing, got {type(result).__name__}")
    raise TypeError(f"Cannot parse a list from {type(value).__name__}")


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


# ── DataProcessor ──────────────────────────────────────────────────────────────


class DataProcessor:
    _MAX_TEST_NAMES_IN_FEEDBACK = 8
    _MAX_LOG_LINES_IN_FEEDBACK = 20
    _MAX_LOG_CHARS_IN_FEEDBACK = 1200

    def __init__(
        self,
        raw_samples_path: str,  # path to sweap_eval_full_v2.jsonl
        scripts_dir: Optional[str],  # path to SWE-bench_Pro-os/run_scripts/
        dockerfiles_dir: Optional[str],  # path to SWE-bench_Pro-os/dockerfiles/
        dockerhub_username: str,  # DockerHub username for image URI construction
        eval_output_dir: str,  # where eval logs/output.json are saved
    ):
        with open(raw_samples_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        for row in data:
            if "context" in row and isinstance(row["context"], str):
                try:
                    context_data = json.loads(row["context"])
                    if "instance_id" in context_data:
                        row["instance_id"] = context_data["instance_id"]
                except json.JSONDecodeError:
                    pass
            elif "instance_id" not in row and isinstance(row.get("context"), dict):
                row["instance_id"] = row["context"].get("instance_id")

        df = pd.DataFrame(data).fillna("")
        # Deduplicate on instance_id, keeping the first occurrence.
        # Without this, self.raw_df.loc[instance_id] returns a DataFrame
        # (multiple rows) instead of a Series when duplicates exist, which
        # silently breaks create_entryscript() that expects scalar field values.
        dupes = df.duplicated(subset=["instance_id"], keep=False)
        if dupes.any():
            n_dupes = dupes.sum()
            import warnings

            warnings.warn(
                f"DataProcessor: found {n_dupes} rows with duplicate instance_ids "
                f"in {raw_samples_path}; keeping first occurrence of each."
            )
            df = df.drop_duplicates(subset=["instance_id"], keep="first")
        self.raw_df = df.set_index("instance_id", drop=False)
        self.scripts_dir = scripts_dir
        self.dockerfiles_dir = dockerfiles_dir
        self.dockerhub_username = dockerhub_username
        self.eval_output_dir = eval_output_dir
        # SWE-bench-Pro accuracy should aggregate first-pass instance outcomes.
        # Re-invoking the harness in evaluate_accuracy can introduce extra
        # infra noise and diverge from benchmark-style single-pass semantics.
        self.supports_single_pass_accuracy = True
        # Evaluation-time exceptions must count against accuracy for SWE runs,
        # otherwise infra drops can silently inflate scores.
        self.supports_failure_accounting = True
        self._thread_state = threading.local()
        os.makedirs(eval_output_dir, exist_ok=True)

    def _set_last_failure_type(self, failure_type: str) -> None:
        self._thread_state.last_failure_type = failure_type

    def get_last_failure_type(self) -> str:
        return getattr(self._thread_state, "last_failure_type", "none")

    def _set_last_environment_feedback(self, feedback: str) -> None:
        self._thread_state.last_environment_feedback = feedback or ""

    def get_last_environment_feedback(self) -> str:
        return getattr(self._thread_state, "last_environment_feedback", "")

    def _format_name_list(self, names: set[str] | list[str]) -> str:
        cleaned_names = sorted({str(name) for name in names if name})
        if not cleaned_names:
            return "(none)"
        if len(cleaned_names) <= self._MAX_TEST_NAMES_IN_FEEDBACK:
            return ", ".join(cleaned_names)
        shown = cleaned_names[: self._MAX_TEST_NAMES_IN_FEEDBACK]
        remaining = len(cleaned_names) - len(shown)
        return f"{', '.join(shown)}, ... (+{remaining} more)"

    def _read_eval_log_excerpt(
        self, instance_id: str, eval_prefix: str, stream_name: str
    ) -> str:
        path = os.path.join(
            self.eval_output_dir, instance_id, f"{eval_prefix}_{stream_name}.log"
        )
        if not os.path.exists(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
        except OSError:
            return ""
        if not content:
            return ""
        excerpt = "\n".join(content.splitlines()[-self._MAX_LOG_LINES_IN_FEEDBACK :])
        if len(excerpt) > self._MAX_LOG_CHARS_IN_FEEDBACK:
            excerpt = excerpt[-self._MAX_LOG_CHARS_IN_FEEDBACK :]
        return excerpt.strip()

    def _build_environment_feedback(
        self,
        *,
        instance_id: str,
        eval_prefix: str,
        output: Optional[dict],
        fail_to_pass: set[str],
        pass_to_pass: set[str],
    ) -> str:
        tests = output.get("tests", []) if isinstance(output, dict) else []
        meta = output.get("_ace_meta", {}) if isinstance(output, dict) else {}
        passed = {
            test.get("name")
            for test in tests
            if test.get("name") and test.get("status") == "PASSED"
        }
        error_tests = {
            test.get("name")
            for test in tests
            if test.get("name") and test.get("status") == "ERROR"
        }
        other_nonpassed = {
            f"{test.get('name')} ({test.get('status')})"
            for test in tests
            if test.get("name") and test.get("status") not in {None, "PASSED", "ERROR"}
        }
        missing_fail_to_pass = fail_to_pass - passed
        missing_pass_to_pass = pass_to_pass - passed
        stderr_excerpt = self._read_eval_log_excerpt(instance_id, eval_prefix, "stderr")
        stdout_excerpt = self._read_eval_log_excerpt(instance_id, eval_prefix, "stdout")

        lines = ["SWE-bench-Pro evaluator summary:", f"- Instance ID: {instance_id}"]
        if output is None:
            lines.append("- Eval harness did not produce output.json.")
        elif meta:
            status = meta.get("status")
            failure_type = meta.get("failure_type")
            error = meta.get("error")
            lines.append(
                "- Eval harness status: "
                f"{status or 'unknown'} (classified as {failure_type or 'unknown'})"
            )
            if error:
                lines.append(f"- Eval harness error: {error}")
            if not tests:
                lines.append("- Parsed test results: no test statuses were extracted.")
        elif not tests:
            lines.append("- Parsed test results: no test statuses were extracted.")
        else:
            lines.append(f"- Passed tests reported: {self._format_name_list(passed)}")
            if error_tests:
                lines.append(
                    f"- Parser reported ERROR tests: {self._format_name_list(error_tests)}"
                )
            if other_nonpassed:
                lines.append(
                    "- Other non-passing parser statuses: "
                    f"{self._format_name_list(other_nonpassed)}"
                )

        if missing_fail_to_pass:
            lines.append(
                "- Required fail_to_pass tests still not passing: "
                f"{self._format_name_list(missing_fail_to_pass)}"
            )
        if missing_pass_to_pass:
            lines.append(
                "- Required pass_to_pass tests missing or regressed: "
                f"{self._format_name_list(missing_pass_to_pass)}"
            )
        if output is not None and not missing_fail_to_pass and not missing_pass_to_pass:
            lines.append("- All required fail_to_pass and pass_to_pass tests passed.")

        if stderr_excerpt:
            lines.extend(["- stderr tail:", "```", stderr_excerpt, "```"])
        if stdout_excerpt:
            lines.extend(["- stdout tail:", "```", stdout_excerpt, "```"])

        return "\n".join(lines)

    # ── Standard ACE interface methods ────────────────────────────────────────

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw JSONL rows into the ACE standard format.

        prepare_data.py already writes {context, question, target}, so this
        method validates the required keys are present and returns the list
        unchanged.  Downstream code (ace.py, utils.py) relies on exactly these
        three keys.
        """
        processed = []
        for i, item in enumerate(raw_data):
            for key in ("context", "question", "target"):
                if key not in item:
                    raise ValueError(
                        f"Row {i} is missing required key '{key}'. "
                        "Re-run prepare_data.py to regenerate the JSONL splits."
                    )
            processed.append(
                {
                    "context": item["context"],
                    "question": item["question"],
                    "target": item["target"],
                }
            )
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        self._set_last_failure_type("none")
        self._set_last_environment_feedback("")
        if not predicted:
            self._set_last_failure_type("agent")
            self._set_last_environment_feedback(
                "SWE-bench-Pro evaluator summary:\n- No patch was submitted by the generator."
            )
            return False

        meta = json.loads(ground_truth)
        instance_id = meta["instance_id"]
        fail_to_pass = set(_safe_list(meta["fail_to_pass"]))
        pass_to_pass = set(_safe_list(meta["pass_to_pass"]))
        eval_prefix = hashlib.sha1(predicted.encode("utf-8")).hexdigest()[:12]

        if instance_id not in self.raw_df.index:
            self._set_last_failure_type("infra")
            self._set_last_environment_feedback(
                "SWE-bench-Pro evaluator summary:\n"
                f"- Instance ID: {instance_id}\n"
                "- Eval could not start because the raw sample metadata was not found."
            )
            return False

        output = eval_with_modal(
            patch=predicted,
            sample=self.raw_df.loc[instance_id],
            output_dir=self.eval_output_dir,
            dockerhub_username=self.dockerhub_username,
            scripts_dir=self.scripts_dir,
            dockerfiles_dir=self.dockerfiles_dir,
            prefix=eval_prefix,
        )
        if output is None:
            self._set_last_failure_type("infra")
            self._set_last_environment_feedback(
                self._build_environment_feedback(
                    instance_id=instance_id,
                    eval_prefix=eval_prefix,
                    output=None,
                    fail_to_pass=fail_to_pass,
                    pass_to_pass=pass_to_pass,
                )
            )
            return False

        meta_output = output if isinstance(output, dict) else None
        meta = meta_output.get("_ace_meta", {}) if meta_output else {}
        self._set_last_environment_feedback(
            self._build_environment_feedback(
                instance_id=instance_id,
                eval_prefix=eval_prefix,
                output=output,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
            )
        )

        passed = {
            t.get("name") for t in output.get("tests", []) if t.get("status") == "PASSED"
        }
        is_correct = (fail_to_pass | pass_to_pass) <= passed
        if meta.get("failure_type") == "infra":
            self._set_last_failure_type("infra")
        elif not is_correct:
            self._set_last_failure_type("agent")
        return is_correct

    def evaluate_accuracy(self, predictions: list, targets: list) -> float:
        if not predictions:
            return 0.0
        correct = sum(
            self.answer_is_correct(p, t) for p, t in zip(predictions, targets)
        )
        return correct / len(predictions)
