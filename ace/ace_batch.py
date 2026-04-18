"""
ACE (Agent-Curator-Environment) System
Main orchestrator class for training and testing with playbook-based learning.

This module coordinates three agents:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and tags bullets
- Curator: Updates the playbook based on feedback
"""

import os
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core import Generator, Reflector, Curator, BulletpointAnalyzer
from playbook_utils import *
from logger import *
from utils import *

# Placeholder answer when API error occurs - will be marked incorrect by answer_is_correct
INCORRECT_DUE_TO_API_ERROR = "INCORRECT_DUE_TO_API_ERROR"


class ACEBatch:
    """
    Batched ACE: parallel generator+reflector per mini-batch, chunked curator (cbs), then parallel post-curate.
    """
    
    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90
    ):
        """
        Initialize the ACE system.
        
        Args:
            api_provider: API provider for LLM calls
            generator_model: Model name for generator
            reflector_model: Model name for reflector
            curator_model: Model name for curator
            max_tokens: Maximum tokens for LLM calls
            initial_playbook: Initial playbook content (optional)
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer for deduplication
            bulletpoint_analyzer_threshold: Similarity threshold for bulletpoint analyzer (0-1)
        """
        # Initialize API clients
        generator_client, reflector_client, curator_client = initialize_clients(api_provider)

        # Initialize the three agents
        self.api_provider = api_provider
        self.generator = Generator(generator_client, api_provider, generator_model, max_tokens)
        self.reflector = Reflector(reflector_client, api_provider, reflector_model, max_tokens)
        self.curator = Curator(curator_client, api_provider, curator_model, max_tokens)
        
        # Initialize bulletpoint analyzer if requested and available
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold
        
        if use_bulletpoint_analyzer:
            self.bulletpoint_analyzer = BulletpointAnalyzer(
                curator_client, 
                curator_model, 
                max_tokens
            )
            print(f"✓ BulletpointAnalyzer initialized (threshold={bulletpoint_analyzer_threshold})")
        else:
            self.bulletpoint_analyzer = None
        
        # Store configuration
        self.generator_client = generator_client
        self.reflector_client = reflector_client
        self.curator_client = curator_client
        self.max_tokens = max_tokens
        
        # Initialize playbook
        if initial_playbook:
            self.playbook = initial_playbook
        else:
            self.playbook = self._initialize_empty_playbook()
        
        self.best_playbook = self.playbook
        # Seed next_global_id from loaded playbook to avoid bullet ID collisions.
        self.next_global_id = get_next_global_id(self.playbook)
    
    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""
    
    def _extract_config_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with extracted parameters
        """
        batch_size = int(config.get("batch_size", 1))
        cbs = config.get("curator_batch_size")
        if cbs is None:
            cbs = batch_size
        else:
            cbs = int(cbs)
        cbs = max(1, cbs)
        use_aug = config.get("augmented_shuffling", True)
        aug_factor = int(config.get("augmented_shuffling_factor", 2))
        if not use_aug:
            aug_factor = 1
        else:
            aug_factor = max(1, aug_factor)
        return {
            'num_epochs': config.get('num_epochs', 1),
            'max_num_rounds': config.get('max_num_rounds', 3),
            'curator_frequency': config.get('curator_frequency', 1),
            'eval_steps': config.get('eval_steps', 100),
            'save_steps': config.get('save_steps', 50),
            'token_budget': config.get('playbook_token_budget', 80000),
            'task_name': config.get('task_name', 'default'),
            'use_json_mode': config.get('json_mode', False),
            'no_ground_truth': config.get('no_ground_truth', False),
            'save_dir': config.get('save_dir', './results'),
            'test_workers': config.get('test_workers', 20),
            'use_bulletpoint_analyzer': config.get('use_bulletpoint_analyzer', False),
            'bulletpoint_analyzer_threshold': config.get('bulletpoint_analyzer_threshold', 0.90),
            'batch_size': batch_size,
            'curator_batch_size': cbs,
            'augmented_shuffling_factor': aug_factor,
            'continue_on_llm_error': config.get('continue_on_llm_error', False),
        }

    def _use_single_pass_accuracy(self, data_processor) -> bool:
        """Whether the task opts into first-pass accuracy aggregation."""
        return supports_single_pass_accuracy(data_processor)

    @staticmethod
    def _accuracy_from_correctness_flags(correct_flags: List[bool]) -> float:
        """Compute accuracy directly from per-sample correctness flags."""
        if not correct_flags:
            return 0.0
        return sum(1 for is_correct in correct_flags if is_correct) / len(correct_flags)

    @staticmethod
    def _is_api_error(error: Exception) -> bool:
        """Best-effort classifier for transient/recoverable API failures."""
        text = f"{type(error).__name__}: {error}".lower()
        markers = (
            "apierror",
            "apiconnectionerror",
            "rate limit",
            "429",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "bad gateway",
            "gateway timeout",
            "server disconnected",
            "502",
            "503",
            "504",
        )
        return any(marker in text for marker in markers)

    def _should_continue_on_error(
        self,
        error: Exception,
        config_params: Dict[str, Any],
    ) -> bool:
        """Whether a batch error should be converted to an incorrect sample."""
        return bool(config_params.get("continue_on_llm_error", False)) or self._is_api_error(error)

    @staticmethod
    def _get_infra_failure_source(
        generation_call_info: Dict[str, Any],
        current_is_correct: bool,
        data_processor,
    ) -> Optional[str]:
        """
        Detect infra failures from generator call info and eval harness metadata.
        """
        call_info = generation_call_info if isinstance(generation_call_info, dict) else {}
        gen_failure_type = call_info.get("failure_type", "none")
        if gen_failure_type == "infra":
            return "generator"

        if (
            not current_is_correct
            and gen_failure_type == "none"
            and hasattr(data_processor, "get_last_failure_type")
        ):
            try:
                eval_failure_type = data_processor.get_last_failure_type()
            except Exception:
                eval_failure_type = "none"
            if eval_failure_type == "infra":
                return "eval harness"

        return None

    def _setup_paths(self, save_dir: str, task_name: str, mode: str) -> Tuple[str, str]:
        """
        Setup logging paths and directories.
        
        Args:
            save_dir: Base path for saving results
            task_name: task name
            mode: 'offline', 'online', or 'eval_only'
            
        Returns:
            Tuple of (usage_log_path, playbook_dir)
        """
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"ace_run_{timestamp}_{task_name}_{mode}"
        save_path = os.path.join(save_dir, run_folder)
        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        if mode == "eval_only":
            return save_path, log_dir

        usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
        playbook_dir = os.path.join(save_path, "intermediate_playbooks")
        os.makedirs(playbook_dir, exist_ok=True)
        
        return save_path, usage_log_path, playbook_dir, log_dir
    
    def run(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entrypoint for running ACE system in different modes.
        
        Args:
            mode: Run mode - 'offline', 'online', or 'eval_only'
            train_samples: Training samples (required for offline mode)
            val_samples: Validation samples (required for offline mode)
            test_samples: Test samples (required for online and eval_only modes)
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            
        Returns:
            Dictionary with results depending on the mode
        """
        # Validate inputs
        if mode not in ['offline', 'online', 'eval_only']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'offline', 'online', or 'eval_only'")
        
        if mode == 'offline' and (train_samples is None or val_samples is None):
            raise ValueError("Offline mode requires train_samples and val_samples")
        
        if mode == 'online' and test_samples is None:
            raise ValueError("Online mode requires test_samples")
        
        if mode == 'eval_only' and test_samples is None:
            raise ValueError("eval_only mode requires test_samples")
        
        # Extract configuration
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        save_dir = config_params['save_dir']
        batch_size = config_params['batch_size']
        # Setup paths based on mode
        if mode == 'eval_only':
            save_path, log_dir = self._setup_paths(save_dir, task_name, mode)
            usage_log_path = None
            playbook_dir = None
        else:
            save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(save_dir, task_name, mode)
        
        # Save configuration
        config_path = os.path.join(save_path, "run_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "task_name": task_name,
                "mode": mode,
                "generator_model": self.generator.model,
                "reflector_model": self.reflector.model,
                "curator_model": self.curator.model,
                "config": config,
            }, f, indent=2)
        
        # Print initial banner
        print(f"\n{'='*60}")
        print(f"ACE SYSTEM - {mode.upper().replace('_', ' ')} MODE")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        if mode == 'offline':
            print(f"Train samples: {len(train_samples)}")
            print(f"Validation samples: {len(val_samples)}")
            if test_samples:
                print(f"Test samples: {len(test_samples)}")
        elif mode == 'online':
            print(f"Test samples (used for training and testing): {len(test_samples)}")
        else:  # eval_only
            print(f"Test samples: {len(test_samples)}")
        print(f"{'='*60}\n")
        
        # Execute based on mode
        results = {}
        
        if mode == 'offline':
            # OFFLINE MODE WORKFLOW
            # 1. Run initial test if test_samples provided
            if test_samples:
                print(f"\n{'='*60}")
                print(f"INITIAL TEST (before training)")
                print(f"{'='*60}\n")
                initial_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="initial"
                )
                results['initial_test_results'] = initial_test_results
                print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run offline training
            print(f"\n{'='*60}")
            print(f"STARTING OFFLINE TRAINING")
            print(f"{'='*60}\n")
            training_results = self._offline_train(
                train_samples=train_samples,
                val_samples=val_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir,
                batch_size = batch_size
            )
            results['training_results'] = training_results
            
            # 3. Run final test if test_samples provided
            if test_samples:
                print(f"\n{'='*60}")
                print(f"FINAL TEST (with best playbook)")
                print(f"{'='*60}\n")
                final_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.best_playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="final"
                )
                results['final_test_results'] = final_test_results
                print(f"Final Test Accuracy: {final_test_results['accuracy']:.3f}\n")
        
        elif mode == 'online':
            # ONLINE MODE WORKFLOW
            # 1. Run initial test
            print(f"\n{'='*60}")
            print(f"INITIAL TEST (before training)")
            print(f"{'='*60}\n")
            initial_test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="initial"
            )
            results['initial_test_results'] = initial_test_results
            print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run online training and testing
            print(f"\n{'='*60}")
            print(f"STARTING ONLINE TRAIN AND TEST")
            print(f"{'='*60}\n")
            online_results = self._online_train_and_test(
                test_samples=test_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir
            )
            results['online_test_results'] = online_results
        
        else:  # eval_only
            # EVAL ONLY MODE WORKFLOW
            print(f"\n{'='*60}")
            print(f"RUNNING TEST")
            print(f"{'='*60}\n")
            test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="test"
            )
            results['test_results'] = test_results
        
        # Save consolidated results
        final_results_path = os.path.join(save_path, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Mode: {mode.upper().replace('_', ' ')}")
        if mode == 'offline':
            print(f"Best Validation Accuracy: {results['training_results']['best_validation_accuracy']:.3f}")
            if test_samples:
                print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
                print(f"Final Test Accuracy: {results['final_test_results']['accuracy']:.3f}")
        elif mode == 'online':
            print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
            print(f"Final Test Accuracy: {results['online_test_results']['accuracy']:.3f}")
        else:  # eval_only
            print(f"Test Accuracy: {results['test_results']['accuracy']:.3f}")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _run_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook: str,
        config: Dict[str, Any],
        log_dir: str,
        save_path: str,
        prefix: str = "test"
    ) -> Dict[str, Any]:
        """
        Run testing
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to use for testing
            config: Configuration dictionary
            log_dir: Directory for detailed logs
            save_path: Path to save results
            prefix: Prefix for saved files (e.g., 'initial', 'final', 'test')
            
        Returns:
            Dictionary with test results
        """
        config_params = self._extract_config_params(config)
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        
        test_results, test_error_log = evaluate_test_set(
            data_processor,
            self.generator,
            playbook,
            test_samples,
            self.max_tokens,
            log_dir,
            max_workers=test_workers,
            use_json_mode=use_json_mode
        )

        # Save test results
        test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_results": test_results,
                "error_log": test_error_log,
            }, f, indent=2)
        
        return test_results
    
    def _generate_and_reflect_single_sample(
        self,
        task_dict: Dict[str, Any],
        data_processor,
        playbook_snapshot: str,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Thread-safe: Run generator + reflector for a single sample.
        
        Uses a playbook snapshot (read-only from the caller's perspective) instead
        of modifying self.playbook, so multiple threads can run concurrently.
        
        Args:
            task_dict: Sample dictionary with question, context, target
            data_processor: Data processor for evaluation
            playbook_snapshot: Snapshot of the playbook at batch start (thread-safe)
            step_id: Identifier string for this step
            epoch: Current epoch number
            step: Current step number
            usage_log_path: Path for bullet usage logging
            log_dir: Path for logging directory
            config_params: Configuration parameters dictionary
            
        Returns:
            Dictionary with results for aggregation:
                - task_dict, pre_train_answer, final_answer, is_correct
                - reflection_content, all_bullet_tags (for curator aggregation)
                - context, step_id, tracking_dict
        """
        max_num_rounds = config_params["max_num_rounds"]
        use_json_mode = config_params["use_json_mode"]
        no_ground_truth = config_params["no_ground_truth"]

        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "")

        reflector_ground_truth = target if not no_ground_truth else None
        if reflector_ground_truth is not None and hasattr(
            self.generator, "get_ground_truth_for_reflector"
        ):
            reflector_ground_truth = self.generator.get_ground_truth_for_reflector(target)

        curator_question_context = context
        if hasattr(self.generator, "get_question_context_for_curator"):
            curator_question_context = self.generator.get_question_context_for_curator(
                question, context
            )

        should_learn_from_initially_correct = True
        if hasattr(self.generator, "should_learn_from_initially_correct_samples"):
            should_learn_from_initially_correct = (
                self.generator.should_learn_from_initially_correct_samples()
            )

        def _get_bullets_for_reflector(playbook: str, ids: List[str]) -> str:
            if hasattr(self.generator, "get_bullets_for_reflector"):
                return self.generator.get_bullets_for_reflector(playbook, ids)
            return extract_playbook_bullets(playbook, ids)

        # Work with a local copy of the playbook (thread-safe)
        local_playbook = playbook_snapshot

        # STEP 1: Initial generation (pre-train)
        print(f"[{step_id}] Generating initial answer...")
        gen_response, bullet_ids, call_info = self.generator.generate(
            question=question,
            playbook=local_playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir,
        )

        # Extract answer and check correctness
        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)
        pre_train_answer = final_answer
        pre_train_was_correct = is_correct
        failure_type = "none"
        curator_reasoning_trace = extract_reasoning_trace(
            gen_response, field="curator_reasoning"
        )

        print(f"[{step_id}] Correct: {is_correct}")

        # Log bullet usage
        log_bullet_usage(
            usage_log_path,
            epoch,
            step,
            task_dict,
            bullet_ids,
            playbook=local_playbook,
            is_correct=is_correct,
        )

        # Collect all bullet tags for later aggregation (not applied to shared playbook)
        all_bullet_tags = []
        reflection_content = "(empty)"

        tracking_dict = {
            "pre_train_result": {
                "final_answer": pre_train_answer,
                "is_correct": pre_train_was_correct,
                "playbook_num_tokens": count_tokens(playbook_snapshot),
                "playbook_length": len(playbook_snapshot),
            }
        }

        def _finish_sample_with_infra_failure(
            source: str,
            final_answer_for_record: str,
        ) -> Dict[str, Any]:
            print(
                f"[{step_id}] Infrastructure failure ({source}) - "
                f"skipping reflection/curation/post-curate"
            )
            tracking_dict["pre_train_result"]["failure_type"] = "infra"
            tracking_dict["post_train_result"] = {
                "final_answer": final_answer_for_record,
                "is_correct": False,
                "failure_type": "infra",
                "playbook_num_tokens": count_tokens(playbook_snapshot),
                "playbook_length": len(playbook_snapshot),
            }
            return {
                "task_dict": task_dict,
                "pre_train_answer": pre_train_answer,
                "final_answer": final_answer_for_record,
                "is_correct": False,
                "reflection_content": reflection_content,
                "all_bullet_tags": [],
                "context": context,
                "question": question,
                "target": target,
                "step_id": step_id,
                "curator_question_context": curator_question_context,
                "curator_reasoning_trace": curator_reasoning_trace,
                "pre_train_was_correct": pre_train_was_correct,
                "skip_learning": True,
                "skip_post_curate_generation": True,
                "failure_type": "infra",
                "tracking_dict": tracking_dict,
            }

        infra_failure_source = self._get_infra_failure_source(
            call_info,
            is_correct,
            data_processor,
        )
        if infra_failure_source:
            failure_type = "infra"
            return _finish_sample_with_infra_failure(
                infra_failure_source, pre_train_answer
            )

        skip_learning = False

        # STEP 2: Reflection and regeneration
        if not is_correct:
            # For incorrect answers - iterate reflection rounds
            for round_num in range(max_num_rounds):
                print(f"[{step_id}] Reflection round {round_num + 1}/{max_num_rounds}")

                playbook_bullets = _get_bullets_for_reflector(local_playbook, bullet_ids)

                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question,
                    reasoning_trace=extract_reasoning_trace(gen_response),
                    predicted_answer=final_answer,
                    ground_truth=reflector_ground_truth,
                    environment_feedback=get_environment_feedback(
                        data_processor,
                        "Predicted answer does not match ground truth",
                    ),
                    bullets_used=playbook_bullets,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir,
                )

                # Collect bullet tags (apply to local copy only)
                if bullet_tags:
                    all_bullet_tags.extend(bullet_tags)
                    local_playbook = update_bullet_counts(local_playbook, bullet_tags)

                # Regenerate with reflection
                gen_response, bullet_ids, post_reflect_call_info = self.generator.generate(
                    question=question,
                    playbook=local_playbook,
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir,
                )

                final_answer = extract_answer(gen_response)
                is_correct = data_processor.answer_is_correct(final_answer, target)
                curator_reasoning_trace = extract_reasoning_trace(
                    gen_response, field="curator_reasoning"
                )

                infra_failure_source = self._get_infra_failure_source(
                    post_reflect_call_info,
                    is_correct,
                    data_processor,
                )
                if infra_failure_source:
                    failure_type = "infra"
                    return _finish_sample_with_infra_failure(
                        infra_failure_source, final_answer
                    )

                if is_correct:
                    print(
                        f"[{step_id}] Corrected after reflection round "
                        f"{round_num + 1}!"
                    )
                    log_bullet_usage(
                        usage_log_path,
                        epoch,
                        step,
                        task_dict,
                        bullet_ids,
                        playbook=local_playbook,
                        reflection_content=reflection_content,
                        is_correct=True,
                    )
                    break

        else:
            if should_learn_from_initially_correct:
                playbook_bullets = _get_bullets_for_reflector(local_playbook, bullet_ids)

                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question,
                    reasoning_trace=extract_reasoning_trace(gen_response),
                    predicted_answer=final_answer,
                    ground_truth=reflector_ground_truth,
                    environment_feedback="Predicted answer matches ground truth",
                    bullets_used=playbook_bullets,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_reflect_on_correct",
                    log_dir=log_dir,
                )

                if bullet_tags:
                    all_bullet_tags.extend(bullet_tags)
            else:
                print(
                    f"[{step_id}] Skipping reflection/curation for initially "
                    "correct sample"
                )
                skip_learning = True

            log_bullet_usage(
                usage_log_path,
                epoch,
                step,
                task_dict,
                bullet_ids,
                playbook=local_playbook,
                reflection_content=reflection_content,
                is_correct=is_correct,
            )

        skip_post_curate_generation = is_correct
        if failure_type == "infra":
            skip_post_curate_generation = True

        return {
            "task_dict": task_dict,
            "pre_train_answer": pre_train_answer,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "reflection_content": reflection_content,
            "all_bullet_tags": all_bullet_tags,
            "context": context,
            "question": question,
            "target": target,
            "step_id": step_id,
            "curator_question_context": curator_question_context,
            "curator_reasoning_trace": curator_reasoning_trace,
            "pre_train_was_correct": pre_train_was_correct,
            "skip_learning": skip_learning,
            "skip_post_curate_generation": skip_post_curate_generation,
            "failure_type": failure_type,
            "tracking_dict": tracking_dict,
        }
    
    def _train_batch(
        self,
        batch: List[Dict[str, Any]],
        data_processor,
        batch_step_start: int,
        epoch: int,
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
        total_samples: int,
        step_id_prefix: str = "train"
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Train on a batch with async parallel generator+reflector, then sync for curator.
        
        Architecture:
            Phase 1 (PARALLEL): Run generator + reflector for each sample in separate threads
            Phase 2 (SYNC):     Aggregate all bullet tags and reflections, run curator once
            Phase 3 (PARALLEL): Run post-curator generation for each sample in separate threads
        
        Args:
            batch: List of task dictionaries (one per sample)
            data_processor: Data processor for evaluation
            batch_step_start: Starting step number for this batch (1-indexed)
            epoch: Current epoch number
            usage_log_path: Path for bullet usage logging
            log_dir: Path for logging directory
            config_params: Configuration parameters dictionary
            total_samples: Total number of samples in the full dataset
            step_id_prefix: Prefix for step IDs (e.g., "train_e_1" or "online_train")
            
        Returns:
            List of (pre_train_answer, post_train_answer, tracking_dict) tuples, one per sample
        """
        token_budget = config_params["token_budget"]
        use_json_mode = config_params["use_json_mode"]
        no_ground_truth = config_params["no_ground_truth"]
        curator_frequency = max(1, int(config_params.get("curator_frequency", 1)))

        # Take a snapshot of the current playbook for all threads
        playbook_snapshot = self.playbook

        # ================================================================
        # PHASE 1: Parallel Generator + Reflector (one thread per sample)
        # ================================================================
        print(f"\n{'='*40}")
        print(f"PHASE 1: Parallel Generator + Reflector ({len(batch)} samples)")
        print(f"{'='*40}")

        sample_results = [None] * len(batch)

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_idx = {}
            for i, task_dict in enumerate(batch):
                step = batch_step_start + i
                step_id = f"{step_id_prefix}_s_{step}"

                future = executor.submit(
                    self._generate_and_reflect_single_sample,
                    task_dict=task_dict,
                    data_processor=data_processor,
                    playbook_snapshot=playbook_snapshot,
                    step_id=step_id,
                    epoch=epoch,
                    step=step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    sample_results[idx] = future.result()
                    print(
                        f"  Sample {idx + 1}/{len(batch)} complete "
                        f"(correct: {sample_results[idx]['is_correct']})"
                    )
                except Exception as e:
                    if self._should_continue_on_error(e, config_params):
                        print(
                            f"  API/LLM ERROR in sample {idx + 1}/{len(batch)}: {e} - "
                            "marking as incorrect, continuing"
                        )
                        task_dict = batch[idx]
                        step = batch_step_start + idx
                        step_id = f"{step_id_prefix}_s_{step}"
                        sample_results[idx] = {
                            "task_dict": task_dict,
                            "pre_train_answer": INCORRECT_DUE_TO_API_ERROR,
                            "final_answer": INCORRECT_DUE_TO_API_ERROR,
                            "is_correct": False,
                            "reflection_content": "(empty)",
                            "all_bullet_tags": [],
                            "context": task_dict.get("context", ""),
                            "question": task_dict.get("question", ""),
                            "target": task_dict.get("target", ""),
                            "step_id": step_id,
                            "curator_question_context": task_dict.get("context", ""),
                            "curator_reasoning_trace": "(not available)",
                            "pre_train_was_correct": False,
                            "skip_learning": True,
                            "skip_post_curate_generation": True,
                            "failure_type": "infra",
                            "tracking_dict": {
                                "pre_train_result": {
                                    "final_answer": INCORRECT_DUE_TO_API_ERROR,
                                    "is_correct": False,
                                    "failure_type": "infra",
                                    "playbook_num_tokens": count_tokens(playbook_snapshot),
                                    "playbook_length": len(playbook_snapshot),
                                }
                            },
                        }
                    else:
                        print(f"  ERROR in sample {idx + 1}/{len(batch)}: {e}")
                        raise

        print(f"Phase 1 complete: All {len(batch)} samples processed")

        # ================================================================
        # PHASE 2: Aggregate bullet tags + Run Curator
        # ================================================================
        curator_batch_size = config_params.get("curator_batch_size", 10)

        print(f"\n{'='*40}")
        print(
            f"PHASE 2: Aggregation + Curator "
            f"(curator_batch_size={curator_batch_size})"
        )
        print(f"{'='*40}")

        # Aggregate tags and curator inputs from samples that are valid for learning.
        all_bullet_tags = []
        curator_entries = []
        api_error_count = 0
        skipped_learning_count = 0
        infra_failure_count = 0
        for result in sample_results:
            if result.get("pre_train_answer") == INCORRECT_DUE_TO_API_ERROR:
                api_error_count += 1
                continue
            if result.get("failure_type") == "infra":
                infra_failure_count += 1
                continue
            if result.get("skip_learning", False):
                skipped_learning_count += 1
                continue
            all_bullet_tags.extend(result.get("all_bullet_tags", []))
            curator_entries.append(
                {
                    "reflection": result.get("reflection_content", "(empty)"),
                    "context": result.get(
                        "curator_question_context", result.get("context", "")
                    ),
                    "reasoning_trace": result.get(
                        "curator_reasoning_trace", "(not available)"
                    ),
                }
            )
        if api_error_count:
            print(
                f"  Excluded {api_error_count} API-error sample(s) from "
                "Phase 2 aggregation"
            )
        if infra_failure_count:
            print(
                f"  Excluded {infra_failure_count} infra-failure sample(s) from "
                "Phase 2 aggregation"
            )
        if skipped_learning_count:
            print(
                f"  Excluded {skipped_learning_count} initially-correct "
                "sample(s) from Phase 2 aggregation"
            )

        # Augmented Shuffling (Hive): duplicate each entry p times, shuffle.
        augmented_factor = config_params.get("augmented_shuffling_factor", 1)
        if augmented_factor > 1 and curator_entries:
            curator_entries = [
                entry for entry in curator_entries for _ in range(augmented_factor)
            ]
            random.shuffle(curator_entries)
            print(
                f"  [Augmented Shuffling] factor={augmented_factor} | "
                f"{len(curator_entries) // augmented_factor} entries -> "
                f"{len(curator_entries)}"
            )

        # Save playbook and next_global_id before Phase 2 updates (for rollback on
        # Phase 3 API errors).
        playbook_before_phase2 = self.playbook
        next_global_id_before_phase2 = self.next_global_id

        # Apply aggregated bullet tags to the shared playbook (once, before Curator).
        if all_bullet_tags:
            self.playbook = update_bullet_counts(self.playbook, all_bullet_tags)
            print(
                f"  Applied {len(all_bullet_tags)} bullet tag updates "
                f"from {len(batch)} samples"
            )

        last_batch_step = batch_step_start + len(batch) - 1
        curator_trigger_steps = [
            step for step in range(batch_step_start, last_batch_step + 1)
            if step % curator_frequency == 0
        ]
        should_run_curator = bool(curator_entries) and bool(curator_trigger_steps)
        curator_step_for_batch = (
            curator_trigger_steps[-1] if curator_trigger_steps else last_batch_step
        )

        def _run_one_curator_call(
            combined_reflection: str,
            combined_context: str,
            combined_reasoning_trace: str,
            last_step: int,
            call_id: str,
            diag_chunk_size: int,
        ) -> None:
            try:
                cr_tokens = count_tokens(combined_reflection)
                cc_tokens = count_tokens(combined_context)
                rt_tokens = count_tokens(combined_reasoning_trace)
                pb_tokens = count_tokens(self.playbook)
                print(
                    f"  [DIAG] curator_chunk_size={diag_chunk_size} | "
                    f"reflection={cr_tokens} tok | context={cc_tokens} tok | "
                    f"reasoning={rt_tokens} tok | playbook={pb_tokens} tok | "
                    f"total~{cr_tokens + cc_tokens + rt_tokens + pb_tokens} tok"
                )
            except Exception:
                pass

            stats = (
                self.generator.get_playbook_stats_for_curator(self.playbook)
                if hasattr(self.generator, "get_playbook_stats_for_curator")
                else get_playbook_stats(self.playbook)
            )
            playbook_for_curator = (
                self.generator.get_playbook_for_curator(self.playbook)
                if hasattr(self.generator, "get_playbook_for_curator")
                else self.playbook
            )
            self.playbook, self.next_global_id, _, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=combined_reflection,
                question_context=combined_context,
                current_step=last_step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=call_id,
                log_dir=log_dir,
                next_global_id=self.next_global_id,
                reasoning_trace=combined_reasoning_trace,
                prompt_playbook=playbook_for_curator,
            )
            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                print(
                    f"  Running BulletpointAnalyzer "
                    f"(threshold={self.bulletpoint_analyzer_threshold})..."
                )
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True,
                )

        if not should_run_curator:
            print(
                "  Skipping Curator for this batch "
                f"(learnable_entries={len(curator_entries)}, "
                f"curator_frequency={curator_frequency}, "
                f"trigger_steps={curator_trigger_steps})"
            )
        else:
            print(
                f"  Chunk by curator_batch_size={curator_batch_size} "
                f"({len(curator_entries)} learnable entries)"
            )
            num_chunks = (
                len(curator_entries) + curator_batch_size - 1
            ) // curator_batch_size
            print(
                f"  Running Curator {num_chunks} times "
                f"(each with up to {curator_batch_size} samples)"
            )
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * curator_batch_size
                end_idx = min(start_idx + curator_batch_size, len(curator_entries))
                chunk_entries = curator_entries[start_idx:end_idx]

                combined_reflection = "\n\n---\n\n".join(
                    f"[Sample {start_idx + i + 1}] {entry['reflection']}"
                    for i, entry in enumerate(chunk_entries)
                    if entry.get("reflection") != "(empty)"
                )
                if not combined_reflection:
                    combined_reflection = "(empty)"

                combined_context = "\n\n---\n\n".join(
                    f"[Sample {start_idx + i + 1}] {entry['context']}"
                    for i, entry in enumerate(chunk_entries)
                    if entry.get("context")
                )
                if not combined_context:
                    combined_context = "(not available)"

                combined_reasoning_trace = "\n\n---\n\n".join(
                    f"[Sample {start_idx + i + 1}] {entry['reasoning_trace']}"
                    for i, entry in enumerate(chunk_entries)
                    if entry.get("reasoning_trace")
                )
                if not combined_reasoning_trace:
                    combined_reasoning_trace = "(not available)"

                print(
                    f"\n--- Curator chunk {chunk_idx + 1}/{num_chunks} "
                    f"(samples {start_idx + 1}-{end_idx}, "
                    f"curator step {curator_step_for_batch}) ---"
                )
                _run_one_curator_call(
                    combined_reflection,
                    combined_context,
                    combined_reasoning_trace,
                    curator_step_for_batch,
                    f"{step_id_prefix}_s_{curator_step_for_batch}_chunk_{chunk_idx + 1}",
                    len(chunk_entries),
                )
            print(
                f"\n  Playbook updated after {num_chunks} Curator calls: "
                f"{count_tokens(self.playbook)} tokens"
            )

        # ================================================================
        # PHASE 3: Parallel Post-Curator Generation
        # ================================================================
        print(f"\n{'='*40}")
        print(f"PHASE 3: Parallel Post-Curator Generation ({len(batch)} samples)")
        print(f"{'='*40}")
        
        post_curate_results = [None] * len(batch)
        
        def _post_curate_generate(result_dict):
            """Thread-safe post-curator generation for a single sample."""
            question = result_dict["question"]
            context = result_dict["context"]
            target = result_dict["target"]
            sid = result_dict["step_id"]

            gen_response, _, call_info = self.generator.generate(
                question=question,
                playbook=self.playbook,
                context=context,
                reflection="(empty)",
                use_json_mode=use_json_mode,
                call_id=f"{sid}_post_curate",
                log_dir=log_dir,
            )

            final_answer = extract_answer(gen_response)
            post_correct = data_processor.answer_is_correct(final_answer, target)
            infra_failure_source = self._get_infra_failure_source(
                call_info,
                post_correct,
                data_processor,
            )
            if infra_failure_source:
                return final_answer, False, "infra"
            return final_answer, post_correct, "none"

        # Pre-fill samples that should skip post-curator generation.
        for i, result in enumerate(sample_results):
            if result.get("pre_train_answer") == INCORRECT_DUE_TO_API_ERROR:
                post_curate_results[i] = (
                    INCORRECT_DUE_TO_API_ERROR,
                    False,
                    "infra",
                )
                continue
            if result.get("failure_type") == "infra":
                post_curate_results[i] = (
                    result.get("final_answer", INCORRECT_DUE_TO_API_ERROR),
                    False,
                    "infra",
                )
                continue
            if result.get("skip_post_curate_generation", False):
                post_curate_results[i] = (
                    result.get("final_answer", INCORRECT_DUE_TO_API_ERROR),
                    bool(result.get("is_correct", False)),
                    result.get("failure_type", "none"),
                )

        phase3_api_error_occurred = False
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_idx = {}
            for i, result in enumerate(sample_results):
                if post_curate_results[i] is not None:
                    continue
                future = executor.submit(_post_curate_generate, result)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    post_curate_results[idx] = future.result()
                except Exception as e:
                    if self._should_continue_on_error(e, config_params):
                        print(
                            f"  API/LLM ERROR in post-curate sample {idx + 1}: {e} - "
                            "marking as incorrect, rolling back playbook, continuing"
                        )
                        post_curate_results[idx] = (
                            INCORRECT_DUE_TO_API_ERROR,
                            False,
                            "infra",
                        )
                        phase3_api_error_occurred = True
                    else:
                        print(f"  ERROR in post-curate sample {idx + 1}: {e}")
                        raise

        if phase3_api_error_occurred:
            self.playbook = playbook_before_phase2
            self.next_global_id = next_global_id_before_phase2
            print(
                "  Playbook and next_global_id rolled back to pre-Phase2 "
                "state due to API error(s)"
            )

        print(f"Phase 3 complete: All {len(batch)} post-curate generations done")

        # ================================================================
        # Assemble final results
        # ================================================================
        batch_output = []
        for i, result in enumerate(sample_results):
            post_result = post_curate_results[i]
            if post_result is None:
                post_result = (
                    result.get("final_answer", INCORRECT_DUE_TO_API_ERROR),
                    bool(result.get("is_correct", False)),
                    result.get("failure_type", "none"),
                )
            post_answer, post_correct, post_failure_type = post_result
            tracking_dict = result["tracking_dict"]
            post_train_result = {
                "final_answer": post_answer,
                "is_correct": post_correct,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook),
            }
            if post_failure_type == "infra":
                post_train_result["failure_type"] = "infra"
            tracking_dict["post_train_result"] = post_train_result
            batch_output.append(
                (
                    result["pre_train_answer"],
                    post_answer,
                    tracking_dict,
                )
            )

        return batch_output
    
    def _offline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str,
        batch_size: int
    ) -> Dict[str, Any]:
        """
        Run offline training
        
        Args:
            train_samples: List of training samples
            val_samples: List of validation samples
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            batch_size: Batch size for training
        Returns:
            Dictionary with training results
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']
        test_workers = config_params['test_workers']
        use_json_mode = config_params['use_json_mode']
        curator_frequency = config_params['curator_frequency']
        
        # Initialize tracking
        results = []
        pre_train_post_train_results = []
        error_logs = []
        best_accuracy = 0.0
        self.best_playbook = self.playbook

        num_batches = (len(train_samples) + batch_size - 1) // batch_size
        
        print(f"Total epochs: {num_epochs}")
        print(f"Train samples per epoch: {len(train_samples)}")
        print(f"Gen batch size: {batch_size} | Curator batch size: {config_params.get('curator_batch_size', 10)}")
        print(f"Batches per epoch: {num_batches}")
        print(f"Val samples: {len(val_samples)}")
        print(
            "Curator frequency policy: run curator on batch boundaries "
            f"that include step % {curator_frequency} == 0"
        )
        print(f"Evaluation frequency: every {eval_steps} steps\n")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            epoch_pre_train_correct_flags = []
            epoch_post_train_correct_flags = []
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(train_samples))
                batch = train_samples[batch_start:batch_end]
                
                print(f"\n{'='*60}")
                print(f"BATCH {batch_idx + 1}/{num_batches} "
                      f"(samples {batch_start + 1}-{batch_end}/{len(train_samples)})")
                print(f"{'='*60}")
                
                # Run parallel gen+reflect -> sync -> curator -> parallel post-curate
                batch_results = self._train_batch(
                    batch=batch,
                    data_processor=data_processor,
                    batch_step_start=batch_start + 1,
                    epoch=epoch,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(train_samples),
                    step_id_prefix=f"train_e_{epoch}"
                )
                
                # Collect per-sample results from this batch
                for i, (pre_train_answer, post_train_answer, tracking_dict) in enumerate(batch_results):
                    step = batch_start + i + 1
                    target = batch[i].get("target", "")
                    
                    epoch_answers_pre_train.append(pre_train_answer)
                    epoch_targets_pre_train.append(target)
                    epoch_answers_post_train.append(post_train_answer)
                    epoch_targets_post_train.append(target)
                    epoch_pre_train_correct_flags.append(
                        bool(
                            tracking_dict.get("pre_train_result", {}).get(
                                "is_correct", False
                            )
                        )
                    )
                    epoch_post_train_correct_flags.append(
                        bool(
                            tracking_dict.get("post_train_result", {}).get(
                                "is_correct", False
                            )
                        )
                    )
                    
                    pre_train_post_train_result = {
                        "epoch": epoch,
                        "step": step,
                        "target": target,
                        **tracking_dict
                    }
                    pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook after each batch
                last_step = batch_end
                if last_step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"epoch_{epoch}_step_{last_step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
                
                # Periodic evaluation (check at batch boundary)
                if last_step % eval_steps == 0:
                    print(f"\n{'='*40}")
                    print(f"EVALUATION AT EPOCH {epoch}, STEP {last_step}")
                    print(f"{'='*40}")
                    
                    # Compute training accuracies
                    if self._use_single_pass_accuracy(data_processor):
                        pre_train_accuracy = self._accuracy_from_correctness_flags(
                            epoch_pre_train_correct_flags
                        )
                        post_train_accuracy = self._accuracy_from_correctness_flags(
                            epoch_post_train_correct_flags
                        )
                    else:
                        pre_train_accuracy = data_processor.evaluate_accuracy(
                            epoch_answers_pre_train, epoch_targets_pre_train
                        )
                        post_train_accuracy = data_processor.evaluate_accuracy(
                            epoch_answers_post_train, epoch_targets_post_train
                        )
                    
                    # Validation evaluation
                    val_results = {}
                    val_error_log = {}
                    if val_samples:
                        val_results, val_error_log = evaluate_test_set(
                            data_processor, self.generator, self.playbook, 
                            val_samples, self.max_tokens, log_dir, 
                            max_workers=test_workers, use_json_mode=use_json_mode
                        )
                    
                    result = {
                        "epoch": epoch,
                        "step": last_step,
                        "train_result": {
                            "pre_train_accuracy": pre_train_accuracy,
                            "post_train_accuracy": post_train_accuracy
                        },
                        "val_result": val_results,
                        "playbook_num_tokens": count_tokens(self.playbook),
                        "playbook_length": len(self.playbook),
                        "playbook_stats": get_playbook_stats(self.playbook)
                    }
                    results.append(result)
                    error_logs.append({
                        "epoch": epoch,
                        "step": last_step,
                        "val_results": val_results,
                        "error_log": val_error_log
                    })

                    # Track best playbook
                    if val_results:
                        acc = val_results["accuracy"]
                        if acc > best_accuracy:
                            best_accuracy = acc
                            self.best_playbook = self.playbook
                            print(f"🎉 New best accuracy: {best_accuracy:.3f}")
                    
                    # Save results
                    results_path = os.path.join(save_path, "train_results.json")
                    with open(results_path, "w") as f:
                        json.dump({
                            "best_accuracy": best_accuracy,
                            "results": results,
                        }, f, indent=2)
                    
                    error_logs_path = os.path.join(save_path, "val_results.json")
                    with open(error_logs_path, "w") as f:
                        json.dump(error_logs, f, indent=2)
            
            # End of epoch - save final playbook
            epoch_playbook_path = os.path.join(
                playbook_dir, f"epoch_{epoch}_final_playbook.txt"
            )
            with open(epoch_playbook_path, "w") as f:
                f.write(self.playbook)

        # Save training results
        results_path = os.path.join(save_path, "train_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_accuracy": best_accuracy,
                "results": results,
            }, f, indent=2)
        
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)
        
        # Save best playbook
        best_playbook_path = os.path.join(save_path, f"best_playbook.txt")
        with open(best_playbook_path, "w") as f:
            f.write(self.best_playbook)
        
        print(f"\n{'='*60}")
        print(f"OFFLINE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_accuracy:.3f}")
        print(f"{'='*60}\n")

        return {"best_validation_accuracy": best_accuracy}

    
    def test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run testing with the playbook (backward compatibility wrapper).
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to be used for generator
            config: Configuration dictionary
            
        Returns:
            Dictionary with test results
        """
        # Temporarily set the playbook
        old_playbook = self.playbook
        self.playbook = playbook
        
        # Use the run method
        results = self.run(
            mode='eval_only',
            test_samples=test_samples,
            data_processor=data_processor,
            config=config
        )
        
        # Restore old playbook
        self.playbook = old_playbook
        
        # Return in the old format for backward compatibility
        return {
            "test_results": results['test_results'],
            "error_log": results.get('test_error_log', {}),
            "playbook": playbook
        }
    
    def _online_train_and_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """
        Run online training and testing
        
        Args:
            test_samples: List of samples to train and test on
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results, test results, and final playbook
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        
        # Validate configuration
        if num_epochs != 1:
            raise ValueError(f"online_train_and_test requires num_epochs=1, got {num_epochs}")
        
        # Extract additional parameters
        curator_frequency = config_params['curator_frequency']
        batch_size = config_params['batch_size']
        task_name = config_params['task_name']
        save_steps = config_params['save_steps']
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        online_eval_frequency = config.get('online_eval_frequency', 100)  # Get from config
        
        # Initialize tracking
        train_results = []
        pre_train_post_train_results = []
        
        # Test tracking - accumulate across all windows
        correct_count_sample_based = 0
        correct_count = 0
        total_count = 0
        all_test_errors = []
        window_test_results = []
        print(f"Total samples: {len(test_samples)}")
        print(f"Window size: {online_eval_frequency}")
        print(f"Batch size: {batch_size}")
        print(f"Number of windows: {(len(test_samples) + online_eval_frequency - 1) // online_eval_frequency}")
        print(f"Curator frequency: every {curator_frequency} steps")
        
        # Split samples into windows
        num_windows = (len(test_samples) + online_eval_frequency - 1) // online_eval_frequency
        
        epoch = 1  # Always 1 epoch
        global_step = 0
        
        for window_idx in range(num_windows):
            start_idx = window_idx * online_eval_frequency
            end_idx = min((window_idx + 1) * online_eval_frequency, len(test_samples))
            window_samples = test_samples[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"WINDOW {window_idx + 1}/{num_windows}")
            print(f"Samples {start_idx} to {end_idx - 1}")
            print(f"{'='*60}")
            
            # =================================================================
            # STEP 1: TEST on window with current playbook (before training)
            # =================================================================
            print(f"\n--- Testing window {window_idx + 1} with current playbook ---")
            
            # Use evaluate_test_set for parallel evaluation
            window_test_results_dict, window_test_error_log = evaluate_test_set(
                data_processor,
                self.generator,
                self.playbook,
                window_samples,
                self.max_tokens,
                log_dir,
                max_workers=test_workers,
                use_json_mode=use_json_mode
            )
            
            # Extract results
            window_accuracy = window_test_results_dict['accuracy']
            window_correct = window_test_results_dict['correct']
            window_total = window_test_results_dict['total']
            correct_count_sample_based += window_correct
            correct_count += window_accuracy * window_total
            total_count += window_total
            
            # Add errors with window and global index information
            for error in window_test_error_log['errors']:
                all_test_errors.append({
                    "window": window_idx + 1,
                    "global_index": start_idx + error['index'],
                    "prediction": error['prediction'],
                    "ground_truth": error['ground_truth']
                })
            
            window_test_results.append({
                "window": window_idx + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window_accuracy": window_accuracy,
                "window_correct": window_correct,
                "window_total": window_total
            })
            
            # Calculate cumulative test accuracy so far
            cumulative_test_accuracy = correct_count / total_count
            
            print(f"Window {window_idx + 1} test accuracy: {window_accuracy:.3f}")
            print(f"Cumulative test accuracy so far: {cumulative_test_accuracy:.3f} "
                  f"({total_count} samples)")
            
            # =================================================================
            # STEP 2: TRAIN on window (parallel batched)
            # =================================================================
            print(f"\n--- Training on window {window_idx + 1} (batch_size={batch_size}) ---")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            epoch_pre_train_correct_flags = []
            epoch_post_train_correct_flags = []
            
            num_window_batches = (len(window_samples) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_window_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(window_samples))
                batch = window_samples[batch_start:batch_end]
                
                # Global step corresponds to the start of this batch
                batch_global_step_start = global_step + 1
                
                print(f"\n--- Window {window_idx + 1}, Batch {batch_idx + 1}/{num_window_batches} "
                      f"(samples {batch_start + 1}-{batch_end}, "
                      f"global steps {batch_global_step_start}-{batch_global_step_start + len(batch) - 1}) ---")
                
                # Run parallel gen+reflect -> sync -> curator -> parallel post-curate
                batch_results = self._train_batch(
                    batch=batch,
                    data_processor=data_processor,
                    batch_step_start=batch_global_step_start,
                    epoch=epoch,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(test_samples),
                    step_id_prefix="online_train"
                )
                
                # Collect per-sample results from this batch
                for i, (pre_train_answer, post_train_answer, tracking_dict) in enumerate(batch_results):
                    global_step += 1
                    target = batch[i].get("target", "")
                    
                    epoch_answers_pre_train.append(pre_train_answer)
                    epoch_targets_pre_train.append(target)
                    epoch_answers_post_train.append(post_train_answer)
                    epoch_targets_post_train.append(target)
                    epoch_pre_train_correct_flags.append(
                        bool(
                            tracking_dict.get("pre_train_result", {}).get(
                                "is_correct", False
                            )
                        )
                    )
                    epoch_post_train_correct_flags.append(
                        bool(
                            tracking_dict.get("post_train_result", {}).get(
                                "is_correct", False
                            )
                        )
                    )
                    
                    pre_train_post_train_result = {
                        "window": window_idx + 1,
                        "global_step": global_step,
                        "target": target,
                        **tracking_dict
                    }
                    pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook after batch
                if global_step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"step_{global_step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
            
            # End of window - compute training accuracies for this window
            if self._use_single_pass_accuracy(data_processor):
                pre_train_accuracy = self._accuracy_from_correctness_flags(
                    epoch_pre_train_correct_flags
                )
                post_train_accuracy = self._accuracy_from_correctness_flags(
                    epoch_post_train_correct_flags
                )
            else:
                pre_train_accuracy = data_processor.evaluate_accuracy(
                    epoch_answers_pre_train, epoch_targets_pre_train
                )
                post_train_accuracy = data_processor.evaluate_accuracy(
                    epoch_answers_post_train, epoch_targets_post_train
                )
            
            window_train_result = {
                "window": window_idx + 1,
                "global_step": global_step,
                "train_result": {
                    "pre_train_accuracy": pre_train_accuracy,
                    "post_train_accuracy": post_train_accuracy
                },
                "cumulative_test_accuracy": cumulative_test_accuracy,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook),
                "playbook_stats": get_playbook_stats(self.playbook)
            }
            train_results.append(window_train_result)
            
            print(f"\nWindow {window_idx + 1} training complete:")
            print(f"  Pre-train accuracy: {pre_train_accuracy:.3f}")
            print(f"  Post-train accuracy: {post_train_accuracy:.3f}")
            
            # Save window playbook
            window_playbook_path = os.path.join(
                playbook_dir, f"window_{window_idx + 1}_final_playbook.txt"
            )
            with open(window_playbook_path, "w") as f:
                f.write(self.playbook)
        
        # All windows complete
        print(f"\n{'='*60}")
        print(f"ONLINE TRAIN AND TEST COMPLETE")
        print(f"{'='*60}")
        
        # Calculate final cumulative test accuracy
        assert total_count == len(test_samples)
        final_test_accuracy = correct_count / total_count
        
        test_results = {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
            "window_results": window_test_results
        }
        
        test_error_log = {
            "accuracy": final_test_accuracy,
            "errors": all_test_errors
        }

        # Save test results
        test_results_path = os.path.join(save_path, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_accuracy": final_test_accuracy,
                "test_results": test_results,
                "test_error_log": test_error_log
            }, f, indent=2)
        
        # Save training results (per window)
        train_results_path = os.path.join(save_path, "train_results.json")
        with open(train_results_path, "w") as f:
            json.dump({"train_results": train_results}, f, indent=2)
        
        # Save pre-train/post-train results
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)
        
        print(f"\n{'='*60}")
        print(f"ONLINE TRAINING AND TESTING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Test Accuracy: {final_test_accuracy:.3f}")
        print(f"{'='*60}\n")
        
        return {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
        }