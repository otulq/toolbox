from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Optional
import time
import random


def execute_batch(
    function: Callable[..., Any],
    *,
    inputs: List[Dict[str, Any]],
    check_function: Callable[[Any], bool],
    retries: Optional[int] = 3,
    batch_retries: Optional[int] = 1,
    max_workers: Optional[int] = None,
    retry_delay: Optional[float] = 0.1,   # base delay per retry (None ⇒ 0 s)
    jitter: Optional[float] = 0.1,        # max ± jitter (None ⇒ 0 s)
) -> List[Dict[str, Any]]:
    """
    Execute *function*(**input_dict) in parallel with retry logic at both
    the call level and the batch level.

    Returns a list of result-dictionaries in the same order as *inputs*.

    Args:
        function: The function to execute in parallel.
        inputs: A list of dictionaries, each containing the input parameters
                for a single function call.
        check_function: A function that checks if the output of the function
                        is valid.
        retries: The number of times to retry the function call if it fails
                 (None → 0).
        batch_retries: The number of times to retry the entire batch
                       if any items fail (None → 0).
        max_workers: The maximum number of workers to use for the
                     parallel execution.
        retry_delay: Base delay in seconds added after each failed attempt.
                     Actual delay is (attempt_number * retry_delay) ± jitter.
                     Pass None to disable deterministic back-off.
        jitter: Maximum uniform random jitter in seconds added to each delay.
                Pass None to disable jitter.

    Returns:
        A list of dictionaries with ordering preserved by input `id` and the
        following keys:
            id: unique identifier for the input
            input: the original input dict
            result: return value of *function* or None if failed
            success: bool
            function_latency: seconds for the (first successful) call
            total_latency: seconds spent across all retries
            retries: number of call-level retries performed
            error_count: number of exceptions raised during retries
            failure_count: number of times check_function returned False
            batch_round: number of batch-level retries attempted (0-based)
    """
    retries = retries or 0
    batch_retries = batch_retries or 0
    retry_delay = retry_delay or 0.0
    jitter = jitter or 0.0

    # ───────────── helper that runs a single item with in-call retries ──────────
    def _attempt(idx: int, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        error_count = 0
        failure_count = 0
        start_total = time.perf_counter()

        for attempt in range(1, retries + 1):
            call_start = time.perf_counter()
            try:
                out = function(**input_dict)
                call_dur = time.perf_counter() - call_start
                if check_function(out):
                    return {
                        "id": idx,
                        "input": input_dict,
                        "result": out,
                        "success": True,
                        "function_latency": call_dur,
                        "total_latency": time.perf_counter() - start_total,
                        "retries": attempt - 1,
                        "error_count": error_count,
                        "failure_count": failure_count,
                    }
                failure_count += 1
            except Exception:
                error_count += 1

            # linear back-off with jitter (skip after final attempt)
            if attempt < retries and (retry_delay or jitter):
                base = attempt * retry_delay
                delta = random.uniform(-jitter, jitter)
                time.sleep(max(0.0, base + delta))

        # exhausted retries
        return {
            "id": idx,
            "input": input_dict,
            "result": None,
            "success": False,
            "function_latency": None,
            "total_latency": time.perf_counter() - start_total,
            "retries": retries,
            "error_count": error_count,
            "failure_count": failure_count,
        }

    # attach synthetic ids in caller order
    indexed_inputs = [{"id": i, "input": inp} for i, inp in enumerate(inputs)]
    remaining: Dict[int, Dict[str, Any]] = {item["id"]: item for item in indexed_inputs}
    results: Dict[int, Dict[str, Any]] = {}

    # ───────────── batch-retry loop ─────────────
    for batch_round in range(batch_retries + 1):
        if not remaining:
            break

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_id = {
                pool.submit(_attempt, item_id, item["input"]): item_id
                for item_id, item in remaining.items()
            }

            for fut in as_completed(future_to_id):
                item_id = future_to_id[fut]
                res = fut.result()
                res["batch_round"] = batch_round
                results[item_id] = res

        # re-queue only the ones that failed
        remaining = {
            item_id: indexed_inputs[item_id]
            for item_id, res in results.items()
            if not res["success"]
        }

    # ───────────── return in original call order ─────────────
    return [results[item["id"]] for item in indexed_inputs]
