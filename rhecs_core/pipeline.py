import asyncio
from typing import Optional, Dict, Any
from uuid import uuid4

from rhecs_core.extraction.extractor import extract_vietnamese_claims, ClaimList
from rhecs_core.verification.sandbox_manager import execute_sandbox_code
from rhecs_core.verification.root_planner import generate_verification_script
from rhecs_core.verification.nli_judge import judge_evidence, VerificationResult, NLIStatus, ErrorCategory
from rhecs_core.restoration.evidence_compiler import compile_evidence
from rhecs_core.restoration.rewriter import fix_claim
from rhecs_core.restoration.replacer import surgical_replace
from rhecs_core.logger import TrajectoryLogger
from rhecs_core.runtime import (
    ClaimRuntimeState,
    RequestRuntimeState,
    RuntimeErrorInfo,
    RuntimeErrorType,
    RuntimeTransition,
    transition_claim_state,
    transition_request_state,
)

MAX_RETRIES = 2

class RHECSPipeline:
    """
    Ties together the entire RAG Hallucination Correction System.
    Orchestrates Extractor -> Verifier -> Restorer autonomously.
    """
    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id

    @staticmethod
    def _to_dict(model_obj: Any) -> dict:
        if hasattr(model_obj, "model_dump"):
            return model_obj.model_dump()
        if hasattr(model_obj, "dict"):
            return model_obj.dict()
        return dict(model_obj)

    @staticmethod
    def _classify_error_type(message: str) -> RuntimeErrorType:
        msg = (message or "").lower()

        if any(token in msg for token in ["timed out", "timeout"]):
            return RuntimeErrorType.TIMEOUT_ERROR
        if any(
            token in msg
            for token in [
                "429",
                "503",
                "resource_exhausted",
                "rate limit",
                "unavailable",
                "model is not found",
                "api key",
                "provider",
            ]
        ):
            return RuntimeErrorType.PROVIDER_ERROR
        if any(token in msg for token in ["policy", "forbidden", "violation"]):
            return RuntimeErrorType.POLICY_ERROR
        if any(token in msg for token in ["json", "schema", "validation", "parse"]):
            return RuntimeErrorType.DATA_ERROR
        if any(
            token in msg
            for token in [
                "traceback",
                "syntaxerror",
                "nameerror",
                "importerror",
                "execution",
                "sandbox",
            ]
        ):
            return RuntimeErrorType.EXECUTION_ERROR
        return RuntimeErrorType.UNKNOWN_ERROR

    def _record_request_transition(
        self,
        runtime: Dict[str, Any],
        target: RequestRuntimeState,
        stage: str,
        reason: Optional[str] = None,
    ) -> None:
        current: RequestRuntimeState = runtime["request_state"]
        if current == target:
            return

        validated = transition_request_state(current, target)
        runtime["request_state"] = validated
        runtime["request_transitions"].append(
            RuntimeTransition(
                entity_type="request",
                entity_id=runtime["request_id"],
                from_state=current.value,
                to_state=validated.value,
                stage=stage,
                reason=reason,
            )
        )
        logger: Optional[TrajectoryLogger] = runtime.get("trajectory_logger")
        if logger:
            logger.log_transition(runtime["request_transitions"][-1])

    def _record_claim_transition(
        self,
        runtime: Dict[str, Any],
        claim_id: str,
        target: ClaimRuntimeState,
        stage: str,
        reason: Optional[str] = None,
    ) -> None:
        current: ClaimRuntimeState = runtime["claim_states"][claim_id]
        if current == target:
            return

        validated = transition_claim_state(current, target)
        runtime["claim_states"][claim_id] = validated
        runtime["claim_transitions"].append(
            RuntimeTransition(
                entity_type="claim",
                entity_id=claim_id,
                from_state=current.value,
                to_state=validated.value,
                stage=stage,
                reason=reason,
            )
        )
        logger: Optional[TrajectoryLogger] = runtime.get("trajectory_logger")
        if logger:
            logger.log_transition(runtime["claim_transitions"][-1])

    def _append_runtime_error(
        self,
        runtime: Dict[str, Any],
        stage: str,
        message: str,
        claim_id: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        runtime["runtime_errors"].append(
            RuntimeErrorInfo(
                error_type=self._classify_error_type(message),
                stage=stage,
                message=message,
                claim_id=claim_id,
                retryable=retryable,
            )
        )
        logger: Optional[TrajectoryLogger] = runtime.get("trajectory_logger")
        if logger:
            logger.log_error(runtime["runtime_errors"][-1])

    def _serialize_runtime(self, runtime: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "request_id": runtime["request_id"],
            "request_state": runtime["request_state"].value,
            "request_transitions": [
                {
                    "entity_type": t.entity_type,
                    "entity_id": t.entity_id,
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "stage": t.stage,
                    "reason": t.reason,
                    "timestamp": t.timestamp,
                }
                for t in runtime["request_transitions"]
            ],
            "claim_states": {
                claim_id: state.value for claim_id, state in runtime["claim_states"].items()
            },
            "claim_transitions": [
                {
                    "entity_type": t.entity_type,
                    "entity_id": t.entity_id,
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "stage": t.stage,
                    "reason": t.reason,
                    "timestamp": t.timestamp,
                }
                for t in runtime["claim_transitions"]
            ],
        }

    def _serialize_runtime_errors(self, runtime: Dict[str, Any]) -> list[dict]:
        return [
            {
                "error_type": err.error_type.value,
                "stage": err.stage,
                "message": err.message,
                "claim_id": err.claim_id,
                "retryable": err.retryable,
                "timestamp": err.timestamp,
            }
            for err in runtime["runtime_errors"]
        ]

    def _build_output(
        self,
        original_text: str,
        restored_text: str,
        claim_count: int,
        verdicts: list[VerificationResult],
        applied_fixes: list[dict],
        runtime: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "original_text": original_text,
            "restored_text": restored_text,
            "metrics": {
                "claims_extracted": claim_count,
                "faults_found": len([v for v in verdicts if v.status == NLIStatus.CONTRADICTED]),
                "patches_applied": len(applied_fixes),
            },
            "audit_trail": {
                "verdicts": [self._to_dict(v) for v in verdicts],
                "patches": applied_fixes,
            },
            "runtime_status": self._serialize_runtime(runtime),
            "runtime_errors": self._serialize_runtime_errors(runtime),
            "trajectory_path": runtime.get("trajectory_path"),
        }

    async def _verify_claim_worker(
        self,
        original_sentence: str,
        claim: dict,
        claim_id: str,
        runtime: Dict[str, Any],
    ) -> VerificationResult:
        """
        Runs the async Verification module to challenge a specific extracted claim.
        """
        error_trace = None
        for attempt in range(MAX_RETRIES):
            try:
                script_code = await asyncio.to_thread(generate_verification_script, claim, error_trace)
                self._record_claim_transition(
                    runtime,
                    claim_id,
                    ClaimRuntimeState.PLAN_GENERATED,
                    stage="planner",
                    reason=f"attempt_{attempt + 1}",
                )
            except Exception as exc:
                error_trace = str(exc)
                self._append_runtime_error(
                    runtime,
                    stage="planner",
                    message=error_trace,
                    claim_id=claim_id,
                    retryable=(attempt < MAX_RETRIES - 1),
                )
                if attempt == MAX_RETRIES - 1:
                    self._record_claim_transition(
                        runtime,
                        claim_id,
                        ClaimRuntimeState.CLAIM_FAILED,
                        stage="planner",
                        reason="planner_failed_max_retries",
                    )
                    return VerificationResult(
                        status=NLIStatus.NOT_MENTIONED,
                        reasoning=(
                            f"Planner failed repeatedly across {MAX_RETRIES} attempts. "
                            f"Final traceback: {error_trace}"
                        ),
                        error_category=ErrorCategory.UNVERIFIABLE,
                        fault_span=None,
                    )
                continue

            try:
                sandbox_result = await asyncio.to_thread(execute_sandbox_code, script_code, self.tenant_id)
            except Exception as exc:
                error_trace = str(exc)
                self._append_runtime_error(
                    runtime,
                    stage="sandbox",
                    message=error_trace,
                    claim_id=claim_id,
                    retryable=(attempt < MAX_RETRIES - 1),
                )
                if attempt == MAX_RETRIES - 1:
                    self._record_claim_transition(
                        runtime,
                        claim_id,
                        ClaimRuntimeState.CLAIM_FAILED,
                        stage="sandbox",
                        reason="sandbox_exception_max_retries",
                    )
                    return VerificationResult(
                        status=NLIStatus.NOT_MENTIONED,
                        reasoning=(
                            f"Sandbox execution failed repeatedly across {MAX_RETRIES} attempts. "
                            f"Final traceback: {error_trace}"
                        ),
                        error_category=ErrorCategory.UNVERIFIABLE,
                        fault_span=None,
                    )
                continue

            if sandbox_result.get("success"):
                self._record_claim_transition(
                    runtime,
                    claim_id,
                    ClaimRuntimeState.SANDBOX_EXECUTED,
                    stage="sandbox",
                )
                evidence_data = sandbox_result.get("output", {})
                evidence_list = evidence_data.get("evidence", evidence_data)

                try:
                    verdict = await judge_evidence(original_sentence, claim, evidence_list)
                    self._record_claim_transition(
                        runtime,
                        claim_id,
                        ClaimRuntimeState.VERDICT_ASSIGNED,
                        stage="judge",
                    )
                    return verdict
                except Exception as exc:
                    error_trace = str(exc)
                    self._append_runtime_error(
                        runtime,
                        stage="judge",
                        message=error_trace,
                        claim_id=claim_id,
                        retryable=(attempt < MAX_RETRIES - 1),
                    )
                    if attempt == MAX_RETRIES - 1:
                        self._record_claim_transition(
                            runtime,
                            claim_id,
                            ClaimRuntimeState.CLAIM_FAILED,
                            stage="judge",
                            reason="judge_failed_max_retries",
                        )
                        return VerificationResult(
                            status=NLIStatus.NOT_MENTIONED,
                            reasoning=(
                                f"Judge failed repeatedly across {MAX_RETRIES} attempts. "
                                f"Final traceback: {error_trace}"
                            ),
                            error_category=ErrorCategory.UNVERIFIABLE,
                            fault_span=None,
                        )
            else:
                error_trace = str(sandbox_result.get("error", "unknown sandbox error"))
                self._append_runtime_error(
                    runtime,
                    stage="sandbox",
                    message=error_trace,
                    claim_id=claim_id,
                    retryable=(attempt < MAX_RETRIES - 1),
                )

                if attempt == MAX_RETRIES - 1:
                    self._record_claim_transition(
                        runtime,
                        claim_id,
                        ClaimRuntimeState.CLAIM_FAILED,
                        stage="sandbox",
                        reason="sandbox_failed_max_retries",
                    )
                    return VerificationResult(
                        status=NLIStatus.NOT_MENTIONED,
                        reasoning=(
                            f"Sandbox crashed repeatedly across {MAX_RETRIES} attempts. "
                            f"Final traceback: {error_trace}"
                        ),
                        error_category=ErrorCategory.UNVERIFIABLE,
                        fault_span=None,
                    )
                
        return VerificationResult(
            status=NLIStatus.NOT_MENTIONED, 
            reasoning=f"Sandbox crashed repeatedly across {MAX_RETRIES} attempts. Final traceback: {error_trace}",
            error_category=ErrorCategory.UNVERIFIABLE,
            fault_span=None
        )

    async def _restore_claim_worker(
        self,
        original_sentence: str,
        claim: dict,
        verification_verdict: VerificationResult,
        claim_id: str,
        runtime: Dict[str, Any],
    ) -> Optional[dict]:
        """
        If a claim is contradicted, pulls correct evidence and proposes an isolated patch.
        """
        if verification_verdict.status != NLIStatus.CONTRADICTED or not verification_verdict.fault_span:
            return None  # No restoration needed
            
        fault_span = verification_verdict.fault_span
        error_category_str = verification_verdict.error_category.value if verification_verdict.error_category else "Unknown"
        
        evidence = await asyncio.to_thread(compile_evidence, claim)
        if not evidence:
            self._append_runtime_error(
                runtime,
                stage="restore_evidence",
                message="No evidence returned for contradicted claim during restoration.",
                claim_id=claim_id,
                retryable=False,
            )
            self._record_claim_transition(
                runtime,
                claim_id,
                ClaimRuntimeState.CLAIM_FAILED,
                stage="restore_evidence",
                reason="missing_evidence",
            )
            return None
            
        try:
            repair_instruction = await fix_claim(original_sentence, fault_span, error_category_str, evidence)
            self._record_claim_transition(
                runtime,
                claim_id,
                ClaimRuntimeState.PATCH_GENERATED,
                stage="rewriter",
            )

            return {
                "claim_id": claim_id,
                "fault_span": fault_span,
                "corrected_span": repair_instruction.corrected_span,
                "analysis": repair_instruction.analysis,
            }
        except Exception as exc:
            self._append_runtime_error(
                runtime,
                stage="rewriter",
                message=str(exc),
                claim_id=claim_id,
                retryable=False,
            )
            self._record_claim_transition(
                runtime,
                claim_id,
                ClaimRuntimeState.CLAIM_FAILED,
                stage="rewriter",
                reason="patch_generation_failed",
            )
            return None

    async def process_document(self, raw_unverified_text: str) -> Dict:
        """
        The Main Engine Entrypoint for RHECS.
        Processes a dirty document and returns the cleaned text alongside auditing metrics.
        """
        request_id = uuid4().hex
        runtime: Dict[str, Any] = {
            "request_id": request_id,
            "request_state": RequestRuntimeState.RECEIVED,
            "request_transitions": [
                RuntimeTransition(
                    entity_type="request",
                    entity_id=request_id,
                    from_state=None,
                    to_state=RequestRuntimeState.RECEIVED.value,
                    stage="request",
                    reason="request_initialized",
                )
            ],
            "claim_states": {},
            "claim_transitions": [],
            "runtime_errors": [],
            "trajectory_logger": None,
            "trajectory_path": None,
        }

        trajectory_logger = TrajectoryLogger(request_id=request_id)
        runtime["trajectory_logger"] = trajectory_logger
        runtime["trajectory_path"] = trajectory_logger.file_path
        trajectory_logger.log_metadata(request_id=request_id, tenant_id=self.tenant_id)
        trajectory_logger.log_transition(runtime["request_transitions"][0])

        restored_draft = raw_unverified_text
        verdicts: list[VerificationResult] = []
        applied_fixes: list[dict] = []
        claim_payloads: list[tuple[str, dict]] = []

        try:
            print("[Engine] 1. Extracting Claims Phase...")
            claim_list: ClaimList = await extract_vietnamese_claims(raw_unverified_text)
            self._record_request_transition(
                runtime,
                RequestRuntimeState.CLAIMS_EXTRACTED,
                stage="extract",
            )

            for idx, claim in enumerate(claim_list.claims, start=1):
                claim_id = f"claim_{idx}"
                runtime["claim_states"][claim_id] = ClaimRuntimeState.CLAIM_CREATED
                runtime["claim_transitions"].append(
                    RuntimeTransition(
                        entity_type="claim",
                        entity_id=claim_id,
                        from_state=None,
                        to_state=ClaimRuntimeState.CLAIM_CREATED.value,
                        stage="extract",
                        reason="claim_initialized",
                    )
                )
                trajectory_logger.log_transition(runtime["claim_transitions"][-1])
                claim_payloads.append((claim_id, claim.dict()))

            print(f"[Engine] Found {len(claim_payloads)} claims. Dispatching Verification workers...")

            self._record_request_transition(
                runtime,
                RequestRuntimeState.VERIFICATION_IN_PROGRESS,
                stage="verify",
            )

            verify_tasks = [
                self._verify_claim_worker(raw_unverified_text, claim_data, claim_id, runtime)
                for claim_id, claim_data in claim_payloads
            ]

            verdicts = await asyncio.gather(*verify_tasks)
            self._record_request_transition(
                runtime,
                RequestRuntimeState.VERIFICATION_DONE,
                stage="verify",
            )

            print("[Engine] 2. Verifications collected. Identifying Faults & Restoring...")
            self._record_request_transition(
                runtime,
                RequestRuntimeState.RESTORATION_IN_PROGRESS,
                stage="restore",
            )

            restore_tasks = [
                self._restore_claim_worker(
                    raw_unverified_text,
                    claim_data,
                    verdict,
                    claim_id,
                    runtime,
                )
                for (claim_id, claim_data), verdict in zip(claim_payloads, verdicts)
            ]

            repair_results = await asyncio.gather(*restore_tasks)

            print("[Engine] 3. Sync Application of Isolated Patches...")
            for patch in repair_results:
                if patch is not None:
                    claim_id = patch["claim_id"]
                    try:
                        restored_draft = surgical_replace(
                            restored_draft,
                            patch["fault_span"],
                            patch["corrected_span"],
                        )
                        applied_fixes.append(patch)
                        if runtime["claim_states"][claim_id] == ClaimRuntimeState.PATCH_GENERATED:
                            self._record_claim_transition(
                                runtime,
                                claim_id,
                                ClaimRuntimeState.PATCH_APPLIED,
                                stage="replace",
                            )
                    except ValueError as exc:
                        self._append_runtime_error(
                            runtime,
                            stage="replace",
                            message=str(exc),
                            claim_id=claim_id,
                            retryable=False,
                        )
                        if runtime["claim_states"][claim_id] != ClaimRuntimeState.CLAIM_FAILED:
                            self._record_claim_transition(
                                runtime,
                                claim_id,
                                ClaimRuntimeState.CLAIM_FAILED,
                                stage="replace",
                                reason="surgical_replace_failed",
                            )

            self._record_request_transition(
                runtime,
                RequestRuntimeState.RESTORATION_DONE,
                stage="restore",
            )

            any_claim_failed = any(
                state == ClaimRuntimeState.CLAIM_FAILED for state in runtime["claim_states"].values()
            )
            has_runtime_errors = len(runtime["runtime_errors"]) > 0

            final_target = (
                RequestRuntimeState.DEGRADED
                if any_claim_failed or has_runtime_errors
                else RequestRuntimeState.FINALIZED
            )
            self._record_request_transition(runtime, final_target, stage="finalize")

            trajectory_logger.log_summary(
                request_state=runtime["request_state"].value,
                claims_extracted=len(claim_payloads),
                faults_found=len([v for v in verdicts if v.status == NLIStatus.CONTRADICTED]),
                patches_applied=len(applied_fixes),
                runtime_errors=len(runtime["runtime_errors"]),
            )

            return self._build_output(
                original_text=raw_unverified_text,
                restored_text=restored_draft,
                claim_count=len(claim_payloads),
                verdicts=verdicts,
                applied_fixes=applied_fixes,
                runtime=runtime,
            )

        except Exception as exc:
            self._append_runtime_error(
                runtime,
                stage="pipeline",
                message=str(exc),
                claim_id=None,
                retryable=False,
            )
            if runtime["request_state"] not in {
                RequestRuntimeState.FAILED,
                RequestRuntimeState.FINALIZED,
                RequestRuntimeState.DEGRADED,
            }:
                try:
                    self._record_request_transition(
                        runtime,
                        RequestRuntimeState.FAILED,
                        stage="pipeline",
                        reason="unhandled_exception",
                    )
                except Exception:
                    runtime["request_state"] = RequestRuntimeState.FAILED

            trajectory_logger.log_summary(
                request_state=runtime["request_state"].value,
                claims_extracted=len(claim_payloads),
                faults_found=len([v for v in verdicts if v.status == NLIStatus.CONTRADICTED]),
                patches_applied=len(applied_fixes),
                runtime_errors=len(runtime["runtime_errors"]),
            )

            return self._build_output(
                original_text=raw_unverified_text,
                restored_text=restored_draft,
                claim_count=len(claim_payloads),
                verdicts=verdicts,
                applied_fixes=applied_fixes,
                runtime=runtime,
            )
