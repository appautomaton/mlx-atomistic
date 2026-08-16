"""Complex64 rank and orthonormalization policy for periodic Davidson solves."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


def _all_finite(values: mx.array) -> bool:
    """Check a device array through one scalar synchronization."""

    return bool(mx.all(mx.isfinite(values)))


@dataclass(frozen=True)
class _RankResult:
    """Rank-filtered vectors and their row transform from the input stack."""

    values: mx.array
    transform: mx.array
    deflated_count: int


@dataclass(frozen=True)
class _Complex64RankPolicy:
    """One deterministic rank and orthogonality policy for Davidson state."""

    relative_tolerance: float = 32.0 * float(np.finfo(np.float32).eps)

    def orthonormality_tolerance(self, vector_count: int) -> float:
        return 64.0 * float(np.finfo(np.float32).eps) * max(vector_count, 1)

    def guard_tolerance(self, vector_count: int) -> float:
        return 8.0 * float(np.finfo(np.float32).eps) * max(vector_count, 1)

    def overlap_error(self, values: mx.array) -> float:
        count = int(values.shape[0])
        overlap = values @ mx.conjugate(mx.transpose(values))
        # This is O(subspace**2); no active plane-wave axis crosses to NumPy.
        overlap_np = np.asarray(overlap, dtype=np.complex64)
        if not np.all(np.isfinite(overlap_np)):
            msg = "Davidson overlap matrix must be finite"
            raise ValueError(msg)
        return float(np.max(np.abs(overlap_np - np.eye(count))))

    def validate(self, values: mx.array, *, required_count: int) -> float:
        stack = mx.array(values).astype(mx.complex64)
        if len(stack.shape) != 2 or int(stack.shape[0]) < required_count:
            msg = "Davidson state has insufficient rank"
            raise ValueError(msg)
        if not _all_finite(stack):
            msg = "Davidson state must be finite"
            raise ValueError(msg)
        error = self.overlap_error(stack)
        if error > self.orthonormality_tolerance(int(stack.shape[0])):
            msg = "Davidson state violates the complex64 rank policy"
            raise ValueError(msg)
        return error

    def _try_batched_choleskyqr2(
        self,
        stack: mx.array,
        *,
        candidate_norms: mx.array,
        locked_count: int,
        required_count: int,
        limit: int,
    ) -> _RankResult | None:
        """Admit a well-resolved row block through CholeskyQR2."""

        input_count = int(stack.shape[0])
        candidate_count = min(input_count - locked_count, limit - locked_count)
        retained_count = locked_count + candidate_count
        if (
            candidate_count <= 0
            or candidate_count != input_count - locked_count
            or retained_count < required_count
        ):
            return None

        identity = mx.eye(input_count, dtype=mx.float32).astype(mx.complex64)
        locked_values = stack[:locked_count]
        locked_transforms = identity[:locked_count]
        candidate_values = stack[locked_count:retained_count]
        candidate_transforms = identity[locked_count:retained_count]
        for _ in range(2):
            overlaps = mx.conjugate(locked_values) @ mx.transpose(candidate_values)
            candidate_values = candidate_values - mx.transpose(overlaps) @ locked_values
            candidate_transforms = candidate_transforms - mx.transpose(overlaps) @ locked_transforms

        norms = mx.array(candidate_norms).astype(mx.float32)
        if norms.shape != (input_count - locked_count,):
            msg = "Davidson candidate norms must match the unlocked row count"
            raise ValueError(msg)
        norms = norms[:candidate_count]
        norms_valid = mx.all(mx.isfinite(norms)) & mx.all(norms > 0.0)
        scale = norms[:, None]
        candidate_values = candidate_values / scale
        candidate_transforms = candidate_transforms / scale

        def cholesky_step(
            values: mx.array,
            *,
            additional_validity: mx.array | None = None,
        ) -> tuple[mx.array, np.ndarray] | None:
            gram = values @ mx.conjugate(mx.transpose(values))
            if additional_validity is not None:
                # Materialize the norm guard with the Gram matrix. The common
                # fast path therefore crosses to the CPU once, not once for
                # norms and again for the small Cholesky factorization.
                mx.eval(gram, additional_validity)
                if not bool(additional_validity):
                    return None
            gram_np = np.asarray(gram, dtype=np.complex64).astype(np.complex128)
            if not np.all(np.isfinite(gram_np)):
                return None
            gram_np = 0.5 * (gram_np + np.conjugate(gram_np.T))
            try:
                lower = np.linalg.cholesky(gram_np)
                solve = np.linalg.solve(
                    lower,
                    np.eye(candidate_count, dtype=np.complex128),
                )
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(solve)):
                return None
            return mx.array(solve.astype(np.complex64)), lower

        first = cholesky_step(
            candidate_values,
            additional_validity=norms_valid,
        )
        if first is None:
            return None
        first_solve, first_lower = first
        pivot_floor = 8.0 * float(np.sqrt(np.finfo(np.float32).eps))
        first_pivots = np.real(np.diag(first_lower))
        first_condition = np.linalg.cond(first_lower)
        if (
            np.any(first_pivots <= pivot_floor)
            or not np.isfinite(first_condition)
            or first_condition > 256.0
        ):
            return None
        candidate_values = first_solve @ candidate_values
        candidate_transforms = first_solve @ candidate_transforms

        second = cholesky_step(candidate_values)
        if second is None:
            return None
        second_solve, _second_lower = second
        candidate_values = second_solve @ candidate_values
        candidate_transforms = second_solve @ candidate_transforms

        result_values = mx.concatenate(
            [locked_values, candidate_values],
            axis=0,
        )
        result_transform = mx.concatenate(
            [locked_transforms, candidate_transforms],
            axis=0,
        )
        if self.overlap_error(result_values) > self.orthonormality_tolerance(retained_count):
            return None
        return _RankResult(
            values=result_values,
            transform=result_transform,
            deflated_count=input_count - retained_count,
        )

    def _sequential_mgs2(
        self,
        stack: mx.array,
        *,
        original_norms: np.ndarray,
        locked_count: int,
        required_count: int,
        limit: int,
    ) -> _RankResult:
        """Recover a rare unstable block projection with ordered MGS2."""

        input_count = int(stack.shape[0])
        identity = mx.eye(input_count, dtype=mx.float32).astype(mx.complex64)
        accepted = [stack[index] for index in range(locked_count)]
        transforms = [identity[index] for index in range(locked_count)]
        for index in range(locked_count, input_count):
            if len(accepted) >= limit:
                break
            original_norm = float(original_norms[index])
            if original_norm == 0.0:
                continue
            vector = stack[index]
            transform = identity[index]
            for _ in range(2):
                for accepted_vector, accepted_transform in zip(
                    accepted,
                    transforms,
                    strict=True,
                ):
                    overlap = mx.sum(mx.conjugate(accepted_vector) * vector)
                    vector = vector - overlap * accepted_vector
                    transform = transform - overlap * accepted_transform
            norm = float(mx.sqrt(mx.real(mx.sum(mx.conjugate(vector) * vector))))
            if not np.isfinite(norm):
                msg = "Davidson rank norm must be finite"
                raise ValueError(msg)
            if norm <= self.relative_tolerance * original_norm:
                continue
            accepted.append(vector / norm)
            transforms.append(transform / norm)

        retained_count = len(accepted)
        if retained_count < required_count:
            msg = (
                "Davidson rank policy retained "
                f"{retained_count} vectors but {required_count} are required"
            )
            raise ValueError(msg)
        result_values = mx.stack(accepted, axis=0)
        result_transform = mx.stack(transforms, axis=0)
        self.validate(result_values, required_count=required_count)
        return _RankResult(
            values=result_values,
            transform=result_transform,
            deflated_count=input_count - retained_count,
        )

    def orthonormalize(
        self,
        values: mx.array,
        *,
        locked_count: int = 0,
        required_count: int = 0,
        max_count: int | None = None,
    ) -> _RankResult:
        """Twice reorthogonalize in complex64 and deterministically deflate."""

        stack = mx.array(values).astype(mx.complex64)
        if len(stack.shape) != 2 or int(stack.shape[0]) == 0:
            msg = "Davidson rank input must be a non-empty matrix"
            raise ValueError(msg)
        input_count = int(stack.shape[0])
        if not 0 <= locked_count <= input_count:
            msg = "locked Davidson rank must lie within the input stack"
            raise ValueError(msg)
        limit = input_count if max_count is None else int(max_count)
        if limit < locked_count or limit <= 0:
            msg = "Davidson rank limit cannot discard locked vectors"
            raise ValueError(msg)
        if required_count < 0 or required_count > limit:
            msg = "required Davidson rank exceeds the rank limit"
            raise ValueError(msg)

        candidate_stack = stack[locked_count:]
        candidate_norms = mx.sqrt(
            mx.real(mx.sum(mx.conjugate(candidate_stack) * candidate_stack, axis=1))
        )

        batched = self._try_batched_choleskyqr2(
            stack,
            candidate_norms=candidate_norms,
            locked_count=locked_count,
            required_count=required_count,
            limit=limit,
        )
        if batched is not None:
            return batched

        candidate_norms_np = np.asarray(candidate_norms, dtype=np.float32)
        if not np.all(np.isfinite(candidate_norms_np)):
            msg = "Davidson rank input must be finite"
            raise ValueError(msg)
        original_norms_np = np.ones((input_count,), dtype=np.float32)
        original_norms_np[locked_count:] = candidate_norms_np

        accepted_values = stack[:locked_count]

        identity = mx.eye(input_count, dtype=mx.float32).astype(mx.complex64)
        accepted_transforms = identity[:locked_count]

        deflated = 0
        for index in range(locked_count, input_count):
            if int(accepted_values.shape[0]) >= limit:
                deflated += input_count - index
                break
            original_norm = float(original_norms_np[index])
            if original_norm == 0.0:
                deflated += 1
                continue
            vector = stack[index : index + 1]
            transform = identity[index : index + 1]
            for _ in range(2):
                if int(accepted_values.shape[0]) == 0:
                    break
                overlaps = mx.conjugate(accepted_values) @ mx.transpose(vector)
                vector = vector - mx.transpose(overlaps) @ accepted_values
                transform = transform - mx.transpose(overlaps) @ accepted_transforms
            norm = float(mx.sqrt(mx.real(mx.sum(mx.conjugate(vector) * vector))))
            if not np.isfinite(norm):
                msg = "Davidson rank norm must be finite"
                raise ValueError(msg)
            if norm <= self.relative_tolerance * original_norm:
                deflated += 1
                continue
            accepted_values = mx.concatenate(
                [accepted_values, vector / norm],
                axis=0,
            )
            accepted_transforms = mx.concatenate(
                [accepted_transforms, transform / norm],
                axis=0,
            )

        retained_count = int(accepted_values.shape[0])
        if retained_count < required_count:
            return self._sequential_mgs2(
                stack,
                original_norms=original_norms_np,
                locked_count=locked_count,
                required_count=required_count,
                limit=limit,
            )

        result_transform = accepted_transforms
        # Preserve the incrementally reorthogonalized values. Collapsing the
        # accumulated transform into one complex64 matmul can reintroduce a
        # deflated component through cancellation for nearly dependent rows.
        result_values = accepted_values
        if not _all_finite(result_values):
            msg = "Davidson state must be finite"
            raise ValueError(msg)
        overlap_error = self.overlap_error(result_values)
        if overlap_error > self.orthonormality_tolerance(retained_count):
            return self._sequential_mgs2(
                stack,
                original_norms=original_norms_np,
                locked_count=locked_count,
                required_count=required_count,
                limit=limit,
            )
        return _RankResult(
            values=result_values,
            transform=result_transform,
            deflated_count=deflated,
        )


_DAVIDSON_RANK_POLICY = _Complex64RankPolicy()
