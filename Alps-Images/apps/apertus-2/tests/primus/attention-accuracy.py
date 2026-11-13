"""Version-pinned AITER BF16 gradient accuracy policy; no torch dependency here."""
import math

# AITER v0.1.22.post1 op_tests/test_mha.py, lines362-368:
# https://github.com/ROCm/aiter/blob/b4d9154d125e09efbe098d986e40fea3549c1244/op_tests/test_mha.py
# Its gradient max-error bound is max(10 * BF16-control max error, 0.01).
# Also require BOTH implementations' relative L2 error below one BF16 epsilon,
# which prevents the upstream max-only bound from masking widespread drift.
UPSTREAM_REVISION='b4d9154d125e09efbe098d986e40fea3549c1244'
BF16_EPSILON=2**-7


def validate_gradient_error(actual_max_abs, control_max_abs, actual_relative_l2, control_relative_l2):
    values=(actual_max_abs,control_max_abs,actual_relative_l2,control_relative_l2)
    if not all(math.isfinite(value) and value>=0 for value in values):
        raise AssertionError('Nonfinite or negative attention error metric')
    bound=max(10*control_max_abs,0.01)
    if control_relative_l2>BF16_EPSILON:
        raise AssertionError('BF16 math control exceeds precision-relative L2 bound')
    if actual_relative_l2>BF16_EPSILON:
        raise AssertionError('AITER gradient exceeds precision-relative L2 bound')
    if actual_max_abs>bound:
        raise AssertionError('AITER gradient exceeds pinned upstream BF16-control max-error bound')
    return dict(upstream_revision=UPSTREAM_REVISION,max_abs_bound=bound,
                relative_l2_bound=BF16_EPSILON,actual_max_abs=actual_max_abs,
                control_max_abs=control_max_abs,actual_relative_l2=actual_relative_l2,
                control_relative_l2=control_relative_l2)
