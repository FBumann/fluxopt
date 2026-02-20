"""Sparse summation helpers for linopy expressions.

Avoids dense broadcast when coefficient arrays are highly sparse
(e.g., conversion constraints where each converter only touches 2-3 flows
out of hundreds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import linopy


def sparse_weighted_sum(
    var: linopy.Variable,
    coeffs: xr.DataArray,
    sum_dim: str,
    group_dim: str,
) -> linopy.LinearExpression:
    """Compute ``(var * coeffs).sum(sum_dim)`` using only non-zero pairs.

    When *coeffs* is sparse along ``(group_dim, sum_dim)`` — e.g. a
    ``(converter, flow)`` matrix where each converter references only a few
    flows — the naive dense broadcast creates a huge intermediate linopy
    expression.  This function finds the non-zero ``(group, sum_dim)`` pairs
    and uses ``groupby().sum()`` to aggregate, avoiding the dense broadcast.

    Args:
        var: linopy Variable or LinearExpression with *sum_dim*.
        coeffs: DataArray with at least ``(group_dim, sum_dim)``.
            Extra dims (e.g. ``eq_idx``, ``time``) are preserved.
        sum_dim: Dimension to sum over (e.g. ``'flow'``).
        group_dim: Dimension to group by (e.g. ``'converter'``).

    Returns:
        linopy expression with *sum_dim* removed, *group_dim* present.
    """
    coeffs_values = coeffs.values
    group_ids = list(coeffs.coords[group_dim].values)
    sum_ids = list(coeffs.coords[sum_dim].values)

    group_axis = coeffs.dims.index(group_dim)
    sum_axis = coeffs.dims.index(sum_dim)

    # Collapse extra axes to find any non-zero (group, sum) pair
    reduce_axes = tuple(i for i in range(coeffs_values.ndim) if i not in (group_axis, sum_axis))
    nonzero_2d = np.any(coeffs_values != 0, axis=reduce_axes) if reduce_axes else coeffs_values != 0

    # Ensure (group, sum) axis order
    if group_axis > sum_axis:
        nonzero_2d = nonzero_2d.T
    group_idx, sum_idx = np.nonzero(nonzero_2d)

    if len(group_idx) == 0:
        # All zeros — fall back to dense (cheap when empty)
        return (var * coeffs).sum(sum_dim)

    # Ensure all groups appear in the result (groups with all-zero coefficients
    # still need a zero entry so groupby produces an output for them)
    represented = set(group_idx.tolist())
    for g in range(len(group_ids)):
        if g not in represented:
            group_idx = np.append(group_idx, g)
            sum_idx = np.append(sum_idx, 0)  # coeff is 0 for absent groups

    pair_sum_ids = [sum_ids[s] for s in sum_idx]
    pair_group_ids = [group_ids[g] for g in group_idx]

    # Transpose coeffs so (group, sum, ...) are the leading axes,
    # ensuring NumPy advanced indexing places the pair axis first.
    remaining_dims = [d for d in coeffs.dims if d not in (group_dim, sum_dim)]
    ordered = coeffs.transpose(group_dim, sum_dim, *remaining_dims)
    ordered_values = ordered.values

    # Extract per-pair coefficients — advanced indexing on axes 0, 1
    pair_coeffs_data = ordered_values[group_idx, sum_idx]  # (pair, *remaining)

    remaining_coords = {d: coeffs.coords[d] for d in remaining_dims if d in coeffs.coords}
    pair_coeffs = xr.DataArray(
        pair_coeffs_data,
        dims=['pair', *remaining_dims],
        coords=remaining_coords,
    )

    # Select var for active pairs, multiply, group-sum
    selected = var.sel({sum_dim: xr.DataArray(pair_sum_ids, dims=['pair'])})
    weighted = selected * pair_coeffs

    mapping = xr.DataArray(pair_group_ids, dims=['pair'], name=group_dim)
    result = weighted.groupby(mapping).sum()

    # Restore original group order (groupby sorts alphabetically)
    result = result.sel({group_dim: group_ids})

    # Vectorized sel() leaves sum_dim as a non-dim coord — drop it
    return cast('linopy.LinearExpression', result.drop_vars(sum_dim, errors='ignore'))
