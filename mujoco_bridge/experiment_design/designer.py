from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Mapping


def design_experiment_matrix(plan: Mapping[str, Any], limit: int | None = None) -> List[Dict[str, Any]]:
    variables = list(plan.get("variables") or [])
    if not variables:
        return [{"run_id": 1}]

    names = [str(variable.get("name")) for variable in variables]
    value_lists = [list(variable.get("values") or []) for variable in variables]
    rows = []
    for idx, values in enumerate(product(*value_lists), start=1):
        row = {"run_id": idx}
        row.update({name: value for name, value in zip(names, values)})
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows
