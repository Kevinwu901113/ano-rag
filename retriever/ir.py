from typing import Any, Dict, List


class ConstraintIR:
    def __init__(
        self,
        variables: List[str],
        constraints: List[Dict[str, Any]],
        ask: Dict[str, Any],
    ) -> None:
        self.variables = variables
        self.constraints = constraints
        self.ask = ask
