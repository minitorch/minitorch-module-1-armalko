from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals = list(vals)

    vals[arg] += epsilon
    f_plus = f(*vals)

    vals[arg] -= 2 * epsilon
    f_minus = f(*vals)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: 'Variable') -> Iterable['Variable']:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable (output of the computation graph).

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    order: List['Variable'] = []

    def dfs(v: 'Variable'):
        if v.unique_id not in visited and not v.is_constant():
            visited.add(v.unique_id)
            for parent in v.parents:
                dfs(parent)
            order.append(v)

    dfs(variable)

    for i, var in enumerate(order):
        print(f"Topological Order {i}: Variable ID = {var.unique_id}, Data = {var.data}")

    return list(reversed(order))



def backpropagate(variable: 'Variable', deriv: Any) -> None:
    sorted_vars = topological_sort(variable)
    gradients = {var.unique_id: 0.0 for var in sorted_vars}
    gradients[variable.unique_id] = deriv
    print(f"JOPA {sorted_vars}")
    for var in sorted_vars:
        current_deriv = gradients[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(current_deriv)

        for parent, parent_deriv in var.chain_rule(current_deriv):
            if parent.unique_id in gradients:
                gradients[parent.unique_id] += parent_deriv
            else:
                gradients[parent.unique_id] = parent_deriv



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
