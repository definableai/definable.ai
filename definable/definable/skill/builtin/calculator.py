"""Calculator skill — safe mathematical expression evaluation.

Gives the agent the ability to perform precise mathematical calculations
using Python's ast module for safe evaluation (no arbitrary code execution).

Example:
    from definable.skill.builtin import Calculator

    agent = Agent(
        model=model,
        skills=[Calculator()],
    )
    output = agent.run("What is 15% tip on a $84.50 bill?")
"""

import ast
import math
import operator
from typing import Any, Dict

from definable.skill.base import Skill
from definable.tool.decorator import tool


# Safe math operators and functions
_SAFE_OPERATORS: Dict[type, Any] = {
  ast.Add: operator.add,
  ast.Sub: operator.sub,
  ast.Mult: operator.mul,
  ast.Div: operator.truediv,
  ast.FloorDiv: operator.floordiv,
  ast.Mod: operator.mod,
  ast.Pow: operator.pow,
  ast.USub: operator.neg,
  ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
  "abs": abs,
  "round": round,
  "min": min,
  "max": max,
  "sum": sum,
  "int": int,
  "float": float,
  # Math module functions
  "sqrt": math.sqrt,
  "ceil": math.ceil,
  "floor": math.floor,
  "log": math.log,
  "log2": math.log2,
  "log10": math.log10,
  "sin": math.sin,
  "cos": math.cos,
  "tan": math.tan,
  "pi": math.pi,
  "e": math.e,
  "pow": math.pow,
  "factorial": math.factorial,
  "gcd": math.gcd,
}


def _safe_eval(node: ast.AST) -> Any:
  """Recursively evaluate an AST node using only safe operations."""
  if isinstance(node, ast.Expression):
    return _safe_eval(node.body)
  elif isinstance(node, ast.Constant):
    if isinstance(node.value, (int, float, complex)):
      return node.value
    raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
  elif isinstance(node, ast.BinOp):
    op_type = type(node.op)
    if op_type not in _SAFE_OPERATORS:
      raise ValueError(f"Unsupported operator: {op_type.__name__}")
    left = _safe_eval(node.left)
    right = _safe_eval(node.right)
    return _SAFE_OPERATORS[op_type](left, right)
  elif isinstance(node, ast.UnaryOp):
    unary_op_type = type(node.op)
    if unary_op_type not in _SAFE_OPERATORS:
      raise ValueError(f"Unsupported unary operator: {unary_op_type.__name__}")
    operand = _safe_eval(node.operand)
    return _SAFE_OPERATORS[unary_op_type](operand)
  elif isinstance(node, ast.Call):
    if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
      args = [_safe_eval(arg) for arg in node.args]
      func = _SAFE_FUNCTIONS[node.func.id]
      if callable(func):
        return func(*args)
      return func  # constants like pi, e
    raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
  elif isinstance(node, ast.Name):
    if node.id in _SAFE_FUNCTIONS:
      val = _SAFE_FUNCTIONS[node.id]
      if not callable(val):
        return val  # constants
    raise ValueError(f"Unsupported name: {node.id}")
  elif isinstance(node, ast.Tuple):
    return tuple(_safe_eval(el) for el in node.elts)
  elif isinstance(node, ast.List):
    return [_safe_eval(el) for el in node.elts]
  else:
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _evaluate_expression(expression: str) -> str:
  """Safely evaluate a mathematical expression."""
  # Clean up common natural-language patterns
  expr = expression.strip()
  expr = expr.replace("^", "**")  # caret to power
  expr = expr.replace("×", "*").replace("÷", "/")

  try:
    tree = ast.parse(expr, mode="eval")
    result = _safe_eval(tree)
    # Format nicely
    if isinstance(result, float):
      if result == int(result) and abs(result) < 1e15:
        return str(int(result))
      return f"{result:.10g}"
    return str(result)
  except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
    return f"Error: {e}"


@tool
def calculate(expression: str) -> str:
  """Evaluate a mathematical expression and return the result.

  Supports: +, -, *, /, //, %, ** (power), parentheses,
  and functions: sqrt, ceil, floor, log, log2, log10,
  sin, cos, tan, abs, round, min, max, sum, factorial, gcd.
  Constants: pi, e.

  Args:
    expression: A mathematical expression to evaluate.
        Examples: "2 + 3 * 4", "sqrt(144)", "15/100 * 84.50",
        "round(3.14159, 2)", "factorial(10)"

  Returns:
    The computed result as a string, or an error message.
  """
  return _evaluate_expression(expression)


class Calculator(Skill):
  """Skill for precise mathematical calculations.

  Adds a ``calculate`` tool and instructions for when and how the agent
  should use it for accurate arithmetic instead of mental math.

  Example:
    agent = Agent(model=model, skills=[Calculator()])
    output = agent.run("What's 18% tip on $127.43?")
  """

  name = "calculator"
  instructions = (
    "You have access to a calculator tool for precise mathematical computation. "
    "Use it whenever you need to perform arithmetic, percentages, conversions, "
    "or any numerical calculation — do not attempt mental math for anything "
    "beyond trivial operations. Always show the expression you are evaluating."
  )

  def __init__(self):
    super().__init__()

  @property
  def tools(self) -> list:
    return [calculate]
