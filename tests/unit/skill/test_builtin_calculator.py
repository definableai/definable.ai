"""Unit tests for the Calculator built-in skill.

Tests cover the pure evaluation logic of the calculator,
including basic arithmetic, functions, and error handling.
No API calls are made.
"""

import pytest

from definable.skill.builtin.calculator import Calculator, _evaluate_expression


@pytest.mark.unit
class TestEvaluateExpression:
  """Tests for the _evaluate_expression helper (pure logic)."""

  def test_addition(self):
    """Addition of two integers."""
    result = _evaluate_expression("2 + 3")
    assert result == "5"

  def test_subtraction(self):
    """Subtraction of two integers."""
    result = _evaluate_expression("10 - 4")
    assert result == "6"

  def test_multiplication(self):
    """Multiplication of two integers."""
    result = _evaluate_expression("7 * 6")
    assert result == "42"

  def test_division(self):
    """Division of two numbers."""
    result = _evaluate_expression("15 / 3")
    assert result == "5"

  def test_division_float_result(self):
    """Division yielding a float."""
    result = _evaluate_expression("7 / 2")
    assert result == "3.5"

  def test_division_by_zero_handled(self):
    """Division by zero returns an error message."""
    result = _evaluate_expression("1 / 0")
    assert "Error" in result

  def test_power(self):
    """Exponentiation using ** syntax."""
    result = _evaluate_expression("2 ** 10")
    assert result == "1024"

  def test_caret_converted_to_power(self):
    """Caret ^ is converted to ** (power)."""
    result = _evaluate_expression("2 ^ 3")
    assert result == "8"

  def test_floor_division(self):
    """Floor division //."""
    result = _evaluate_expression("7 // 2")
    assert result == "3"

  def test_modulo(self):
    """Modulo operator %."""
    result = _evaluate_expression("10 % 3")
    assert result == "1"

  def test_parentheses_order_of_operations(self):
    """Parentheses enforce order of operations."""
    result = _evaluate_expression("(2 + 3) * 4")
    assert result == "20"

  def test_negative_number(self):
    """Unary negation."""
    result = _evaluate_expression("-5 + 3")
    assert result == "-2"

  def test_sqrt_function(self):
    """sqrt() is a safe function."""
    result = _evaluate_expression("sqrt(144)")
    assert result == "12"

  def test_pi_constant(self):
    """pi is accessible as a constant."""
    result = _evaluate_expression("pi")
    assert result.startswith("3.14159")

  def test_factorial_function(self):
    """factorial() returns correct result."""
    result = _evaluate_expression("factorial(5)")
    assert result == "120"

  def test_invalid_expression_returns_error(self):
    """A syntactically invalid expression returns an error."""
    result = _evaluate_expression("2 +* 3")
    assert "Error" in result

  def test_unsafe_function_blocked(self):
    """Calling a non-whitelisted function returns an error."""
    result = _evaluate_expression("__import__('os')")
    assert "Error" in result

  def test_percentage_calculation(self):
    """Percentage calculation: 15% of 84.50."""
    result = _evaluate_expression("15 / 100 * 84.50")
    assert float(result) == pytest.approx(12.675)


@pytest.mark.unit
class TestCalculatorSkill:
  """Tests for the Calculator skill class."""

  def test_calculator_name(self):
    """Calculator skill is named 'calculator'."""
    calc = Calculator()
    assert calc.name == "calculator"

  def test_calculator_has_instructions(self):
    """Calculator provides non-empty instructions."""
    calc = Calculator()
    assert len(calc.instructions) > 0
    assert "calculator" in calc.instructions.lower()

  def test_calculator_has_calculate_tool(self):
    """Calculator exposes exactly one tool named 'calculate'."""
    calc = Calculator()
    assert len(calc.tools) == 1
    assert calc.tools[0].name == "calculate"

  def test_calculate_tool_is_callable(self):
    """The calculate tool has a callable entrypoint."""
    calc = Calculator()
    fn = calc.tools[0]
    result = fn.entrypoint(expression="3 + 4")
    assert result == "7"
