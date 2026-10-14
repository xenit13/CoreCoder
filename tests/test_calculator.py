"""Unit tests for the safe arithmetic calculator engine."""

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

import corecoder.calculator as calculator
from corecoder.calculator import CalculatorError, calculate

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # literals & whitespace
        ("42", 42.0),
        ("  3.5 ", 3.5),
        ("1e3", 1000.0),
        (".5", 0.5),
        # precedence
        ("1+2*3", 7.0),
        ("(1+2)*3", 9.0),
        ("2+3*4-6/2", 11.0),
        # left associativity
        ("10-3-2", 5.0),
        ("100/10/2", 5.0),
        ("17//5", 3.0),
        ("17%5", 2.0),
        # power
        ("2**10", 1024.0),
        ("2**3**2", 512.0),
        ("-2**2", -4.0),
        ("(-2)**2", 4.0),
        # unary
        ("-5", -5.0),
        ("+5", 5.0),
        ("--5", 5.0),
        ("3*-2", -6.0),
        # float results
        ("1/4", 0.25),
        ("7//2", 3.0),
    ],
)
def test_calculate_returns_expected_float(expression, expected):
    result = calculate(expression)
    assert result == expected
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "1+",
        "(1+2",
        "1+2)",
        "2**",
        "1 2",
        "1/0",
        "1//0",
        "5%0",
        "a+1",
        "1;$",
        "__import__('os')",
    ],
)
def test_bad_expressions_raise_calculator_error(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "1e308**2",
        "10**1000",
    ],
)
def test_overflow_raises_calculator_error(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "(" * 2000 + "1" + ")" * 2000,
        "-" * 5000 + "5",
    ],
)
def test_deeply_nested_expression_raises_calculator_error(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "1+1; import os",
    ],
)
def test_code_injection_attempts_are_rejected(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_calculator_error_is_a_value_error():
    assert issubclass(CalculatorError, ValueError)


# ---------------------------------------------------------------------------
# Task 2: command-line interface
# ---------------------------------------------------------------------------


def test_cli_evaluates_expression_arguments(capsys):
    assert calculator.main(["1+2*3"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "7\n"
    assert captured.err == ""


def test_cli_prints_non_integral_result_as_repr(capsys):
    assert calculator.main(["1/4"]) == 0
    assert capsys.readouterr().out == "0.25\n"


def test_cli_joins_multiple_arguments(capsys):
    assert calculator.main(["1", "+", "2", "*", "3"]) == 0
    assert capsys.readouterr().out == "7\n"


def test_cli_defaults_argv_to_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["corecoder.calculator", "2+2"])
    assert calculator.main() == 0
    assert capsys.readouterr().out == "4\n"


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_cli_help_prints_usage_to_stdout(flag, capsys):
    assert calculator.main([flag]) == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out
    assert captured.err == ""


def test_cli_reads_expression_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("2**8\n"))
    assert calculator.main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == "256\n"
    assert captured.err == ""


def test_cli_empty_stdin_prints_usage_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert calculator.main([]) == 2
    captured = capsys.readouterr()
    assert "usage" in captured.err
    assert captured.out == ""


def test_cli_division_by_zero_reports_error(capsys):
    assert calculator.main(["1/0"]) == 1
    captured = capsys.readouterr()
    assert "division by zero" in captured.err
    assert captured.out == ""


def test_cli_bad_input_error_line_starts_with_error(capsys):
    assert calculator.main(["a+1"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert captured.out == ""


@pytest.mark.skipif(
    importlib.util.find_spec("corecoder") is None,
    reason="corecoder cannot be imported by the current interpreter",
)
def test_cli_end_to_end_subprocess():
    completed = subprocess.run(
        [sys.executable, "-m", "corecoder.calculator", "6*7"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "42"
