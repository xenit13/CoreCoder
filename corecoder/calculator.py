"""A tiny, dependency-free arithmetic calculator.

Supports ``+ - * / // % **``, unary ``+``/``-``, parentheses and
int/float/scientific literals. Expressions are parsed by hand -- ``eval`` is
never used -- so untrusted input cannot execute code.
"""

from __future__ import annotations

import sys

__all__ = ["CalculatorError", "calculate", "main"]

USAGE = "usage: python -m corecoder.calculator [expression]"

NUMERIC_CHARS = set("0123456789.eE+-")


class CalculatorError(ValueError):
    """Raised when an expression cannot be tokenized or parsed."""


def tokenize(expression: str) -> list[str]:
    """Split *expression* into number and operator tokens.

    Numbers are returned as their source text; operators are single
    characters. Raises CalculatorError on unknown characters.
    """
    tokens: list[str] = []
    i, n = 0, len(expression)
    while i < n:
        ch = expression[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            start = i
            while i < n and expression[i] in NUMERIC_CHARS:
                # 'e'/'E' only continues a number when followed by an
                # optional sign and a digit; a bare '+'/'-' always ends the
                # literal so that binary operators stay separate tokens.
                current = expression[i]
                if current in "+-":
                    break
                if current in "eE":
                    j = i + 1
                    if j < n and expression[j] in "+-":
                        j += 1
                    if j >= n or not expression[j].isdigit():
                        break
                    i = j + 1
                    continue
                i += 1
            text = expression[start:i]
            try:
                float(text)
            except ValueError:
                raise CalculatorError(f"invalid number: {text!r}") from None
            tokens.append(text)
            continue
        if ch in "+-*/%()":
            tokens.append(ch)
            i += 1
            continue
        raise CalculatorError(f"unexpected character: {ch!r}")
    return tokens


class _Parser:
    """Recursive-descent parser over a list of single-character tokens."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> float:
        """Parse the whole token list, which must be fully consumed."""
        if not self.tokens:
            raise CalculatorError("empty expression")
        value = self._expression()
        if self.pos != len(self.tokens):
            raise CalculatorError(f"unexpected token: {self.tokens[self.pos]!r}")
        return float(value)

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _peek_op(self) -> str | None:
        """Return the operator at the cursor; ``//`` and ``**`` join up."""
        token = self._peek()
        if token is None:
            return None
        nxt = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
        if token == "*" and nxt == "*":
            return "**"
        if token == "/" and nxt == "/":
            return "//"
        return token

    def _advance(self, width: int = 1) -> None:
        self.pos += width

    def _expression(self) -> float:
        """``+``/``-``, left-associative."""
        value = self._term()
        while True:
            op = self._peek_op()
            if op not in ("+", "-"):
                return value
            self._advance()
            rhs = self._term()
            value = value + rhs if op == "+" else value - rhs

    def _term(self) -> float:
        """``*``/``/``/``//``/``%``, left-associative."""
        value = self._unary()
        while True:
            op = self._peek_op()
            if op not in ("*", "/", "//", "%"):
                return value
            self._advance(len(op))
            rhs = self._unary()
            if op == "*":
                value = value * rhs
                continue
            if rhs == 0:
                raise CalculatorError("division by zero")
            if op == "/":
                value = value / rhs
            elif op == "//":
                value = value // rhs
            else:
                value = value % rhs

    def _unary(self) -> float:
        """Leading ``+``/``-``; binds looser than ``**``."""
        op = self._peek_op()
        if op == "+":
            self._advance()
            return +self._unary()
        if op == "-":
            self._advance()
            return -self._unary()
        return self._power()

    def _power(self) -> float:
        """``**`` is right-associative: the right operand is a unary."""
        base = self._primary()
        if self._peek_op() != "**":
            return base
        self._advance(2)
        exponent = self._unary()
        try:
            result = base**exponent
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError:
            raise CalculatorError("result out of range") from None
        if isinstance(result, complex):
            raise CalculatorError("result is not a real number")
        return float(result)

    def _primary(self) -> float:
        """A number literal or a parenthesized expression."""
        token = self._peek()
        if token is None:
            raise CalculatorError("unexpected end of expression")
        if token == "(":
            self._advance()
            value = self._expression()
            if self._peek() != ")":
                raise CalculatorError("missing closing parenthesis")
            self._advance()
            return value
        if token == ")":
            raise CalculatorError("unexpected token: ')'")
        if token[0].isdigit() or token[0] == ".":
            self._advance()
            return float(token)
        raise CalculatorError(f"unexpected token: {token!r}")


def calculate(expression: str) -> float:
    """Evaluate *expression* and return the result as a float.

    Raises CalculatorError on unknown characters, malformed syntax, or
    division/modulo by zero.
    """
    try:
        return _Parser(tokenize(expression)).parse()
    except RecursionError:
        raise CalculatorError("expression too deeply nested") from None


def _format_result(value: float) -> str:
    """Render *value* without a trailing ``.0`` when it is integral."""
    if value.is_integer():
        return str(int(value))
    return repr(value)


def main(argv: list[str] | None = None) -> int:
    """Run the calculator as a command-line program.

    ``argv`` defaults to ``sys.argv[1:]``; the arguments are joined with
    spaces and evaluated. With no arguments, one expression is read from
    stdin. ``-h``/``--help`` print the usage line. Returns the process exit
    code: 0 on success, 1 on a bad expression, 2 when no expression was
    given.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv:
        expression = " ".join(argv)
    else:
        line = sys.stdin.readline()
        if not line.strip():
            print(USAGE, file=sys.stderr)
            return 2
        expression = line
    try:
        result = calculate(expression)
    except CalculatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(_format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
