import re
from typing import Optional

import sympy
from sympy.parsing import sympy_parser

try:
    from pylatexenc import latex2text
    _HAS_PYLATEXENC = True
except ImportError:
    _HAS_PYLATEXENC = False


def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a, b = substr[0], substr[1]
                    if b != "{":
                        new_str += "{" + a + "}{" + b + "}" + substr[2:] if len(substr) > 2 else "{" + a + "}{" + b + "}"
                    else:
                        new_str += "{" + a + "}" + b + substr[2:] if len(substr) > 2 else "{" + a + "}" + b
        return new_str

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a, b = string.split("/")[0], string.split("/")[1]
        try:
            a, b = int(a), int(b)
            assert string == "{}/{}".format(a, b)
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
        except Exception:
            return string

    def _remove_right_units(string):
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            if len(splits) == 2:
                return splits[0]
        return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split and split[0] != "{":
                new_string += "\\sqrt{" + split[0] + "}" + split[1:]
            else:
                new_string += "\\sqrt" + split
        return new_string

    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string).replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    if _HAS_PYLATEXENC:
        try:
            expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
            expr = expr.replace("\\frac", " \\frac")
            expr = latex2text.LatexNodes2Text().latex_to_text(expr)
            expr = expr.replace("√", "sqrt").replace("π", "pi").replace("∞", "inf")
            expr = expr.replace("∪", "U").replace("·", "*").replace("×", "*")
            return expr.strip()
        except Exception:
            return expr
    return expr.replace("\\", "").replace("{", "").replace("}", "").strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return expr


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str):
    x = x.replace(",", "")
    return int(float(x))


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile(r"([0-9]) +([0-9])")
    return p1.sub(r"\1+\2", step)


def _normalize(expr: str) -> Optional[str]:
    if expr is None:
        return None
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in ["degree", "cm", "centimeter", "meter", "mile", "second", "minute", "hour",
                 "day", "week", "month", "year", "foot", "feet", "inch", "yard"]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(r",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass
    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr).replace(" ", "")
    expr = expr.replace("{", "").replace("}", "").lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "").replace("frac", "")
    return len(set(x for x in expr if x.isalpha()))


def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad in BAD_SUBSTRINGS:
        if bad in expr:
            return False
    for bad in BAD_REGEXES:
        if re.search(bad, expr):
            return False
    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            if sympy.simplify(sympy_diff) == 0:
                return True
    except Exception:
        pass
    return False


def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (len(expr) > 2 and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS
            and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)):
        return [e.strip() for e in expr[1:-1].split(",")]
    return [expr]


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None or len(given_normalized) == 0:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    gt_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)
    if len(gt_elems) > 1 and (ground_truth_normalized[0] != given_normalized[0]
                              or ground_truth_normalized[-1] != given_normalized[-1]):
        return False
    if len(gt_elems) != len(given_elems):
        return False
    for gt_e, g_e in zip(gt_elems, given_elems):
        if _is_frac(gt_e) and _is_frac(g_e):
            if gt_e != g_e:
                return False
        elif _str_is_int(gt_e) != _str_is_int(g_e):
            return False
        elif not are_equal_under_sympy(gt_e, g_e):
            return False
    return True


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    a = mathd_normalize_answer(given_answer)
    b = mathd_normalize_answer(ground_truth)
    return a is not None and b is not None and a == b
