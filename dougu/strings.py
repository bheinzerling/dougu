import re
import random

import numpy as np

try:
    from colorama import Fore, Back, Style
    red = Fore.RED
    green = Fore.GREEN
    reset = Style.RESET_ALL
    weight_colors = True
except ImportError:
    red = green = reset = ""
    weight_colors = False


re_digit = re.compile("\d")
re_http_url = re.compile(r"https?:\/\/[^ ]*")
re_www_url = re.compile(r"www.[^ ]*")


def capitalize(s):
    try:
        return s[0].upper() + s[1:].lower()
    except IndexError:
        return s[0].upper()


def normalize_digits(text):
    return re_digit.sub("0", text)


def normalize_urls(text):
    text = re_http_url.sub("<url>", text)
    text = re_www_url.sub("<url>", text)
    return text


def random_cap(s):
    return "".join([random.choice([str.upper, str.lower])(c) for c in s])


def random_swap(s):
    l = len(s)
    if l < 2:
        return s
    i = random.randint(0, l - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def random_del(s):
    l = len(s)
    if l < 2:
        return s
    i = random.randint(0, l - 1)
    return s[:i] + s[i + 1:]


def random_insert(s):
    l = len(s)
    if l < 2:
        return s
    i = random.randint(0, l - 1)
    j = random.randint(0, l - 1)
    return s[:i] + s[j] + s[i:]


def augment(strings, augment_funcs=None):
    if not augment_funcs:
        augment_funcs = [
            (str.upper, 1),
            (str.lower, 1),
            (random_cap, 5),
            (random_swap, 5),
            (random_del, 5),
            (random_insert, 5)
            ]
    for s in strings:
        for f, times in augment_funcs:
            for _ in range(times):
                yield f(s)


def yesno_mark(cond):
    if cond:
        return f"{green}✓{reset}"
    return f"{red}✗{reset}"


def color_by_weight(tokens, weights, styles=None, thresholds=None):
    """Return a string representing weights assigned to tokens via
    background colors. Useful for visualizing neural attention
    directly in the terminal window, e.g. during training.

    The visualization styles for each weight range
    are given as color and style flags from the colorama package. Weight
    ranges can be specified as list of thresholds.

    If the dependendy colorama is not installed,
    show the weights instead."""
    if not styles:
        styles = [
            Style.RESET_ALL,
            Back.BLACK,
            Back.BLUE + Fore.MAGENTA,
            Back.RED + Fore.MAGENTA,
            Back.RED]
    if thresholds is None:
        thresholds = np.array([0.0, 0.1, 0.3, 0.45, 0.6])

    def get_style(weight):
        diffs = weight - thresholds
        return styles[len(diffs[diffs > 0]) - 1]

    def styled(token, weight):
        style = get_style(weight)
        return f"{style}{token}{reset}"

    if weight_colors:
        return " ".join(map(
            lambda t: styled(t[0], t[1]),
            zip(tokens, weights)))
    else:
        return " ".join(map(
            lambda t: f"{t[0]}|{t[1]:.2f}",
            zip(tokens, weights)))


if __name__ == "__main__":
    tokens = "return a string representing weights assigned to tokens".split()
    for _ in range(10):
        energies = np.random.gamma(2, 2, len(tokens))
        attn = np.exp(energies) / np.sum(np.exp(energies), axis=0)
        print(color_by_weight(tokens, attn))
