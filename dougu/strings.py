import re
import random
import string


# source: https://github.com/juliaelman/anora/blob/master/anora/templatetags/anora.py
CONSONANT_SOUND = re.compile(r'''one(![ir])''', re.IGNORECASE | re.VERBOSE)
VOWEL_SOUND = re.compile(
    r'''[aeio]|u([aeiou]|[^n][^aeiou]|ni[^dmnl]|nil[^l])|h(ier|onest|onou?r|ors\b|our(!i))|[fhlmnrsx]\b''', re.IGNORECASE | re.VERBOSE)


try:
    from colorama import Fore, Back, Style
    red = Fore.RED
    green = Fore.GREEN
    yellow = Fore.YELLOW
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


def random_del_many(s):
    l = len(s)
    if l < 2:
        return s
    i = random.randint(0, l - 1)
    n = random.randint(1, int(l / 3) or 1)
    return s[:i] + s[i + n:]


def random_insert(s):
    l = len(s)
    if l < 2:
        return s
    i = random.randint(0, l - 1)
    j = random.randint(0, l - 1)
    return s[:i] + s[j] + s[i:]


def make_random_insert_chars(chars=string.ascii_letters + string.digits):

    def random_insert_chars(s):
        l = len(s)
        if l < 2:
            return s
        i = random.randint(0, l - 1)
        n = random.randint(1, 4)
        rnd_chars = "".join(random.sample(chars, n))
        return s[:i] + rnd_chars + s[i:]

    return random_insert_chars


def random_duplicate(s):
    l = len(s)
    i = random.randint(0, l - 1)
    n = random.randint(1, 10)
    return s[:i] + s[i] * n + s[i:]


def augment(strings, augment_funcs):
    for s in strings:
        yield from (f(s) for f in augment_funcs)


def augment_with_id(ids, strings, augment_funcs):
    for id, s in zip(ids, strings):
        for aug in (f(s) for f in augment_funcs):
            yield id, aug


def yesno_mark(condition):
    """Return a colored "yes" or "no" check mark depending
    on wether condition is True or False"""
    if condition:
        return f"{green}✓{reset}"
    return f"{red}✗{reset}"


def random_string(length=8, chars=None):
    if chars is None:
        import string
        chars = string.ascii_letters + string.digits
    import random
    return "".join(random.choices(chars, k=length))


def color_by_weight(tokens, weights, styles=None, thresholds=None):
    """Return a string representing weights assigned to tokens via
    background colors. Useful for visualizing neural attention
    directly in the terminal window, e.g. during training.

    The visualization styles for each weight range
    are given as color and style flags from the colorama package. Weight
    ranges can be specified as list of thresholds.

    If the dependendy colorama is not installed,
    show the weights instead."""
    import numpy as np
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


def token_shapes(tokens, collapse=True):
    """Returns strings which encode the shape of tokens. If collapse
    is set, repeats are collapsed and infrequent shapes encoded as "other":
        Aa  | capitalized
        a   | all lowercase
        .   | all punctuation
        0   | all digits
        A   | all UPPERCASE
        0a0 | digits - lower - digits
        %   | other
    """
    collapsed_shapes = {"Aa", "a", ".", "0", "A", "0a0"}

    def char_shape(char):
        if not char.isalnum():
            return "."
        if char.isdigit():
            return "0"
        if char.isupper():
            return "A"
        return "a"

    shapes = [[char_shape(c) for c in token] for token in tokens]
    if collapse:
        def _collapse(chars):
            last = None
            for c in chars:
                if c != last:
                    yield c
                    last = c
    else:
        def _collapse(chars):
            return chars
    shapes = ["".join(_collapse(shape)) for shape in shapes]
    if collapse:
        return [s if s in collapsed_shapes else "%" for s in shapes]
    return shapes


def bpe_demo(s, latex=False):
    """Apply Byte-Pair Encoding to string s.
    Returns learned BPE vocabulary and the result of encoding s
    with it.
    """
    from collections import Counter
    from collections import deque, defaultdict

    space = ' '
    s = s.replace(' ', space)

    def chunks(items, size):
        d = deque(items[:2], 2)
        for item in items[2:]:
            yield tuple(d)
            d.append(item)
        yield tuple(d)

    enc = list(s)
    vocab = []

    def to_latex(prev_enc, enc, poss=None, step=None):
        if latex and poss:
            _enc = list(enc)
            for p in poss:

                if step is None:
                    hl = '\\hl{{'
                else:
                    hl = f'\\hl<{step}>{{'
                prev_enc[p] = hl + prev_enc[p] + '}'
                prev_enc[p + 1] = hl + prev_enc[p + 1] + '}'

                # if step is None:
                #     hl = '\\hl{{'
                # else:
                #     hl = f'\\hl<{step + 1}>{{'
                # _enc[p] = hl + _enc[p] + '}'
            return _enc
        else:
            return enc

    def format_enc(enc):
        return " ".join(enc).replace('   ', ' \\_ ')

    encs = [enc]
    step = 3
    enc = list(enc)
    while True:
        pairs = list(chunks(enc, 2))
        pair2pos = defaultdict(list)
        for pos, pair in enumerate(pairs):
            pair2pos[pair].append(pos)
        char_pairs = filter(lambda p: p[0] != space and p[1] != space, pairs)
        most_freq, count = Counter(char_pairs).most_common(1)[0]
        if count == 1:
            break
        prev_pos = -2
        to_del_pos = []
        poss = pair2pos[most_freq]
        for pos in poss:
            if pos == prev_pos + 1:
                continue
            enc[pos] = "".join(most_freq)
            prev_pos = pos
            to_del_pos.append(pos + 1)
        _enc = to_latex(encs[-1], enc, poss, step)
        for pos in reversed(to_del_pos):
            del enc[pos]
            del _enc[pos]
        encs.append(_enc)
        vocab.append((most_freq[0], most_freq[1], "".join(most_freq)))
        step += 2
    encs = list(map(format_enc, encs))
    if latex:
        encs = [f'\\item<{2 * (i + 1)}-> ' + enc for i, enc in enumerate(encs)]
    return vocab, encs


if __name__ == "__main__":
    s = 'the netherlands are neither here nor there'
    vocab, encs = bpe_demo(s, latex=True)
    print(s)
    for enc in encs:
        print(enc)
    print(", ".join([v[2] for v in vocab]))
    import numpy as np
    tokens = "return a string representing weights assigned to tokens".split()
    for _ in range(10):
        energies = np.random.gamma(2, 2, len(tokens))
        attn = np.exp(energies) / np.sum(np.exp(energies), axis=0)
        print(color_by_weight(tokens, attn))


def conditional_color(string, condition, true_color=green, false_color=red):
    """Color string according to whether condition is True or False.
    """
    color = true_color if condition else false_color
    return f"{color}{string}{reset}"


def conditional_colors(strings, conditions, true_color=green, false_color=red):
    return [
        conditional_color(
            string, condition, true_color=true_color, false_color=false_color)
        for string, condition in zip(strings, conditions)
        ]


# source: https://github.com/juliaelman/anora/blob/master/anora/templatetags/anora.py
def an_or_a(text):
    return (
        'an' if not CONSONANT_SOUND.match(text) and VOWEL_SOUND.match(text)
        else 'a'
        )


def longest_prefix_match(prefix, targets):
    from os.path import commonprefix
    max_len = 0
    longest_match = None
    for target in targets:
        match = commonprefix([prefix, target])
        match_len = len(match)
        if match_len > max_len:
            max_len = match_len
            longest_match = target
    return longest_match
