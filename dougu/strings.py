import re
import random


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


if __name__ == "__main__":
    s = "Muster Mark"
    for a in augment([s]):
        print(a)
