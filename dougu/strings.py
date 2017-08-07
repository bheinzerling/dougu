import re


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
