def capitalize(s):
    try:
        return s[0].upper() + s[1:].lower()
    except IndexError:
        return s[0].upper()
