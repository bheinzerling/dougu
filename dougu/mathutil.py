def avg(values):
    n_values = len(values)
    if n_values:
        return sum(values) / n_values
    return 0


def decimal_round(number, n_decimal_places=0):
    from decimal import Decimal, ROUND_HALF_UP
    round_fmt = '1'
    if n_decimal_places > 0:
        round_fmt += '.' + '0' * n_decimal_places
    return Decimal(number).quantize(Decimal(round_fmt), rounding=ROUND_HALF_UP)
