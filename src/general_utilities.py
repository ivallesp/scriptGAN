flatten = lambda l: [item for sublist in l for item in sublist]

def batching(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_exponential_generator(current_value, final_value, eta):
    while 1:
        current_value = current_value*(1-eta) + final_value*(eta)
        yield current_value