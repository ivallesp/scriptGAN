flatten = lambda l: [item for sublist in l for item in sublist]

def batching(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]