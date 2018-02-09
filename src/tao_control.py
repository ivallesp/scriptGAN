import numpy as np


class exponential_decay_generator:
    def __init__(self, start, finish):
        self.x = start
        self.finish = finish

    def get_value(self, lambda_):
        self.x = self.x * lambda_ + self.finish * (1-lambda_)
        return self.x


def compute_goodness_softmax(generation_code):
    generation_code_reshaped = generation_code.reshape(generation_code.shape[0]*generation_code.shape[1], generation_code.shape[2])
    maximum_indices = generation_code_reshaped.argmax(axis=1)
    maximums = generation_code.max(axis=2)
    tuple_indices = list(zip(range(len(maximum_indices)), maximum_indices))
    generation_code_reshaped[list(zip(*tuple_indices))] = -np.Inf
    maximums_2 = generation_code.max(axis=2)
    goodness = np.mean((maximums - maximums_2)/maximums)
    return goodness



class adaptive_decay_generator:
    def __init__(self, start, finish, min_lambda=0.999, max_lambda=0.9999):
        self.x = start
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.decay_gen = exponential_decay_generator(start, finish)

    def get_value(self, generations):
        goodness = compute_goodness_softmax(generations)
        lambda_calculated = self.min_lambda + (1-goodness)*(self.max_lambda-self.min_lambda)
        print(goodness, lambda_calculated)
        self.x = self.decay_gen.get_value(lambda_calculated)
        return self.x, goodness

