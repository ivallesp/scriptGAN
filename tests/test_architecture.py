import unittest
import math
from src.architecture import __architectures__
import numpy as np
import tensorflow as tf



class ArchitectureTester(unittest.TestCase):
    def test_architecture_graph_compilation(self):
        for name, architecture in __architectures__.items():
            try:
                architecture()
            except ValueError:
                self.fail("It was not possible to compile the '{}' architecture graph!".format(name))
