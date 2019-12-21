# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import math

import doctest
import unittest


from numpy.testing import assert_allclose, assert_array_equal

import mandel


class gaussianTestCase(unittest.TestCase):
    
    def test_60(self):
        x = math.radians(60.0001)
        mu=1
        sigma=2
        assert_allclose(math.exp(-0.5*((x-mu)/sigma)*((x-mu)/sigma)) / sigma / math.sqrt(2*math.pi), mandel.gaussian(x,mu,sigma), rtol=1e-5)


#def load_tests(loader, tests, ignore):
    #tests.addTests(doctest.DocTestSuite(mandel.mandelRender))
    #return tests
if __name__ == '__main__':
    unittest.main()
