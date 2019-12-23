# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import math

import doctest
import unittest


import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import mandel


class mandelTestCase(unittest.TestCase):
    
    def test_60(self):
        x = math.radians(60.0001)
        mu=1
        sigma=2
        assert_allclose(math.exp(-0.5*((x-mu)/sigma)*((x-mu)/sigma)) / sigma / math.sqrt(2*math.pi), mandel.gaussian(x,mu,sigma), rtol=1e-5)
        
    def test_gaussian(self):
        mu=0
        sigma=1
        
        a = np.random.normal(0,1,20000)
        b=np.histogram(a)
        c=[(b[1][i]+b[1][i+1])/2 for i in range(len(b[1])-1)]
        d = list(map(lambda x: mandel.gaussian(x,0,1),c))
        e=b[0]/np.max(b[0])
        d=d/np.max(d)
        assert_allclose(e, d, atol=0.05)
        
    def test_cycle(self):
        result=mandel.cycle(0,0,20000000)
        self.assertEqual(result[0],20000000)
        self.assertLessEqual(result[1],4)

        
        


#def load_tests(loader, tests, ignore):
    #tests.addTests(doctest.DocTestSuite(mandel.mandelRender))
    #return tests
if __name__ == '__main__':
    unittest.main()
