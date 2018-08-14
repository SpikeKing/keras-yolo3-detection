#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/2
"""

import numpy as np

a = np.array([[1, 2]])
print(a.shape)  # (1, 2)
b = np.array([[1], [2]])
print(b.shape)  # (2, 1)
c = a + b
print(c.shape)  # (2, 2)
print(c)
"""
[[2 3]
 [3 4]]
"""
