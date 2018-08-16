#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/2
"""

from keras import backend as K

sess = K.get_session()

a = K.ones_like([[2, 4], [1, 2], [5, 6]])
b = a * 2

print(sess.run(b))

# K.square(raw_true_wh - raw_pred[..., 2:4])
