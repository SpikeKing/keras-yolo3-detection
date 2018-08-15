#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/2
"""

from keras import backend as K

sess = K.get_session()
a = K.constant([2, 4])
b = K.constant([3, 2])
c = K.square(a - b)
print(sess.run(c))

# K.square(raw_true_wh - raw_pred[..., 2:4])
