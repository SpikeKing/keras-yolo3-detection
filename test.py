#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/2
"""

from keras import backend as K
import tensorflow as tf

sess = K.get_session()

a = K.constant([[2, 4], [1, 2], [5, 6]])
print(a.shape)
b = K.max(a, -1)
print(b.shape)
c = K.cast(0.4 < 0.5, dtype=tf.float32)
print(sess.run(c))

# K.square(raw_true_wh - raw_pred[..., 2:4])
