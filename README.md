# YOLO v3 目标检测算法 源码

> 欢迎关注，微信公众号 **深度算法** （ID: DeepAlgorithm） 

相关文章：

- [探索 YOLO v3 源码 - 第1篇 训练](https://mp.weixin.qq.com/s/T9LshbXoervdJDBuP564dQ)
- [探索 YOLO v3 源码 - 第2篇 模型](https://mp.weixin.qq.com/s/N79S9Qf1OgKsQ0VU5QvuHg)
- [探索 YOLO v3 源码 - 第3篇 网络](https://mp.weixin.qq.com/s/hC4P7iRGv5JSvvPe-ri_8g)
- [探索 YOLO v3 源码 - 第4篇 真值](https://mp.weixin.qq.com/s/5Sj7QadfVvx-5W9Cr4d3Yw)
- [探索 YOLO v3 源码 - 第5篇 Loss](https://mp.weixin.qq.com/s/4L9E4WGSh0hzlD303036bQ)
- [探索 YOLO v3 源码 - 完结篇 预测](https://mp.weixin.qq.com/s/J1ddmUvT_F2HcljLtg_uWQ)

通过6篇文章，完整的呈现YOLO v3的源码细节。慢慢读完，掌握一些高级的深度学习开发技巧。

参考：

- [YOLO v3 Paper](https://arxiv.org/abs/1804.02767)
- [What’s new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

勘误：

1. 第4篇 真值，最后：“y_true的第0和1位是中心点xy，范围是`(0~13/26/52)`” -> “y_true的第0和1位是中心点xy，范围是`(0~1)`”；
2. 第3篇 网络，其中关于补充部分``1*1``卷积参数那个有误。不是``13*13*1*1*18``应该是``1*1*1024*18``； Thx@草绛ly
3. 第6篇 预测，max_boxes是在每层的feature_map中的每个类别分别最多产生20个框，而不是每张图片； Thx@略略略
