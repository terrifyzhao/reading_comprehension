# reading_comprehension



后续改进思路：
+ 训练数据做增强
+ cls拼接其它embedding

______

***** New Feb 20th, 2021 *****
+ 对抗学习，添加embedding层的干扰，防止过拟合的效果很好

线上acc：47.13584(bert_base)

______

***** New Feb 19th, 2021 *****

无效方法：
+ 替换成wobert，提升序列长度的同时，能保证max_length还是512

有效方法：
+ 添加c3数据，做Curriculum Learning
+ 添加warmup，调整合适的学习率，每n个steps计算一次验证集，根据loss保存模型(warmup第一个epoch收敛会很慢)
+ 采用bert_large
+ 做test time augmentation，分前中后三段，稳定提升一个点

线上acc：46.31751

______

***** New Feb 13th, 2021 *****

基于BertForMultipleChoice的baseline，max_length 512

线上acc：41.85761
