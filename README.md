# reading_comprehension



后续改进思路：
+ 对抗学习，添加embedding层的干扰
+ 做test time augmentation

______

***** New Feb 19th, 2021 *****


无效方法：
+ 替换成wobert，提升序列长度的同时，能保证max_length还是512

有效方法：
+ 添加c3数据，做Curriculum Learning
+ 添加warmup，调整合适的学习率，每n个steps计算一次验证集，根据acc保存模型

线上acc：44.72177

______


***** New Feb 13th, 2021 *****

基于BertForMultipleChoice的baseline，max_length 512

线上acc：41.85761
