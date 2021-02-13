# reading_comprehension



后续改进思路：
+ 添加warmup，调整合适的学习率，每n个steps计算一次验证集，根据acc保存模型
+ 替换成wobert，提升序列长度的同时，能保证max_length还是512

***** New Feb 13th, 2021 *****

基于BertForMultipleChoice的baseline，max_length 512，线上acc：41.85761
