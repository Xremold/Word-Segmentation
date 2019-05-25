########################################### 配置项 ##############################################

# 设置为False表示采用训练好的模型训练
isTrain = True

# 训练循环次数
Epoc = 20

# 训练Dropout率
Dropout = 0.2

# 训练LSTM神经元数目
LTSMUnitCnt = 64

# 设置为0表示全部数据都要处理
wantProcessCnt = 0

# 设置为0表示全部跑完test数据集
wantTestCnt = 0

testFileName = "test.txt"

resultFileName = "result.txt"

# 最长句子的长度
magic_max_sentence_len = 30

# 选择一个model，model1加了CRF，model2无
choose_model1 = True