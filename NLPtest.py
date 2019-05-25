import keras
from keras_contrib.layers import CRF
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from NLPmodel import build_model
import re
import math
from config import *


def SplitByBD(line):
    SplitWordStr = "，。！？、／：；《》（）、"
    flag = False
    sentenceList = []
    sentence = ""

    for i in range(len(line)):
        sentence += line[i]
        if line[i] in SplitWordStr and i + 1 < len(line) and line[i + 1] not in SplitWordStr:
            sentenceList.append(sentence)
            sentence = ""

    if sentence != "":
        sentenceList.append(sentence)
    return sentenceList


def getsplittext(label, sentence, Flag):
    splittext = ""
    for i in range(len(sentence)):
        splittext += sentence[i]
        if label[i] == 1 or label[i] == 4:
            splittext += "  "
    return splittext


def test(max_sentence_len, dict_len, label_cnt, char2ind, ind2char):
    global wantTestCnt

    # load model
    model = build_model(max_sentence_len, dict_len, label_cnt)
    model.load_weights('train_model.h5')

    # open file
    testFile = open(testFileName, mode="r", encoding="UTF-8")
    resultFile = open(resultFileName, mode="w", encoding="UTF-8")
    
    ihhhh = 0
    for line in testFile:
        ihhhh += 1
        if ihhhh > wantTestCnt and wantTestCnt != 0:
            break

        sentenceList = []
        sentenceList = SplitByBD(line[0:-1])

        for i in range(len(sentenceList)):
            sentence = sentenceList[i]

            maxIter = math.ceil(len(sentence) / max_sentence_len)
            iter = 0
            while iter < maxIter:

                if len(sentence[iter * max_sentence_len:]) <= max_sentence_len:
                    tmp_sentence = sentence[iter * max_sentence_len:]
                    Flag = True
                else:
                    tmp_sentence = sentence[iter * max_sentence_len: iter * max_sentence_len + max_sentence_len]
                    Flag = False

                mark_sentence = [char2ind.get(x, 0) for x in tmp_sentence]
                pro_sentence = pad_sequences([mark_sentence], max_sentence_len, padding="post")
                label = model.predict(pro_sentence)  # 应该是一个max_sentence_len * 6维的向量
                label = np.argmax(label, axis=2)[0]

                splittext = getsplittext(label, tmp_sentence, Flag)
                resultFile.write(splittext)

                iter += 1

        resultFile.write("\n")

    testFile.close()
    resultFile.close()
    return

