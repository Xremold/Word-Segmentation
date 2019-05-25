from keras_contrib.layers import CRF
from keras.layers import *
from keras.models import *
from config import *


def build_model(max_sentence_len, dict_len, label_cnt):

    # model1: embedding + bilstm + dense + crf
    model1 = Sequential()
    model1.add(Embedding(dict_len, output_dim=50, input_length=max_sentence_len))
    lstm = LSTM(LTSMUnitCnt, return_sequences=True, dropout=Dropout)
    model1.add(Bidirectional(lstm, merge_mode="sum"))
    model1.add(TimeDistributed(Dense(label_cnt, activation="softmax")))
    crf = CRF(label_cnt, sparse_target=False)
    model1.add(crf)
    model1.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

    # model2: embedding + bilstm + dense
    model2 = Sequential()
    model2.add(Embedding(dict_len, output_dim=50, input_length=max_sentence_len))
    lstm = LSTM(LTSMUnitCnt, return_sequences=True, dropout=Dropout)
    model2.add(Bidirectional(lstm, merge_mode="sum"))
    model2.add(TimeDistributed(Dense(label_cnt, activation="softmax")))
    model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    if choose_model1:
        model1.summary()
        return model1
    else:
        model2.summary()
        return model2