import keras
from keras_contrib.layers import CRF
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from NLPtest import *
from NLPmodel import build_model
from keras.utils import plot_model
import re
import matplotlib.pyplot as plt
from config import *


def generate_graph(history):
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    fig.savefig("model accuracy.png")

    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower left')
    # plt.show()
    fig.savefig("model loss.png")

def padding_and_vec(X_data, Y_label, max_sentence_len, ind2char):
    X_data = pad_sequences(X_data, max_sentence_len, padding="post", truncating="post")
    Y_label = pad_sequences(Y_label, max_sentence_len, padding="post", truncating="post")

    Y_label = keras.utils.to_categorical(np.array(Y_label), 5)
    Y_label = Y_label.reshape(len(Y_label), max_sentence_len, -1)

    return X_data, Y_label


def give_max_sentence_len(X_data):
    return magic_max_sentence_len


def data_process():
    input_file = open("train.txt", mode="r", encoding="UTF-8")
    X_data = []
    Y_label = []
    ihhhhh = 0
    for line in input_file:

        ihhhhh += 1
        if ihhhhh >= wantProcessCnt and wantProcessCnt != 0:
            break

        sentenceList = SplitByBD(line[0:-1])
        for sentence in sentenceList:
            wordList = sentence.split()
            Y_label_item = []
            if len(sentence) == 0 or sentence[0] == "\n":
                continue
            for word in wordList:
                if len(word) == 0 or word[0] == "\n":
                    continue
                for i in range(len(word)):
                    if len(word) == 1:
                        Y_label_item.append(1)  # S
                    elif i == 0:
                        Y_label_item.append(2)  # B
                    elif i + 1 < len(word):
                        Y_label_item.append(3)  # M
                    else:
                        Y_label_item.append(4)  # E

            X_data.append(list("".join(wordList)))
            Y_label.append(Y_label_item)

    input_file.close()
    return X_data, Y_label


def creat_index(X_data):
    chars = set()
    for X_data_item in X_data:
        chars.update(X_data_item)

    ind2char = {k+1: v for k, v in enumerate(chars)}
    ind2char[0] = "U"
    char2ind = {v: k for k, v in ind2char.items()}

    for i in range(len(X_data)):
        X_data[i] = [char2ind[x] for x in X_data[i]]
    return X_data, char2ind, ind2char


def train(X_data, Y_label, max_sentence_len, char2ind, dict_len, label_cnt):
    model = build_model(max_sentence_len, dict_len, label_cnt)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

    history = model.fit(X_data, Y_label, batch_size=256, epochs=Epoc, validation_split=0.1, verbose=1)
    score = model.evaluate(X_data, Y_label, batch_size=512)
    print("Training Finished!", score, sep="\n")

    # save model
    model.save_weights('train_model.h5')
    model.save("wordSplit.mod")

    generate_graph(history)

    return


def main():
    X_data, Y_label = data_process()
    X_data, char2ind, ind2char = creat_index(X_data)

    max_sentence_len = give_max_sentence_len(X_data)

    X_data, Y_label = padding_and_vec(X_data, Y_label, max_sentence_len, ind2char)

    if isTrain:
        train(X_data, Y_label, max_sentence_len, char2ind, dict_len=len(char2ind), label_cnt=5)

    test(max_sentence_len, len(char2ind), 5, char2ind, ind2char)
    return


if __name__ == "__main__":
    main()
