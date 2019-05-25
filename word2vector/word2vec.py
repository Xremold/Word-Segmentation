import gensim

class Sentence(object):
    def __init__(self, inputStr):
        self.Fname = inputStr

    def __iter__(self):
        inputFile = open(self.Fname, mode="r", encoding="UTF-8")
        for line in inputFile:
            line = line.strip().replace(" ", "")
            wordList = [word for word in line]
            yield wordList


Mysentence = Sentence("data/train.txt")
print("hhhh")
model=gensim.models.Word2Vec(Mysentence, size=100, window=5, min_count=1, iter=10, workers=4)
model.save("word2vec")
print("hhhh")
print(model["我"])