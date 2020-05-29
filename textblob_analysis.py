'''
Quais serão os passos:
    1. Carregar os dados positivos e negativos
    2. Fazer uma limpeza (tirar letras maiusculas, stopwords, etc) e colocar um label (pos/neg)
    3. Separar train e test data
    4. Jogar no classificador Naive Bayes
    5. Fazer um input de novos dados para teste
'''

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import random
import re
import os

# Arrays para os dados de treino e test
reviews_train = []
reviews_test = []

# Arrays para os reviews positivos e negativos
train_pos = []
train_neg = []
test_pos = []
test_neg = []

# Função para adicionar os reviews separados no array
def add_review(path, review_array):
    for line in open(path, 'r'):
        review_array.append(line.strip())

# Adicionando os reviews positivos e negativos para o array reviews_train
directory_positive_reviews = r'/media/gustavo/StorageDevice/programming/python/newsbot/datasets/aclImdb/train/pos'
for entry in os.scandir(directory_positive_reviews):
    if (entry.path.endswith(".txt") and entry.is_file()):
        add_review(entry.path, train_pos)


directory_negative_reviews = r'/media/gustavo/StorageDevice/programming/python/newsbot/datasets/aclImdb/train/neg'
for entry in os.scandir(directory_negative_reviews):
    if (entry.path.endswith(".txt") and entry.is_file()):
        add_review(entry.path, train_neg)

# Adicionando os reviews positivos e negativos para o array reviews_test
test_pos_reviews = r'/media/gustavo/StorageDevice/programming/python/newsbot/datasets/aclImdb/test/pos'
test_neg_reviews = r'/media/gustavo/StorageDevice/programming/python/newsbot/datasets/aclImdb/test/neg'

for entry in os.scandir(test_pos_reviews):
    if (entry.path.endswith(".txt") and entry.is_file()):
        add_review(entry.path, test_pos)


for entry in os.scandir(test_neg_reviews):
    if (entry.path.endswith(".txt") and entry.is_file()):
        add_review(entry.path, test_neg)

# Em teoria, agora os reviews estao nos arrays correspondentes


# Regex para limpar dados
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

# Preprocess data
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

# Limpando os dados e adicionando em seus respectivos arrays
train_pos_clean = preprocess_reviews(train_pos)
train_neg_clean = preprocess_reviews(train_neg)
test_pos_clean = preprocess_reviews(test_pos)
test_neg_clean = preprocess_reviews(test_neg)

# Criando as tuplas train pos
for text in train_pos_clean:
    reviews_train.append((text, 'pos'))


# Criando as tuplas test pos
for text in test_pos_clean:
    reviews_test.append((text, 'pos'))

# Criando as tuplas train neg
for text in train_neg_clean:
    reviews_train.append((text, 'neg'))

# Criando as tuplas test neg
for text in test_neg_clean:
    reviews_test.append((text, 'neg'))

# Embaralhando os reviews
random.shuffle(reviews_train)
random.shuffle(reviews_test)

# Agora, em teoria, os meus dados estão prontos
# Estão no formato lista = [('texto', 'pos/neg')]

# Criando o cçassificador Naive Bayes
cl = NaiveBayesClassifier(reviews_train)

# User input
user_input = input("Write your review here: ")

prob_dist = cl.prob_classify(user_input)
if ((round(prob_dist.prob("pos"), 2)) > 0.5):
    print("Your review is positive")
else:
    print("Your review is negative")