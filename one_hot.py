import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation, Input
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import nltk
import re
from nltk.corpus import stopwords
import os
nltk.download('stopwords')
# from nltk.stem.porter import PorterStemmer
# ps = PorterStemmer()

# Since the above approach doesn't seem to be promising, continue with embeddings from glove-50
def read_glove_file(filename = 'glove.6B.50d.txt'):
    with open(filename, 'r') as f:
        words = set()
        words_to_vec_map = dict()
        for line in f:
            # Remove extra white spaces and split the line
            li = line.strip().split()
            words.add(li[0])
            words_to_vec_map[li[0]] = np.array(li[1:], dtype=np.float64)
    idx_to_words = dict()
    words_to_idx = dict()
    i = 1
    for word in sorted(words):
        words_to_idx[word] = i
        idx_to_words[i] = word
        i += 1
    return words_to_vec_map, words_to_idx, idx_to_words

def preprocess_text(txt):
    review = re.sub('[^a-zA-Z]', ' ', txt)
    review = review.lower()
    review = review.split()

    # review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = [word for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

def load_data(label_to_idx, csv_file = 'data/twitter_training.csv'):
    df = pd.read_csv(csv_file, header=None)
    df = df.dropna()
    data = set()
    for _, row in tqdm(df.iterrows(), total = len(df)):
        if (row[2]=="Irrelevant"): continue
        txt = row[3]
        txt = preprocess_text(txt)
        data.add((txt, label_to_idx[row[2]]))
    data = list(data)
    X = np.asarray([tup[0] for tup in data])
    y = np.asarray([tup[1] for tup in data], dtype=int)
    return X, y

def sentence_to_indices(X, words_to_idx ,maxLen):
    m = X.shape[0]
    X_out = np.zeros((m, maxLen))
    for i in range(m):
        li = X[i].lower().strip().split()
        j = 0
        for w in li:
            if (j >= maxLen): break
            if w in words_to_idx.keys():
                X_out[i,j] = words_to_idx[w]
            j += 1
    return X_out

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_size = len(word_to_index) + 1              # adding 1 to fit Keras embedding (requirement)
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]    # define dimensionality of your GloVe word vectors (= 50)
      
    ### START CODE HERE ###
    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros((vocab_size, emb_dim))
    
    # Step 2
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(input_dim = vocab_size, output_dim = emb_dim, trainable = False)
    ### END CODE HERE ###

    # Step 4 (already done for you; please do not modify)
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def convert_to_one_hot(y, C = 3):
    return np.eye(C)[y.reshape(-1)]

def build_model(input_shape, words_to_vec_map, words_to_idx):
    sentence_indices = Input(shape = input_shape)

    embedding_layer = pretrained_embedding_layer(words_to_vec_map, words_to_idx)

    embeddings = embedding_layer(sentence_indices)

    X = LSTM(units=128, return_sequences=True)(embeddings)

    X = Dropout(rate = 0.5)(X)

    X = LSTM(units = 128, return_sequences=False)(X)

    X = Dropout(rate = 0.5)(X)

    X = Dense(units= 3)(X)

    X = Activation('softmax')(X)

    model = Model(inputs = sentence_indices, outputs = X)

    return model

label_to_idx = {'Irrelevant': 0, 'Negative': 1, 'Neutral': 0, 'Positive': 2}

words_to_vec_map, words_to_idx, idx_to_words = read_glove_file()

print("\n>>Done loading word vectors")
print("\n>>Loading data...")


X_train, y_train = load_data(label_to_idx, 'data/twitter_training.csv')
X_val, y_val = load_data(label_to_idx, 'data/twitter_validation.csv')

print(">>Done loading data\n")

maxLen = 64
print(f"Maximum length of a message is taken as: {maxLen}\n")

X_train_indices = sentence_to_indices(X_train, words_to_idx, maxLen)
X_val_indices = sentence_to_indices(X_val, words_to_idx, maxLen)

y_train_oh = convert_to_one_hot(y_train, 3)
y_val_oh = convert_to_one_hot(y_val, 3)

model_save_name = "model_one_hot_preprocess.h5"

if (os.path.isfile(model_save_name)):
    model = load_model(model_save_name)
    print(f"Model loaded: {model_save_name}")
    model_save_name = "model_one_hot_preprocess_again.h5"
else:
    model = build_model((maxLen, ), words_to_vec_map, words_to_idx)
    print(">>Model build completed\n")

    model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    print(">>Model compilation completed\n")

history = model.fit(X_train_indices, y_train_oh, epochs = 5, batch_size = 32, shuffle=True)

model.save(model_save_name)
print(f">>Done fitting, saved the model '{model_save_name}'\n")

print("Running on validation data:")
model.evaluate(X_val_indices, y_val_oh)
