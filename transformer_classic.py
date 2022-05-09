import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from network import TransformerBlock
from network import TokenAndPositionEmbedding


sz_vocab = 20000  # Only consider the top 20k words
len_max = 200  # Only consider the first 200 words of each movie review

(tr_x, tr_y), (val_x, val_y) = imdb.load_data(num_words=sz_vocab)
print(len(tr_x), "Length of Sequences for Training")
print(len(val_x), "Length of Sequences for Validation")

tr_x = tf.keras.preprocessing.sequence.pad_sequences(tr_x, maxlen=len_max)
val_x = tf.keras.preprocessing.sequence.pad_sequences(val_x, maxlen=len_max)

emb_sz = 32
att_head = 2 
transformer_ffnet = 32

inp = Input(shape=(len_max,))
layer_emb = TokenAndPositionEmbedding(len_max, sz_vocab, emb_sz)
k = layer_emb(inp)
blk_transformer = TransformerBlock(emb_sz, att_head, transformer_ffnet)
k = blk_transformer(k)
k = GlobalAveragePooling1D()(k)
k = Dropout(0.1)(k)
k = Dense(20, activation="relu")(k)
k = Dropout(0.1)(k)
out = Dense(2, activation="softmax")(k)

mdl = Model(inputs=inp, outputs=out)

mdl.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

sz_batch = 64
num_ep = 10

net = mdl.fit(tr_x, tr_y, 
                    batch_size=sz_batch, epochs=num_ep, 
                    validation_data=(val_x, val_y)
                   )

mdl.save_weights("transformer_network.h5")

res = mdl.evaluate(val_x, val_y, verbose=2)

for n, v in zip(mdl.metrics_names, res):
    print("%s: %.3f" % (n, v))