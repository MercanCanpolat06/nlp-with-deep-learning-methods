import numpy as np
import math
import random
import gc # since i work with large data

with open("data/text8", "r") as f:
    data = f.read().split()

from collections import Counter
counts = Counter(data)

word2id = {}
id2word = {}
i = 0
filtered_words = [word for word, count in counts.items() if count > 4] # removing rare words
#rare words : occuring less than 5 times according to the article

word2id = {word: i for i, word in enumerate(filtered_words)}
id2word = {i: word for i, word in enumerate(filtered_words)}

vocab_size = len(word2id) #different words

#subsampling of frequent words
threshold = 1e-5 # Genelde bu değer iyi sonuç verir

def get_discard_prob(word_count, total_count):
    f = word_count / total_count
    # P(wi) = 1 - sqrt(t / f(wi)) 
    return 1 - math.sqrt(threshold / f)

#if the random number is larger than the probability, keep the word
train_data = []
corpus_size = len(data) 
for word in data:
    if word not in word2id: # eliminated lesser than 5
        continue
        
    p_discard = get_discard_prob(counts[word], corpus_size)

    if random.random() > p_discard:
        train_data.append(word2id[word])

train_data = np.array(train_data, dtype=np.int32) # train data is now a numpy array

#creating a numpy matrix for each word, 100 dimensional vectors

embed_dim = 100

W_in = np.random.uniform( #center matrix
    low=-0.5 / embed_dim, #at start, we want to keep the values close to 0
    high=0.5 / embed_dim, 
    size=(vocab_size, embed_dim)
).astype(np.float32)

W_out = np.random.uniform( #context matrix
    low=-0.5 / embed_dim, 
    high=0.5 / embed_dim, 
    size=(vocab_size, embed_dim)
).astype(np.float32)

#creating unigram table for negative sampling

z = np.array([counts[id2word[i]]**0.75 for i in range(len(word2id))])
probs = z / np.sum(z)

table_size = 100_000_000 # 10^8 eleman
unigram_table = np.zeros(table_size, dtype=np.int32)

import numpy as np

counts_in_table = (probs * table_size).astype(np.int32)

#add difference to first element
diff = table_size - np.sum(counts_in_table)
counts_in_table[0] += diff
unigram_table = np.repeat(np.arange(len(word2id)), counts_in_table)
np.random.shuffle(unigram_table)
#int32 for less area
unigram_table = unigram_table.astype(np.int32)

del counts_in_table, probs, z
gc.collect()

context_size = 5

def create_pairs(train_data, window_size): #each pair is center_id & context_id
    pairs = []
    for i in range(len(train_data)):
        center_id = train_data[i]
        current_window = np.random.randint(1, window_size + 1) #weighting trick, so closer words are more important
        
        start = max(0, i - current_window)
        end = min(len(train_data), i + current_window + 1) #avoid getting values out of the list
        
        for j in range(start, end):
            if i != j:
                pairs.append((center_id, train_data[j]))
    return np.array(pairs, dtype=np.int32)

def get_batches(all_pairs, unigram_table, batch_size, n_negs):
    for i in range(0, len(all_pairs), batch_size):
        batch_slice = all_pairs[i : i + batch_size]
    
        inputs = batch_slice[:, 0]
        labels = batch_slice[:, 1]
        neg_samples = unigram_table[np.random.randint(0, len(unigram_table), (len(inputs), n_negs))]
        
        yield inputs, labels, neg_samples

#FORWARD AND BACKWARDS PASS
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

initial_lr = 0.025
min_lr = 0.0001
epochs = 5
all_pairs = create_pairs(train_data, window_size=5)
total_steps = (len(all_pairs) // 512) * epochs #512 is batch size
current_step = 0


for epoch in range(epochs):
     #shuffelled the pairs
    np.random.shuffle(all_pairs)

    # get_batches jeneratörünü çağır
    batches = get_batches(all_pairs, unigram_table, batch_size=512, n_negs=5)
    
    
    for batch_inputs, batch_labels, batch_negatives in batches:
        learning_rate = initial_lr * (1.0 - (current_step / total_steps))
        learning_rate = max(learning_rate, min_lr)

        # v_c: center words
        # Size: [512, 100] (batch_size, embed_dim)
        v_c = W_in[batch_inputs] 
        
        # v_p: neighbor words from W_out
        # Size: [512, 100]
        v_p = W_out[batch_labels] 
        
        # v_n: Wrong neighbors (from W_out)
        # Size: batch_size, n_negs, embed_dim)
        v_n = W_out[batch_negatives] 

        pos_dot_product = np.sum(v_c * v_p, axis=1)
        pos_probs = sigmoid(pos_dot_product)

        pos_error = pos_probs - 1.0 # 1 because were sure that these words are neighbors
        
        neg_dot_product = np.sum(v_c[:, np.newaxis, :] * v_n, axis=2) 
        # added imaginary dimension to v_c so it is multiplieble
        neg_probs = sigmoid(neg_dot_product)   
        neg_error = neg_probs - 0.0

        #Attraction of the positive neighbor
        grad_vc_pos = pos_error[:, np.newaxis] * v_p 
        
        #repulsion by the negative neighbor
        grad_vc_neg = np.sum(neg_error[:, :, np.newaxis] * v_n, axis=1)
        
        # total change
        grad_vc = grad_vc_pos + grad_vc_neg 

        #change of context words according to the center word
        grad_vp = pos_error[:, np.newaxis] * v_c # [512, 100]
        grad_vn = neg_error[:, :, np.newaxis] * v_c[:, np.newaxis, :] # [512, 5, 100]

        # we are subtracting the gradients from the cells
        # W_in (center matrix)
        W_in[batch_inputs] -= learning_rate * grad_vc
        
        # W_out (context_matrix)
        W_out[batch_labels] -= learning_rate * grad_vp
        W_out[batch_negatives] -= learning_rate * grad_vn

        current_step +=1

#VISUALIZATION

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

num_words_to_plot = 200
vectors_to_plot = W_in[:num_words_to_plot]

tsne_model = TSNE(n_components=2, random_state=42, perplexity=30)
vectors_2d = tsne_model.fit_transform(vectors_to_plot)

plt.figure(figsize=(14, 10))

for i in range(num_words_to_plot):
    x = vectors_2d[i, 0]
    y = vectors_2d[i, 1]
    
    plt.scatter(x, y, c='steelblue', edgecolors='k')
    word = id2word[i]
    plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.title("Word2Vec Word Space (t-SNE)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


        