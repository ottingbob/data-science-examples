# Script for coding the self-attention mechanism of large language models
# from scratch.
# Reference:
# https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

import torch

# Start with a sentence embedding
sentence = "Life is short, eat dessert first"

# We will sort words by their lexiographic ordering
words = {
    word: sort_pos
    for sort_pos, word in enumerate(sorted(sentence.replace(",", " ").split()))
}

# Assign an integer index to each word
sentence_int = torch.tensor([words[s] for s in sentence.replace(",", "").split()])

# Encode inputs into real-vector embedding:
# We use 16-dimensional embedding such that each word will be represented by a
# corresponding 16-dimensional vector.
# Since the sentence consists of 6 words, this will result in a 6x16 dimensional
# embedding.
torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)

# Create the scaled dot-product attention:
# We use three sequences [query, key, value] which are adjusted as model parameters
# during training.
torch.manual_seed(123)
d = embedded_sentence.shape[1]
d_q, d_k, d_v = 24, 24, 28
w_query = torch.rand(d_q, d)
w_key = torch.rand(d_k, d)
w_value = torch.rand(d_v, d)

# Compute the attention-vector for the second input element
x_2 = embedded_sentence[1]
query_2 = w_query.matmul(x_2)
key_2 = w_key.matmul(x_2)
value_2 = w_value.matmul(x_2)

# Generalize to compute the remaining key and vaue elements for all inputs
keys = w_key.matmul(embedded_sentence.T).T
values = w_value.matmul(embedded_sentence.T).T

# Now we compute the unnormalized attention weights:

# Here is the unnormalized attention weight for the query and 5th input
# element (corresponding to index postition 4):
omega_24 = query_2.dot(keys[4])
print(omega_24)

# And compute the omega values for all input tokens
omega_2 = query_2.matmul(keys.T)
print(omega_2)

# Computing the attention scores:
# Now we normalize the unnormalized weights by applying the softmax function.
# Additionally (1/sqrt(d_k)) is used to scale the unnormalized weights before
# normalizing it through the softmax function
# By scaling with d_k we ensure the euclidean length of the weight vectors will
# be the same magnitude. This helps variations in the resulting attention weights
# which could lead to numerical instability to affect the models ability to converge
import torch.nn.functional as F

attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)
print(attention_weights_2)

# Finally compute the context vector z which is an attention-weighted version of our
# original query input x, including all the other input elements as its context via
# the attention weights
context_vector_2 = attention_weights_2.matmul(values)

print(context_vector_2.shape)
print(context_vector_2)

# Now we use multi-head attention to use the q, k, v matrices to create multiple contexts.
# We could also use cross-attention, used in Stable diffusion, to create a mix or combination
# of different input sequences.
