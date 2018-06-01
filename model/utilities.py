import numpy as np


def load_glove_embedding(embedding_path, dict, embeding_len):
    embeddings_index = {}
    f = open(embedding_path, 'r', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
    f.close()

    embedding_matrix = np.random.random([len(dict) + 1, embeding_len])
    embedding_matrix.astype(dtype=np.float32)
    for word, i in dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix