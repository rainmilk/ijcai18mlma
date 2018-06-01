import numpy as np
import math
import random
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from metric.ranking import mrrAtK, precisionAtK, nDCG, avgPrecisionAtK, recallAtK
from test.MovieLens.loadData import *


train_cast_set = set()
for d in train_doc_set:
    train_cast_set |= set(doc_cast_list[str(d)]['cast'])

doc_all_dict = {}
cast_all_dict = {}
unseen_cast = set()
for k,v in doc_cast_list.items():
    padding_doc = np.zeros([max_doc_len, max_sent_len], dtype='int32')
    padding_cast = np.zeros(max_cast_len, dtype='int32')
    nb_sent = len(v)
    for i,words in enumerate(v['sent']):
        padding_doc[(-nb_sent+i), -len(words):] = [vocab_dict[w] for w in words]
    doc_all_dict[int(k)] = padding_doc

    cast = v['cast']
    # cast = [c for c in cast if c in train_cast_set]
    if cast: padding_cast[-len(cast):] = cast
    cast_all_dict[int(k)] = padding_cast

unseen_cast = list(unseen_cast)

train_user_dict = {u:train_set[train_set.user_id == u].item_id.values
                   for u in set(train_set.user_id)}
doc_id_list = list(doc_all_dict.keys())
train_user_list = list(train_user_dict.keys())

def make_doc_cast_batch(ind):
    docs = [doc_all_dict[i] for i in ind]
    cast = [cast_all_dict[i] for i in ind]
    return np.stack(docs), np.stack(cast)

# Build Model
actor_emb_len = 100
from model.movieattrac import MovieAttractionModel
from model.losses import max_margin_loss, infinite_margin_loss

model = MovieAttractionModel(user_size=len(user_dict), movie_size=len(doc_cast_list),
                             cast_size = len(cast_dict), voca_size=len(vocab_dict),
                             max_sent_len=max_sent_len, max_doc_len=max_doc_len,
                             max_cast_len=max_cast_len,
                             word_embed_len=embed_len, sent_embed_len=embed_len*2, doc_embed_len=embed_len*4,
                             actor_embed_len=actor_emb_len, cast_embed_len=actor_emb_len*2,
                             dropout=0.2, score_activation=None, hid_activation='tanh',
                             alpha_att_word=1, alpha_att_sent=1, alpha_att_cast=1,
                             alpha_word=64, alpha_sent=16, alpha_cast=4,
                             user_reg=l2(1e-6), actor_reg=l2(1e-6),
                             use_story=False, use_cast=True)
if model.use_story:
    model.word_embedding.trainable = False
    model.word_embedding.set_weights([embeddings])
model.contrast_model.compile(loss=max_margin_loss, optimizer='Adam')


def test():
    nb_test = len(test_set)
    pred_data = np.zeros(nb_test)
    nb_batches = math.ceil(nb_test / batch_size)
    for i in range(nb_batches):
        test_idx = np.arange(i * batch_size, min(len(pred_data), (i + 1) * batch_size))
        batch_data = test_set.iloc[test_idx, :]
        doc_input, cast_input = make_doc_cast_batch(batch_data.item_id)
        scores = model.score_model.predict_on_batch([batch_data.user_id, doc_input, cast_input])
        pred_data[test_idx] = np.squeeze(scores, axis=-1)

    atK = range(5, 51, 5)
    auclist = []
    precision = np.zeros(len(atK))
    ap = np.zeros(len(atK))
    mrr = np.zeros(len(atK))
    ndcg = np.zeros(len(atK))
    recall = np.zeros(len(atK))
    for u in test_user_set:
        test_idx = test_set.user_id == u
        true_data = test_set[test_idx]
        tu_data = pred_data[test_idx]
        pIdx = true_data.rating > 0
        pScore = tu_data[pIdx]
        nScore = tu_data[~pIdx]
        ap += avgPrecisionAtK(pScore, nScore, atK)
        mrr += mrrAtK(pScore, nScore, atK)
        ndcg += nDCG(pScore, nScore, atK)
        precision += precisionAtK(pScore, nScore, atK)
        recall += recallAtK(pScore, nScore, atK)
        auclist.append(roc_auc_score(true_data.rating, tu_data))

    nTest = len(test_user_set)
    print("AUC: %g" % (np.mean(auclist)))
    print("MAP:")
    print(ap / nTest)
    print("nDCG:")
    print(ndcg / nTest)
    print("MRR:")
    print(mrr / nTest)
    print("PR")
    print(precision / nTest)
    print("RECALL")
    print(recall / nTest)

def train_with_test(train_set, testepoch=2):
    for epoch in range(max_epoch):
        train_set = shuffle(train_set)
        loss = 0
        for i in range(nb_batch):
            # poslist = train_user_list[i*batch_size:min(len(train_user_list), (i+1)*batch_size)]
            # pos_input = [random.choice(train_user_dict[u]) for u in poslist]
            poslist = train_set.iloc[i*batch_size:min(nb_samples,(i+1)*batch_size), :]
            pos_doc_input, pos_cast_input = make_doc_cast_batch(poslist.item_id)
            neg_input = []
            for u in poslist.user_id:
                d = random.choice(doc_id_list)
                while d in train_user_dict[u]:
                    d = random.choice(doc_id_list)
                neg_input.append(d)
            neg_doc_input, neg_cast_input = make_doc_cast_batch(neg_input)
            loss += model.contrast_model.train_on_batch(x=[poslist.user_id,
                                                          pos_doc_input, pos_cast_input,
                                                          neg_doc_input, neg_cast_input],
                                                       y=margin*np.ones([len(poslist), 2]))
            if (i+1) % 100 == 0:
                print("Epoch: %d/%d, Batch: %d/%d" % (epoch + 1, max_epoch, i+1, nb_batch))

        print("Epoch: %d/%d, Loss: %g"%(epoch + 1, max_epoch, loss))
        if (epoch + 1) % testepoch == 0 or (epoch + 1) == max_epoch:
            test()
            model.contrast_model.save("MovieLens_Model.hd5")


margin = 20
batch_size = 500
max_epoch = 10
nb_samples = train_set.shape[0]
nb_batch = math.ceil(nb_samples / batch_size)
nb_ts_samples = test_set.shape[0]
nb_ts_batch = math.ceil(nb_ts_samples / batch_size)

if model.use_story and not model.use_cast:
    print('Training story model')
elif model.use_cast and not model.use_story:
    print('Training cast model')

train_with_test(train_set, 2)

if model.use_cast:
    actor_embed = model.actor_embedding.get_weights()

model.use_story = True
model.rebuild()
model.word_embedding.trainable = False
model.contrast_model.compile(loss=max_margin_loss, optimizer='Adam')
model.word_embedding.set_weights([embeddings])
if model.use_cast: model.actor_embedding.set_weights(actor_embed)

max_epoch = 20
print('Training full model')
train_with_test(train_set, 2)