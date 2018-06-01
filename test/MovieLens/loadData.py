import json
import pandas as pd
import model.utilities as ut

relative_path = "../../data/MovieLens/"
embedding_path = "../../data/"
embed_len = 200
with open(relative_path + "movielens_vocab.json", 'r') as jsonfile:
    vocab_dict = json.load(jsonfile)

with open(relative_path + "movielens_doc_cast.json", 'r') as jsonfile:
    article_seq = json.load(jsonfile)

with open(relative_path + "movielens_cast.json", 'r') as jsonfile:
    cast_dict = json.load(jsonfile)

negclass_label = 0
train_set = pd.read_csv(relative_path + "Train_CS0.10.csv")
test_set = pd.read_csv(relative_path + "Test_CS0.10.csv")
if negclass_label == 0: test_set.loc[test_set.rating <= 0, 'rating'] = negclass_label

train_user_set = set(train_set.user_id)
test_user_set = set(test_set.user_id)
user_dict = train_user_set | test_user_set
train_doc_set = set(train_set.item_id)

max_doc_len = article_seq['max_doc_len']
max_sent_len = article_seq['max_sent_len']
max_cast_len = article_seq['max_cast_len']
doc_cast_list = article_seq['doc_info']
embeddings = ut.load_glove_embedding(embedding_path + "glove.6B.%dd.txt"%embed_len, vocab_dict, embed_len)