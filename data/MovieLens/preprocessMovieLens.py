from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import json
import re
import random


class preprocessDBBook(object):
    def create_dataset(self):
        max_sent_len = 40
        max_doc_len = 35
        max_cast_len = 0
        voca = set()
        cast_set = set()
        doc_list = {}
        wordTokenizer = RegexpTokenizer(r'\w+')
        noPeriodSentTokenizer = re.compile(r'<[A-Za-z/]+\s*\/?>|(?<=[a-z])[,.;!](?=[A-Z])|\.{3,}')
        longSentTokenizer = re.compile("[;:—*]")
        commaTokenizer = re.compile(r'[,]')
        file_path = 'movielens_abstract_cast.json'
        with open(file_path, 'r') as jsonfile:
            json_dict = json.load(jsonfile)

        movielens_abstract_cast = json_dict['data']
        doc_lens = []
        sent_lens = []
        for movie in movielens_abstract_cast:
            abstract = movie['abstract']
            cast = movie['cast']
            if abstract == 'NA' or not cast:
                continue

            abstract_tokens = sent_tokenize(abstract)
            sent_list = []
            sent_voc = set()
            sent_max_len = 0
            for s in abstract_tokens:
                tokens = noPeriodSentTokenizer.split(s)
                if len(tokens) < 2:
                    tokens = longSentTokenizer.split(s)
                tk = []
                for t in tokens:
                    if len(t) > max_sent_len:
                        tk += commaTokenizer.split(t)
                    else:
                        tk += t
                tokens = tk
                for ss in tokens:
                    word_seq = wordTokenizer.tokenize(ss)
                    word_seq_s = self.clear_wordseq(word_seq)
                    sent_len = len(word_seq_s)
                    if sent_len > max_sent_len:
                        sent_list.clear()
                        print(s)
                        continue
                    # A sentence at least 5 words
                    if sent_len >= 5:
                        sent_lens.append(sent_len)
                        sent_list.append(word_seq_s)
                        if sent_len > sent_max_len:
                            sent_max_len = sent_len
                        sent_voc |= set(word_seq_s)

            doc_len = len(sent_list)
            if doc_len > 0:
                if doc_len > max_doc_len: max_doc_len = doc_len
                doc_lens.append(doc_len)
                doc_list[int(movie['item_id'])] = {'sent':sent_list, 'cast':movie['cast']}
                voca |= sent_voc
                cast_set |= set(movie['cast'])
                if sent_max_len > max_sent_len: max_sent_len = sent_max_len
                if len(movie['cast']) > max_cast_len: max_cast_len = len(movie['cast'])

        vocabulary = { w:i for i, w in enumerate(voca, 1)}
        cast_dict = {c:i for i, c in enumerate(cast_set, 1)}


        result = pd.DataFrame({"doc_len":doc_lens})["doc_len"].value_counts()
        print(result)

        result = pd.DataFrame({"sent_len": sent_lens})["sent_len"].value_counts()
        print(result)

        dbRating = 'ratings.dat'
        rat_df = pd.read_csv(dbRating, delimiter='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
        rat_df = rat_df[rat_df.item_id.isin(doc_list.keys())]

        user_dict = {u: i for i, u in enumerate(rat_df.user_id.unique())}
        item_dict = {m: i for i, m in enumerate(rat_df.item_id.unique())}

        pd.DataFrame({"Idx":list(user_dict.values()),
                      "RawId":list(user_dict.keys())}).to_csv(path_or_buf="movielens_userDict.csv", index=False)
        pd.DataFrame({"Idx": list(item_dict.values()),
                      "RawId": list(item_dict.keys())}).to_csv(path_or_buf="movielens_itemDict.csv", index=False)

        rat_df.user_id = [user_dict[uid] for uid in rat_df.user_id]
        rat_df.item_id = [item_dict[iid] for iid in rat_df.item_id]
        rat_df.rating = 1
        rat_df.to_csv(path_or_buf="movielens_dataset.csv", index=False)

        doc_list = {item_dict[id]:doc_cast for id,doc_cast in doc_list.items() if item_dict.get(id) is not None}
        for v in doc_list.values():
            v['cast'] = [cast_dict[c] for c in  v['cast']]

        vocabulary_file = "movielens_vocab.json"
        with open(vocabulary_file, 'w') as jsonfile:
            json.dump(vocabulary, jsonfile)

        cast_file = "movielens_cast.json"
        with open(cast_file, 'w') as jsonfile:
            json.dump(cast_dict, jsonfile)

        processed_file = "movielens_doc_cast.json"
        with open(processed_file, 'w') as jsonfile:
            json.dump({'max_sent_len':max_sent_len, 'max_doc_len':max_doc_len,
                       'max_cast_len':max_cast_len, 'doc_info':doc_list}, jsonfile)

    def clear_wordseq(self, wordseq):
        # Remove words less than 3 letters
        wordseq = [word for word in wordseq if len(word) >= 3 and len(word) <= 15]
        # Remove numbers
        wordseq = [word for word in wordseq if not word.isnumeric() and word.isalnum()]
        # Lowercase all words (default_stopwords are lowercase too)
        wordseq = [word.lower() for word in wordseq]
        return wordseq


    def create_train_test_set(self, prop=0.8, negprop=10):
        rat_df = pd.read_csv("movielens_dataset.csv")
        rndidx = np.random.permutation(rat_df.shape[0])
        split = int(rat_df.shape[0] * prop)
        train_idx = rndidx[:split]
        test_idx = rndidx[split:]
        train_data = rat_df.iloc[train_idx]
        test_data = rat_df.iloc[test_idx]
        test_users = test_data.user_id.unique()
        docs = list(rat_df.item_id.unique())
        neg_list = [test_data]
        for u in test_users:
            udata = test_data[test_data.user_id==u]
            if len(udata) > 0:
                samples = random.sample(docs, min(len(docs), len(udata)*(negprop+1)))
                samples = [s for s in samples if s not in udata.item_id]
                datalen = int(min(len(docs), len(udata)*negprop))
                neg_list.append(pd.DataFrame({'user_id': u, 'item_id':samples[:datalen], 'rating':-1, 'timestamp':0}))
        test_data = pd.concat(neg_list)

        train_data.to_csv(path_or_buf="Train%.2f.csv" % prop, index=False)
        test_data.to_csv(path_or_buf="Test%.2f.csv" % prop, index=False)


    def create_coldstart_set(self, prop=0.2, negprop=10):
        rat_df = pd.read_csv("movielens_dataset.csv")
        item_set = list(rat_df.item_id.unique())
        cs_items = random.sample(item_set, int(len(item_set) * prop))
        test_idx = rat_df.item_id.isin(cs_items)
        test_data = rat_df[test_idx]
        train_data = rat_df[~test_idx]
        test_users = test_data.user_id.unique()
        neg_list = [test_data]
        docs = list(test_data.item_id.unique())
        for u in test_users:
            udata = test_data[test_data.user_id == u]
            if len(udata) > 0:
                samples = random.sample(docs, min(len(docs), len(udata)*(negprop+1)))
                samples = [s for s in samples if s not in udata.item_id]
                datalen = int(min(len(docs), len(udata) * negprop))
                neg_list.append(
                    pd.DataFrame({'user_id': u, 'item_id': samples[:datalen], 'rating': -1, 'timestamp':0}))
        test_data = pd.concat(neg_list)
        train_data.to_csv(path_or_buf="Train_CS%.2f.csv" % prop, index=False)
        test_data.to_csv(path_or_buf="Test_CS%.2f.csv" % prop, index=False)


prepro = preprocessDBBook()
prepro.create_dataset()
prepro.create_train_test_set(0.8)
prepro.create_train_test_set(0.5)
prepro.create_coldstart_set(0.1)