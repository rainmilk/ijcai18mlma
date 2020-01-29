from keras.layers import Input, add, dot, multiply, RepeatVector, Dropout, Activation, \
    Lambda, Dense, Embedding, concatenate
from keras.models import Model
from .scorelayer import ScoreLayer
import keras.backend as K
from .attentionlayer import BatchAttention, BiasedAttention, Attention, AverageAttention
from .timedistributed import TimeDistributedMultiInput


class MovieAttractionModel(object):
    def __init__(self,
                 user_size, movie_size, cast_size, voca_size, max_sent_len, max_doc_len, max_cast_len,
                 word_embed_len=100, sent_embed_len=200, doc_embed_len = 400,
                 actor_embed_len=200, cast_embed_len=200,
                 hid_activation='tanh', score_activation=None,
                 alpha_att_word=1, alpha_att_sent=1, alpha_att_cast=1,
                 alpha_word=32, alpha_sent=4, alpha_cast=4,
                 dropout=0, user_reg=None, actor_reg=None,
                 use_story=True, use_cast=True):
        self.user_size = user_size
        self.movie_size = movie_size
        self.cast_size = cast_size
        self.voca_size = voca_size
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.max_cast_len = max_cast_len
        self.word_embed_len = word_embed_len
        self.sent_embed_len = sent_embed_len
        self.doc_embed_len = doc_embed_len
        self.actor_embed_len = actor_embed_len
        self.cast_embed_len = cast_embed_len
        self.dropout = dropout
        self.alpha_att_word = alpha_att_word
        self.alpha_att_sent = alpha_att_sent
        self.alpha_att_cast = alpha_att_cast
        self.alpha_word = alpha_word
        self.alpha_sent = alpha_sent
        self.alpha_cast = alpha_cast
        self.user_reg = user_reg
        self.actor_reg = actor_reg
        self.hid_activation = hid_activation
        self.score_activation = score_activation
        self.use_story = use_story
        self.use_cast = use_cast
        self.rebuild()

    def rebuild(self):
        self.sent_encoder, self.score_model, self.contrast_model = self._build()

    def _build(self):
        u_input = Input(shape=(1,), dtype='int32', name='user_index')
        word_input = Input(shape=(self.max_sent_len,), dtype='int32', name='word_index')
        sent_input = Input(shape=(self.max_doc_len, self.max_sent_len), dtype='int32', name='sentence_index')
        sent_encoder = None
        if self.use_story:
            self.word_embedding = Embedding(self.voca_size + 1, self.word_embed_len, mask_zero=True, name='word_embedding')
            # Word-level model for sentence encoding
            w_embed = self.word_embedding(word_input)
            w_user_embed = Embedding(self.user_size, K.int_shape(w_embed)[-1],
                                     name='user_word_embedding', embeddings_regularizer=self.user_reg)(u_input)
            sent_embed_att = BatchAttention(name='word_attention', alpha=self.alpha_att_word)([w_embed, w_user_embed])
            sent_embed_avg = Attention(name='word_avg_attention', alpha=self.alpha_word)(w_embed)
            sent_embed = concatenate([sent_embed_att, sent_embed_avg])
            sent_encoder = Model([u_input, word_input], sent_embed, name='model_sent_encoder')

            # Sentence-level model for document encoding
            rep_u_input = RepeatVector(self.max_doc_len)(u_input)
            s_embed = TimeDistributedMultiInput(sent_encoder)([rep_u_input, sent_input])
            if self.dropout > 0: s_embed = Dropout(rate=self.dropout)(s_embed)
            s_embed = Dense(self.sent_embed_len, activation=self.hid_activation)(s_embed)

            s_user_embed = Embedding(self.user_size, K.int_shape(s_embed)[-1],
                                     name='user_sent_embedding', embeddings_regularizer=self.user_reg)(u_input)
            doc_embed_att = BatchAttention(name='sent_attention', alpha=self.alpha_att_sent, keepdims=True)([s_embed, s_user_embed])
            doc_embed_avg = Attention(name='sent_avg_attention', alpha=self.alpha_sent, keepdims=True)(s_embed)
            doc_embed = concatenate([doc_embed_att, doc_embed_avg])
            if self.dropout > 0: doc_embed = Dropout(rate=self.dropout)(doc_embed)
            doc_embed = Dense(self.doc_embed_len, activation=self.hid_activation)(doc_embed)

        # cast model
        a_input = Input(shape=(self.max_cast_len,), dtype='int32', name='cast_index')
        if self.use_cast:
            self.actor_embedding = Embedding(self.cast_size + 1, self.actor_embed_len, mask_zero=True,
                                       name='cast_embedding', embeddings_regularizer=self.actor_reg)

            actor_embed = self.actor_embedding(a_input)
            a_user_embed = Embedding(self.user_size, K.int_shape(actor_embed)[-1],
                                     name='user_cast_embedding', embeddings_regularizer=self.user_reg)(u_input)
            cast_embedding_att = BatchAttention(name='cast_attention', alpha=self.alpha_att_cast, keepdims=True)([actor_embed, a_user_embed])
            cast_embedding_avg = Attention(name='cast_avg_attention', alpha=self.alpha_cast, keepdims=True)(actor_embed)
            cast_embedding = concatenate([cast_embedding_att, cast_embedding_avg])
            cast_embedding = Dense(self.cast_embed_len, activation=self.hid_activation)(cast_embedding)

        if self.use_story and self.use_cast:
            joint_embedding = concatenate([doc_embed, cast_embedding], name='concat_embedding')
        elif self.use_cast:
            joint_embedding = cast_embedding
        else:
            joint_embedding = doc_embed

        # joint_embedding = Dense(self.cast_embed_len + self.doc_embed_len, activation=self.hid_activation)(joint_embedding)

        # top-level score
        score_user_embed = Embedding(self.user_size, K.int_shape(joint_embedding)[-1],
                                     name='user_score_embedding', embeddings_regularizer=self.user_reg)(u_input)

        # scores = ScoreLayer(name='score')([joint_embedding, score_user_embed])
        scores = dot([joint_embedding, score_user_embed], name='score', axes=-1)
        scores = Lambda(lambda x : K.squeeze(x, axis=-1))(scores)

        if self.score_activation is not None: scores = Activation(self.score_activation)(scores)

        score_model = Model([u_input, sent_input, a_input], scores, name='score_model')

        # Contrastive model with positive doc and negative doc
        pos_doc_input = Input(shape=(self.max_doc_len, self.max_sent_len), dtype='int32', name="pos_doc_input")
        neg_doc_input = Input(shape=(self.max_doc_len, self.max_sent_len), dtype='int32', name="neg_doc_input")

        pos_cast_input = Input(shape=(self.max_cast_len,), dtype='int32', name="pos_cast_input")
        neg_cast_input = Input(shape=(self.max_cast_len,), dtype='int32', name="neg_cast_input")

        pos_score_doc = score_model([u_input, pos_doc_input, pos_cast_input])
        neg_score_doc = score_model([u_input, neg_doc_input, neg_cast_input])
        concat_score = concatenate([pos_score_doc, neg_score_doc])

        contrastive_model = Model(inputs=[u_input,
                                          pos_doc_input, pos_cast_input,
                                          neg_doc_input, neg_cast_input],
                                  outputs=[concat_score], name='contrastive_model')

        return sent_encoder, score_model, contrastive_model



