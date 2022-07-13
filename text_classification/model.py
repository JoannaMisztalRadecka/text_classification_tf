import tensorflow as tf
import tensorflow_hub as hub
from abc import abstractmethod


class BaseTextClassificationModel(tf.keras.Model):
    """
    Abstract class for text classification and embedding visualization.
    """

    def __init__(self, dense_units=8, reg_coef=0.0001, dropout_rate=0.3,
                 n_output_units=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = tf.keras.layers.Dense(dense_units, activation='relu', name='dense',
                                           kernel_regularizer=tf.keras.regularizers.l2(reg_coef))
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name='dropout')
        self.classifier = tf.keras.layers.Dense(n_output_units, name='output_pred')

    def call(self, inputs):
        x = self.get_embedding_layers_output(inputs)
        x = self.dense(x)
        x = self.dropout(x)
        return self.classifier(x)

    @abstractmethod
    def get_embedding_layers_output(self, inputs):
        pass


class StandardTextClassificationModel(BaseTextClassificationModel):
    """
    Standard text classification model that builds
     task-specific text embeddings during the training.
    """

    def __init__(self, train_ds: tf.data.Dataset, max_features=10000,
                 sequence_length=256, embedding_dim=32, n_output_units=1,
                 *args, **kwargs):
        super().__init__(n_output_units=n_output_units, *args, **kwargs)
        self.max_features = max_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.vectorize_layer = self._get_vectorizer(train_ds)
        self.embedding = tf.keras.layers.Embedding(self.max_features + 1,
                                                   self.embedding_dim, name='embedding')
        self.pooling = tf.keras.layers.GlobalAveragePooling1D(name='pooling')

    def _get_vectorizer(self, train_ds):
        vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=self.max_features,
                                                            output_mode='int',
                                                            output_sequence_length=self.sequence_length,
                                                            name='vectorizer')
        vectorize_layer.adapt(train_ds.map(lambda txt, label: txt))
        return vectorize_layer

    def get_embedding_layers_output(self, inputs):
        x = self.vectorize_layer(inputs)
        x = self.embedding(x)
        x = self.pooling(x)
        return x


class TFHubEmbeddingTextClassificationModel(BaseTextClassificationModel):
    """
   Binary text classification model that uses transfer learning with embeddings downloaded from tensorflow-hub.
   """

    def __init__(self, train_ds: tf.data.Dataset, tf_hub_url: str,
                 n_output_units=1, trainable=True, *args, **kwargs):
        super().__init__(n_output_units=n_output_units, *args, **kwargs)
        self.text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='txt')
        self.vectorize_layer = hub.KerasLayer(tf_hub_url, input_shape=[],
                                              dtype=tf.string, trainable=trainable, name='vectorizer')

    def get_embedding_layers_output(self, inputs):
        return self.vectorize_layer(inputs)


class BertTextClassificationModel(BaseTextClassificationModel):
    """
    Text classification model that uses transfer learning with Bert
     sentence embeddings (Bert model downloaded from tensorflow-hub).
    """

    def __init__(self, train_ds: tf.data.Dataset, n_output_units=1,
                 trainable=True, *args, **kwargs):
        super().__init__(n_output_units=n_output_units, *args, **kwargs)
        self.text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='txt')
        self.vectorize_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", name='vectorizer')
        self.encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",
            trainable=trainable, name='embedding')

    def get_embedding_layers_output(self, inputs):
        encoder_inputs = self.vectorize_layer(inputs)
        outputs = self.encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]
        return pooled_output
