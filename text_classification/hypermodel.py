import keras_tuner
import os
import tensorflow as tf
from abc import abstractmethod

from text_classification.embedding_projector import EmbeddingVisualizer
from text_classification.model import StandardTextClassificationModel, TFHubEmbeddingTextClassificationModel, \
    BertTextClassificationModel, BaseTextClassificationModel


class BaseTextClassificationHyperModel(keras_tuner.HyperModel):
    """
    Base class for tuning text classification models.
    """

    def __init__(self, train_ds, n_output_units, loss, metric):
        self.train_ds = train_ds
        self.n_output_units = n_output_units
        self.loss = loss
        self.metric = metric

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, **kwargs)

    @abstractmethod
    def build(self, hp) -> BaseTextClassificationModel:
        pass


class StandardTextClassificationHyperModel(BaseTextClassificationHyperModel):
    def build(self, hp) -> StandardTextClassificationModel:
        dense_units = hp.Choice('units', [8, 16, 32])
        learning_rate = hp.Float("learning_rate", 1e-6, 1e-2, sampling="log", default=1e-3)
        model = StandardTextClassificationModel(
            train_ds=self.train_ds, n_output_units=self.n_output_units, embedding_dim=dense_units)
        model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=self.metric)
        return model


class TFHubEmbeddingTextClassificationHyperModel(BaseTextClassificationHyperModel):
    def build(self, hp) -> TFHubEmbeddingTextClassificationModel:
        learning_rate = hp.Float("learning_rate", 1e-6, 1e-2, sampling="log", default=1e-3)
        trainable = hp.Boolean("trainable")
        tf_hub_url = hp.Choice('tf_hub_url', ["https://tfhub.dev/google/nnlm-en-dim50/2",
                                              "https://tfhub.dev/google/universal-sentence-encoder/4"])
        model = TFHubEmbeddingTextClassificationModel(
            train_ds=self.train_ds, tf_hub_url=tf_hub_url,
            n_output_units=self.n_output_units, trainable=trainable)
        model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=self.metric)
        return model


class BertTextClassificationHyperModel(BaseTextClassificationHyperModel):
    def build(self, hp) -> BertTextClassificationModel:
        learning_rate = hp.Float("learning_rate", 1e-6, 1e-2, sampling="log", default=1e-3)
        trainable = hp.Boolean("trainable")

        model = BertTextClassificationModel(
            train_ds=self.train_ds, n_output_units=self.n_output_units, trainable=trainable)
        model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=self.metric)
        return model


def tune_model(hypermodel: BaseTextClassificationHyperModel, log_dir: str, objective: str, train_ds: tf.data.Dataset,
               val_ds: tf.data.Dataset, epochs: int, max_trials: int,
               executions_per_trial: int) -> keras_tuner.Tuner:
    tuner = keras_tuner.RandomSearch(objective=objective,
                                     hypermodel=hypermodel,
                                     max_trials=max_trials,
                                     executions_per_trial=executions_per_trial,
                                     overwrite=True,
                                     directory=log_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=objective,
        patience=1,
        verbose=1,
        restore_best_weights=True)

    with tf.device('/GPU:0'):
        history = tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[tensorboard_callback, early_stopping])
    tuner.results_summary()

    return tuner


def get_best_model(hypermodel: BaseTextClassificationHyperModel, log_dir: str, objective: str,
                   train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int, max_trials: int,
                   executions_per_trial: int) -> BaseTextClassificationModel:
    tuner = tune_model(hypermodel, log_dir, objective, train_ds, val_ds, epochs, max_trials, executions_per_trial)
    best_model = tuner.get_best_models(num_models=1)[0]
    EmbeddingVisualizer.visualize_embeddings(best_model, log_dir, val_ds)
    return best_model
