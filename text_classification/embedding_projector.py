import os

import tensorflow as tf
from tensorboard.plugins import projector

from text_classification.model import BaseTextClassificationModel


class EmbeddingVisualizer:
    """
    Static class for text embedding visualization.
    """
    @staticmethod
    def visualize_embeddings(model: BaseTextClassificationModel, log_dir: str, dataset: tf.data.Dataset):
        data_text = [txt for txts, labels in dataset for txt in txts]
        data_label = [label for txts, labels in dataset for label in labels]

        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            f.write(f"text \t label \n")
            for i, txt in enumerate(data_text):
                f.write(f"{txt.numpy().decode('utf-8')}\t{data_label[i].numpy()}\n")

        weights = tf.Variable(model.get_embedding_layers_output(data_text))
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(log_dir, config)
