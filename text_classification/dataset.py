import tensorflow as tf


def get_dataset(dataset_url, data_dir_file, train_dir, batch_size, seed):
    tf.keras.utils.get_file(data_dir_file, dataset_url,
                            untar=True, cache_dir='..',
                            cache_subdir='')
    train_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds
