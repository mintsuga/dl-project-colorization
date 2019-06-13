import tensorflow as tf

def write_log(callback, names, logs, batch_cnt):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_cnt)
        callback.writer.flush()
