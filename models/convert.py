import onnx
import numpy as np
from onnx_tf.backend import prepare

model = onnx.load('./test.onnx')
tf_rep = prepare(model) 

img = np.zeros((96,96,3))
output = tf_rep.run(img.reshape([1, 3,96,96]))

print("outpu mat: \n",output)
print("The digit is classified as ", np.argmax(output))

import tensorflow as tf
with tf.Session() as persisted_sess:
    print("load graph")
    persisted_sess.graph.as_default()
    tf.import_graph_def(tf_rep.predict_net.graph.as_graph_def(), name='')
    inp = persisted_sess.graph.get_tensor_by_name(
        tf_rep.predict_net.tensor_dict[tf_rep.predict_net.external_input[0]].name
    )
    out = persisted_sess.graph.get_tensor_by_name(
        tf_rep.predict_net.tensor_dict[tf_rep.predict_net.external_output[0]].name
    )
    res = persisted_sess.run(out, {inp: img.reshape([1, 3,96,96])})
    print(res)
    print("The digit is classified as ",np.argmax(res))

tf_rep.export_graph('tf.pb')