import onnx

from onnx_tf.backend import prepare

# Specify  the directory where to save the model
weights_dir = "C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\model_weights"

# Define a path to the model weights
input_path = 'C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\model_weights\\U_Net__0_001.onnx'

# Define a path to the model weights
output_path = 'C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\model_weights\\U_Net__0_001'

onnx_model = onnx.load(input_path)  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph(output_path)  # export the model