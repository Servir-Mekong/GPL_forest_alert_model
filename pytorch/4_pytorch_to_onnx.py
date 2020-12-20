import torch
import ee
import time
import os
import json
from utils.unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net


ee.Initialize()

if __name__ == "__main__":
    
    # Specify  the directory where to save the model
    weights_dir = "C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\model_weights"
    
    # Define a path to the model weights
    model_path = 'C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\model_weights\\R2AttU_Net__0_001.pth'
    
    # Define the name of the model
    model_name = model_path.split('\\')[-1].split('.')[0]
    
    # Load in the model 
    model = R2AttU_Net(img_ch=7, output_ch=1).cuda()
    
    # Load the model
    model.load_state_dict(torch.load(model_path))
    
    # Input to the model
    x = torch.randn(16, 7, 128, 128, requires_grad=True).cuda()
    
    # Creat the ONNX model name
    onnx_model_name = weights_dir + '\\' + model_name + '.onnx'
    
    # Create the input names
    inputs = ['input']
    
    # Outputs to the model
    outputs = ['output']
    
    # Export the model
    torch.onnx.export(model,   # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      onnx_model_name,           # where to save the model (can be a file or file-like object)
                      export_params = True,        # store the trained parameter weights inside the model file
                      opset_version = 12,          # the ONNX version to export the model to
                      do_constant_folding = True,  # whether to execute constant folding for optimization
                      input_names = inputs,   # the model's input names
                      output_names = outputs, # the model's output names
                      dynamic_axes = {'input' : {0 : 'batch_size'},    # variable lenght axes
                                      'output' : {0 : 'batch_size'}},
                      verbose = False)