import torch
import ee
import time
import os
import json

ee.Initialize()

if __name__ == "__main__":
    
    # Specify cloud storage bucket to save data too
    BUCKET = 'ee-rsqa'
    
    TF_DIR = ""
    
    # Make a dictionary that maps Earth Engine outputs and inputs to 
    # AI Platform inputs and outputs, respectively.
    
    input_dict = "'" + json.dumps({input_name: "array"}) + "'"
    output_dict = "'" + json.dumps({output_name: 'qa'}) + "'"
    print(input_dict)
    print(output_dict)
    
    # Put the EEified model next to the trained model directory.
    EEIFIED_DIR = 'gs://{}/eeified_{}/'.format(BUCKET,MODEL_NAME)
    
    # change to your specific project
    PROJECT = 'ee-sandbox'

    # # You need to set the project before using the model prepare command.
    os.system("earthengine set_project {}").format(PROJECT)
    
    in_str = "earthengine --no-use_cloud_api model prepare \
        --source_dir {TF_DIR} \
        --dest_dir {EEIFIED_DIR} \
        --input {input_dict} \
        --output {output_dict}".format(TF_DIR, EEIFIED_DIR, input_dict, output_dict)
    os.system(in_str)
        
    MODEL_NAME = 'lc8_qa_model'
    VERSION_NAME = 'v' + str(int(time.time()))
    print('Creating version: ' + VERSION_NAME)
    
    in_str = ":gcloud ai-platform versions create {VERSION_NAME} \
      --project {PROJECT} \
      --model {MODEL_NAME} \
      --origin {EEIFIED_DIR}/ \
      --runtime-version=1.14 \
      --framework 'TENSORFLOW' \
      --python-version=3.5
      os.system(in_str)