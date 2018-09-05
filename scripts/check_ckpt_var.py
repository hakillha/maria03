from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path = os.path.join('inf_04', 'model-10000')
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
	if key.startswith('id_') and not key.startswith('id_head/g'):
	    print("tensor_name: ", key)
    # print(reader.get_tensor(key)) # Remove this is you want to print only variable names