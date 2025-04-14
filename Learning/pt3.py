import numpy as np

# Lists are vectors
inputs=[1,2,3,2.5]

# List of Lists are Matrix (2D Vector)
# List of list of lists is 3D Vector!!!
weights=[[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]

biases=[2,3,0.5] 

output=np.dot(weights,inputs)+biases
print(output)