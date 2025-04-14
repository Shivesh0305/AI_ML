import numpy as np

inputs=[[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]

weights=[[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]

biases=[2,3,0.5] 

#output=np.dot(weights,inputs)+biases 
#This will throw shape error!! (3X4).(3X4)

#Instead we will multiply with transpose matrix to get the output
output=np.dot(inputs,np.array(weights).T)+biases  
print(output)
