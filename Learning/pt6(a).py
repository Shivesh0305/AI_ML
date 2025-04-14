# import math
import numpy as np
import nnfs
nnfs.init()

layer_output=[[4.8,1.21,2.385],[8.9,-1.81,0.2],[1.41,1.051,0.026]]
# E=math.e

# exp_values=[]
# for i in layer_output:
#     exp_values.append(E**i)

# print(exp_values)

#Instead of the above steps we can use numpy!
exp_values=np.exp(layer_output)

# norm_base=sum(exp_values)
# norm_values=[]

# for i in exp_values:
#     norm_values.append(i/norm_base)

#Simillarly instead of the above loop we can do-
# norm_values=exp_values/np.sum(exp_values)

# for a batch of values we do this-
#np.sum(layer_output,axis=1,keepdims=True)

norm_values=exp_values/np.sum(layer_output,axis=1,keepdims=True)

print(norm_values)