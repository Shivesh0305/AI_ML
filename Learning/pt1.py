inputs=[1,2,3,2.5]

weights1=[0.2,0.8,-0.5,1]
weights2=[0.5,-0.91,0.26,-0.5]
weights3=[-0.26,-0.27,0.17,0.87]

bias1=2 
bias2=3 
bias3=0.5 

output=[2,3,0.5]
for i in range(len(inputs)):
    output[0]+=(inputs[i]*weights1[i])
for i in range(len(inputs)):
    output[1]+=(inputs[i]*weights2[i])
for i in range(len(inputs)):
    output[2]+=(inputs[i]*weights3[i])

print(output)