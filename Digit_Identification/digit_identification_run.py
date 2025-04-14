import os, cv2, numpy as np, matplotlib.pyplot as plt, tensorflow as tf

# Runs the trained model
model=tf.keras.models.load_model('handwritten.keras')

img_num=1
while os.path.isfile(f"digits/{img_num}.png"):
    try:
        img=cv2.imread(f"digits/{img_num}.png")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"The number is {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        img_num+=1