from keras.src.layers import Dense
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from keras import  Sequential
import  numpy as np
from keras.models import  load_model
import  kera

model = load_model("mymodel.h5")
loss,acc = model.evaluate(kera.x_test,kera.y_test)

print('loss: ',loss)
print('acc: ',acc)
#lay mot gia tri bất kì vào thử model
x_new = [10,129,62,36,0,41.2,0.441,38]
y_new =1
# luu y phai convert ve dang ma keras có thể dùng được
x_new = np.expand_dims(x_new,axis=0)

predict = model.predict(x_new)
print("ket qua chinh xác: ",y_new)
print("ket qua du doan cua model: ",predict)
