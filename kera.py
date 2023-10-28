from keras.src.layers import Dense
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from keras import  Sequential

# load du lieu va chia train,val va test
dataset = loadtxt('data.csv',delimiter=',')
x = dataset[:, 0:8]
y = dataset[:,8]
x_train_val,x_test,y_train_val,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train_val,y_train_val,test_size=0.2)

# tiến hành tọa model và lựa chọn các hàm train sao cho phuf hợp

# model = Sequential()
# model.add(Dense(16,input_dim=8,activation="relu"))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
# model.fit(x_train,y_train,epochs=100,batch_size=8,validation_data=[x_val,y_val])
# model.save('mymodel.h5')
