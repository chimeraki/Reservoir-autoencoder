from numpy import *
import pickle, gzip
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import scipy.linalg as linalg
import scipy
import itertools
from scipy import signal
from scipy import signal as sig
import random
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib.cm as cm
import math
import matplotlib as mpl
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error as mse
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mdp
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from keras import regularizers

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
random.seed(44)



res=[]
cd=1.0
mix=0.5#sp*0.1

def rKN(x, fx, hs):
    xk = []

    k1=fx(x)*hs

    xk=x + k1*0.5
    k2=fx(xk)*hs

    xk = x + k2*0.5
    k3=fx(xk)*hs

    xk = x + k3
    k4=fx(xk)*hs

    x = x + (k1 + 2*(k2 + k3) + k4)/6
    return x

param=1.2
def lorenz2(inp, s=10*param, r=28*param, b=2.667*param):
    x,y,z=inp[0],inp[1],inp[2]
    x_dot = (s*(y - x))
    y_dot = (r*x - y - x*z)
    z_dot = (x*y - b*z)
    return np.array([x_dot, y_dot, z_dot])


def lorenz(inp, s=10, r=28, b=2.667):
    x,y,z=inp[0],inp[1],inp[2]
    x_dot = cd*(s*(y - x))
    y_dot = cd*(r*x - y - x*z)
    z_dot = cd*(x*y - b*z)
    return np.array([x_dot, y_dot, z_dot])


dt=0.01
x=np.arange(0,1000,0.01)
inp1=[]
inp2=[]
a=[4,3,10]
b=[1,4,2]
start1=300 #input signal transients
for i in range(len(x)):
    inp1.append(a[0])      #use only the z signal for training
    inp2.append(b[0])
    a=rKN(a,lorenz,dt)
    b=rKN(b,lorenz2,dt)
max1=max(inp1)
max2=max(inp2)
inp1=(np.array(inp1)[start1:]-np.mean(inp1))/np.std(inp2[2000:3000])
inp2=(np.array(inp2)[start1:]-np.mean(inp2))/np.std(inp2[2000:3000])


data = inp1
data_mix = sqrt(mix)*inp1+sqrt(1-mix)*inp2


# load the data
trainLen =10000
testLen = 2000
initLen = 100

# generate the ESN reservoir
inSize = outSize = 1
resSize = 300
a = 0.3 # leaking rate
den=0.13  # this is equivalent to the input lorenz signal having standard deviation of one
Win = (np.random.rand(resSize,1+inSize)-0.5)*den
W = np.random.rand(resSize,resSize)-0.5 #(scipy.sparse.rand(resSize, resSize, density=0.05, format='coo', random_state=100).A-0.5)
# normalizing and setting spectral radius:
print ('Computing spectral radius...'),
rhoW = max(abs(linalg.eig(W)[0]))
print ('done.')
sr= 0.9
W *=sr/ rhoW

# allocated memory for the design (collected states) matrix
X = zeros((1+inSize+resSize,trainLen-initLen))
# set the corresponding target matrix directly
Yt = data[None,initLen+1:trainLen+1] 

# run the reservoir with the data and collect X
x = zeros((resSize,1))
for t in range(trainLen):
    u = data_mix[t]
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    if t >= initLen:
        X[:,t-initLen] = vstack((1,u,x))[:,0]





# run the reservoir on training data and collect X_out
# because x is initialized with training data and we continue from there.
Y = zeros((outSize,testLen))
X_out = zeros((1+inSize+resSize,testLen-initLen))
u = data_mix[trainLen]
for t in range(testLen):
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    if t >= initLen:
        X_out[:,t-initLen] = vstack((1,u,x))[:,0]
    u = data_mix[trainLen+t+1]



##################### autoencoder ---- here we take the reservoir activity and use an autoencoder to get a lower dimensional representation of it
X=X.T
X_out=X_out.T
spacing=1
ae_inp_time=10
X_tr = zeros((int((shape(X)[0]-ae_inp_time)/spacing),shape(X)[1]*ae_inp_time))
X_te = zeros((int((shape(X_out)[0]-ae_inp_time)/spacing),shape(X_out)[1]*ae_inp_time))


for i in range(int((shape(X)[0]-ae_inp_time)/spacing)):
    X_tr[i,:]=X[i*spacing:i*spacing+ae_inp_time,:].flatten()



for i in range(int((shape(X_out)[0]-ae_inp_time)/spacing)):
    X_te[i,:]=X_out[i*spacing:i*spacing+ae_inp_time,:].flatten()


X_tr=(X_tr-np.mean(X_tr))/np.std(X_tr)
X_te=(X_te-np.mean(X_te))/np.std(X_te)



compression_fac=30

inp_shape=int(shape(X_tr)[1])  #we treat all reservoir nodes at each timepoint as a sample
encoding_dim=int(inp_shape/compression_fac)
res_input= Input(shape=(inp_shape,))
encoded = Dense(int(inp_shape/(compression_fac/4)), activation='relu')(res_input)
encoded = Dense(int(inp_shape/(compression_fac/2)), activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(int(inp_shape/(compression_fac/2)), activation='relu')(encoded)
decoded = Dense(int(inp_shape/(compression_fac/4)), activation='relu')(decoded)
decoded = Dense(inp_shape, activation='relu')(decoded)

autoencoder = Model(res_input, decoded)
encoder = Model(res_input, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
#decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(X_tr, X_tr,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_te, X_te))


####################### we take the lower dimensional representation of the autoencoder and use it to train the reservoir weights

encoded_train= encoder.predict(X_tr)
reg = 1e-6  # regularization coefficient
#Wout = dot( dot(Yt,encoded_train), linalg.inv( dot(encoded_train,encoded_train.T) + reg*eye(1+inSize+resSize) ) )
Yt=Yt[:,ae_inp_time:]
Wout=dot( Yt, linalg.pinv(encoded_train.T) )


### see how we do on test data - encoded
encoded_test= encoder.predict(X_te)
predicted_test = autoencoder.predict(X_te)

Y=dot(Wout,encoded_test.T)




# compute MSE for the first errorLen time steps
errorLen = testLen-initLen
mse = sum( square( data[ae_inp_time+trainLen+1:trainLen+errorLen+1] - Y[0,0:errorLen] ) ) /sum( square( data[ae_inp_time+trainLen+1:trainLen+errorLen+1] - sqrt(mix)*data_mix[ae_inp_time+trainLen+1:trainLen+errorLen+1] ) )

res.append(mse)




figure(figsize=(10,5))
subplot(211)
plot( data[ae_inp_time+trainLen+10:trainLen+testLen+1], 'r' )
plot( Y.T[10:testLen+1], 'k' )
legend([r'Actual $x_1$','Reservoir $x_1$'], loc=1)
xticks([])
ylabel(r'$x_1(t), v_R(t)$', fontsize=14)   


ax=subplot(212)
plot( inp1[ae_inp_time+trainLen+10:trainLen+testLen+1], 'r' )
plot( inp2[ae_inp_time+trainLen+10:trainLen+testLen+1], 'g' )
legend( [r'Actual $x_1$', 'Actual $x_2$'], loc=1)
ylabel(r'$x_1(t) ,x_2(t)$', fontsize=14)
xlabel('Time ', fontsize=20)
#ticks = [int(t) for t in plt.xticks()[0]]
#plt.xticks(ticks, [t*dt for t in ticks])
subplots_adjust(hspace=0.4)
show()
