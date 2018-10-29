#!/usr/bin/env python
import numpy as np 
import sys
import math
import csv
import time
from numpy import genfromtxt
from numpy.linalg import inv
import matplotlib.pyplot as plt


#	The indices will refer to different algorithms:
#	0 : BP
#	1 : BP_online
#	2 : LF
#	3 : LF_online
#	4 : LM


f=[0]*5  #flags for algorithm selection.
f[0]=1  # BP selected.
nhidden=25
ntrials=1
par=[0]*3
par[0]= 30 #Epochs
par[1] = 0.0002 #Max_Error
par[2] = 0.001 #TOL_Error

eta = [0]*5
savet = [0]*5
save_err = [0]*5
ctr = [0]*5
niter_conv = [0]*5


eta[0] = 0.01 # Learning Rate for BP
eta[1] = # Learning Rate for BP_online
eta[2] = # Learning Rate for LF
eta[3] = # Learning Rate for LF_online
eta[4] = # Learning Rate for LM




#################################################****** Please set parameters above *******##################################################################
class NN(object):

	#Class Defined for Neural Network Structure.

	def __init__ (Nn,W1,W2,Nw):
		self.Nn=Nn;
		self.W1=W1;
		self.W2=W2;
		self.Nw=Nw;


def dnn_init(l):

	# Initialising the network.
	W1=2*np.random.rand(l[2],l[1])-1;
	W2=2*np.random.rand(l[3],l[2])-1;
	Nw=np.multiply(l[1:2],l[2:3]);
	nn=NN(l,W1,W2,Nw);
	return nn ;


def dnn_fwd(NN,x):
	v = np.zeroes((length(x),3))
	v[:,1]=x
	h1=NN.W1.dot(v[:,1])
	v[:,2]=1/(1+math.exp(-h1));
	h2=NN.W2.dot(v[:,2])
	v[:,3]=1/(1+math.exp(-h2));

	return v;

def dnn_findJ(NN,x):
	v = dnn_fwd(NN,x);
	end = length(v)-1;
	for i in range(NN.Nn[2]):
		delta = np.zeros((NN.Nn[2],1))
		delta[i]=v[end].v[i]*(1-v[end].v[i])
	#Still Left.
	return [J,v]

def dnn_update(NN,u):
	temp =0;
	NN.W1 = NN.W1 + 
	temp=temp + NN.Nw[1]
	NN.W2 = NN.W2 + 
	temp=temp + NN.Nw[2]
	return NN ;

def fun_analyzetrial(l):
	NN=l[0]
	test_data = l[1]
	err = l[2]
	maxerr = l[3]
	temp = [err[i] for i, x in enumerate(err) if x<maxerr ]
	if length(temp)!=0:
		n=temp;
	else:
		n=length(err)+1;
	return n ;



def train_bp(l):
	data_in=l[1];
	data_out=l[2];
	NN=l[0]
	Np=length(data_in[:,1]);
	ctr=0
	e=np.zeros((Np*length(NN.Nn),1))
	J=np.zeros((Np*length(NN.Nn),sum(NN.Nw)))
	v1=np.zeros()
	while 1:
		for i in range(Np):
			a=np.ones((len(data_in[:,1]+1,),dtype=np.float)
			a[:len(data_in[:,1])]=data_in[i,:]
			x=a.T
			yd=data_out[i,:].T 			
			#Finding Jacobian 
			[Jtemp,v] = dnn_findJ(NN,x);
			J[(i-1)*NN.Nn[2]+1:i*NN.Nn[2],:] = Jtemp 
			e[(i-1)*NN.Nn[2]+1:i*NN.Nn[2],1] = yd - v[:,2]
		ctr = ctr +1
		save_err  = e.T.dot(e)
		NN=dnn_update(NN,u);
	return  [NN,save_err,ctr];


def main():
	for i in range(length(f)):
		if f[i]:
			savet[i] = np.zeros(ntrials);
			ctr[i] = np.zeros(ntrials);
			niter_conv[i] = np.zeros(ntrials);


	#Initialising Weights.
	data = genfromtxt('breast-cancer-wisconsin.data.csv', delimiter=',')
	X_train=data[1:587,1:9]
	Y_train=data[1:587,10]
	X_test=data[588:683,1:9]
	Y_test=data[588:683,10]
	l=[length(X_train[1:])+1,nhidden,length(Y_train[1:])]
	NN_init=dnn_init(l)
	NN = [0]*5
	for i in range(f):
		NN[i]=NN_init

	#Training the network.
	for i in range(ntrials): 
		print i
		if f[0]:
			to=time.time()
			[NN[0],save_err[0],ctr[0]]=train_bp(NN[0],X_train,Y_train,par,eta[0]);   
			savet[0] = time.time() - to    
			niter_conv[0] = fun_analyzetrial(NN[0],X_test,save_err[0],par[1]);  

		if f[1]:
			to=time.time()
			[NN[1],save_err[1],ctr[1]]=train_bp_online(NN[1],X_train,Y_train,par,eta[1]);   
			savet[1] = time.time() - to    
			niter_conv[1] = fun_analyzetrial(NN[1],X_test,save_err[1],par[1]);  

		if f[2]:
			to=time.time()
			[NN[2],save_err[2],ctr[2]]=train_bp_online(NN[2],X_train,Y_train,par,eta[2]);   
			savet[2] = time.time() - to    
			niter_conv[2] = fun_analyzetrial(NN[2],X_test,save_err[2],par[1]);  

		if f[3]:
			to=time.time()
			[NN[3],save_err[3],ctr[3]]=train_bp_online(NN[3],X_train,Y_train,par,eta[3]);   
			savet[3] = time.time() - to    
			niter_conv[3] = fun_analyzetrial(NN[3],X_test,save_err[3],par[1]);  

		if f[4]:
			to=time.time()
			[NN[4],save_err[4],ctr[4]]=train_bp_online(NN[4],X_train,Y_train,par,eta[4]);   
			savet[4] = time.time() - to    
			niter_conv[4] = fun_analyzetrial(NN[4],X_test,save_err[4],par[1]);  


if __name__ == '__main__':
	main()