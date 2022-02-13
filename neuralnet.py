import pandas as pd
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="Data File")
parser.add_argument('-l', '--eta', type=float, metavar='', required=True, help='Learning rate')
parser.add_argument('-t', '--iterations', type=int, metavar='', required=True, help='iterations')
args = parser.parse_args()
file= args.data
n= args.eta
iterations= args.iterations

# iterations = 2
# n = 0.2

# dataset = pd.read_csv('F:\DE\ML5Credits\ML Win2021 Assignments\Gauss_new/Gauss3.csv',header=None)  
# file= 'F:\DE\ML5Credits\ML Win2021 Assignments\Gauss_new/Gauss3.csv'

dataset = pd.read_csv(file,header=None)  
#file= 'F:\DE\ML5Credits\ML Win2021 Assignments\Gauss_new/Gauss3.csv'

X = dataset.iloc[:, :].values
Z = dataset.iloc[:, :-1].values
rows = Z.shape[0]
L = dataset.iloc[:, -1].values
leng=len(dataset)
#print(leng)

#epoch = 2*rows
epoch = iterations*rows


W = np.array([[0.2,-0.5,0.3],
				[-0.3,-0.1,0.2],  
				[0.4,-0.4,0.1]])


def sigmoid(x): return 1/(1 + np.exp(-x))

ones =np.ones((rows,1))
arr1= np.array([1])
x_values = np.concatenate((ones, Z), axis=1)

F=np.dot(x_values[0],W)
#print(type(F))
#print(F)
H = sigmoid(F)
#print(H)
Hc =np.append(H,arr1)
#print(Hc)
#print(type(Hc))
#WH = np.array([0.1,0.3,-0.4,-0.1])
WHo = np.array([0.1,0.3,-0.4])
Wob = -0.1
#oW= np.sum(np.multiply(Hc,WH))
#po=sigmoid(oW)
#print(sigmoid(oW))
#error = L[0]- po 
#print(L)
#deltAo = po*(1-po)*float(error)
#print(deltAo)
#deltWo=np.multiply((H*(1-H)*deltAo),WHo)
#print(np.multiply((H*(1-H)*deltAo),WHo))


#print(x_values[0].reshape(1,3))
#print(deltWo.reshape(3,1))
#print(x_values[0].reshape(1,3).shape[0])
#AB = x_values[0].reshape(1,3)
#print(np.dot(deltWo.reshape(3,1),AB))
#Fb = 0.2*np.dot(deltWo.reshape(3,1),AB)
#print(Fb)
#print(Fb.transpose())
#K =Fb.transpose()
#print(W +K)
#0.20068	-0.29957	0.40024	-0.4982	-0.09884	-0.39936	0.29743	0.19834	0.09909	-0.07285	0.11458	0.30897	-0.38333
#print(WHo+(0.2*H*deltAo))

#print(x_values[5999])
print('-'+','+'-'+','+'-'+','+'-'+','+'-'+','+'-'+','+'-'+','+'-'+',''-'+',''-'+','+'-'+','+str(f'{W[0,0]:.5f}')+','+str(round(W[1,0],5))+','+str(round(W[2,0],5))+','+str(round(W[0,1],5))+','+str(round(W[1,1],5))+','+str(f'{W[2,1]:.5f}')+','+str(round(W[0,2],5))+','+str(round(W[1,2],5))+','+str(round(W[2,2],5))+','+str(f'{Wob:.5f}')+','+str(round(WHo[0],5))+','+str(round(WHo[1],5))+','+str(round(WHo[2],5)))

#print(epoch)
# for i in range(iterations*leng):
# 	j=i
# 	if j<rows:
# 		j=i
# 	elif j==rows:
# 		j=0
# 	else:
# 		j=i-rows	
		
	#j = i-rows if i >= rows else i
	# r=0
	# j = i	
	# if r >= rows:
	# 	r = j-rows
	#print(x_values[j])
	# WH= np.append(WHo,Wob)
	# net=np.dot(x_values[j],W)
	# Hout=sigmoid(net)
	# Hcout =np.append(Hout,arr1)
	# OnetW= np.sum(np.multiply(Hcout,WH))
	# Ooutw = sigmoid(OnetW)
	# err = L[j]- Ooutw
	# erroro= Ooutw*(1-Ooutw)*float(err)
	# deltaH = np.multiply((Hout*(1-Hout)*erroro),WHo)
	# inputrow = x_values[j].reshape(1,3)
	# change =  np.dot(deltaH.reshape(3,1),inputrow)
	# changeT = change.transpose()
	# W += n*changeT
	# Wob+=n*erroro
	# WHo += n*Hout*erroro
	

	#print(str(f'{x_values[j][1]:.5f}')+','+str(f'{x_values[j][2]:.5f}')+','+str(f'{Hout[0]:.5f}')+','+str(f'{Hout[1]:.5f}')+','+str(f'{Hout[2]:.5f}')+','+str(f'{Ooutw:.5f}')+','+str(L[j])+','+str(f'{deltaH[0]:.5f}')+','+str(f'{deltaH[1]:.5f}')+','+str(f'{deltaH[2]:.5f}')+','+str(f'{erroro:.5f}')+','+str(f'{W[0,0]:.5f}')+','+str(f'{W[1,0]:.5f}')+','+str(f'{W[2,0]:.5f}')+','+str(f'{W[0,1]:.5f}')+','+str(f'{W[1,1]:.5f}')+','+str(f'{W[2,1]:.5f}')+','+str(f'{W[0,2]:.5f}')+','+str(f'{W[1,2]:.5f}')+','+str(f'{W[2,2]:.5f}')+','+str(f'{Wob:.5f}')+','+str(f'{WHo[0]:.5f}')+','+str(f'{WHo[1]:.5f}')+','+str(f'{WHo[2]:.5f}'))


mi=0
while mi < iterations:
	j=0
	while j<rows:
		WH= np.append(WHo,Wob)
		net=np.dot(x_values[j],W)
		Hout=sigmoid(net)
		Hcout =np.append(Hout,arr1)
		OnetW= np.sum(np.multiply(Hcout,WH))
		Ooutw = sigmoid(OnetW)
		err = L[j]- Ooutw
		erroro= Ooutw*(1-Ooutw)*float(err)
		deltaH = np.multiply((Hout*(1-Hout)*erroro),WHo)
		inputrow = x_values[j].reshape(1,3)
		change =  np.dot(deltaH.reshape(3,1),inputrow)
		changeT = change.transpose()
		W += n*changeT
		Wob+=n*erroro
		WHo += n*Hout*erroro
		print(str(f'{x_values[j][1]:.5f}')+','+str(f'{x_values[j][2]:.5f}')+','+str(f'{Hout[0]:.5f}')+','+str(f'{Hout[1]:.5f}')+','+str(f'{Hout[2]:.5f}')+','+str(f'{Ooutw:.5f}')+','+str(L[j])+','+str(f'{deltaH[0]:.5f}')+','+str(f'{deltaH[1]:.5f}')+','+str(f'{deltaH[2]:.5f}')+','+str(f'{erroro:.5f}')+','+str(f'{W[0,0]:.5f}')+','+str(f'{W[1,0]:.5f}')+','+str(f'{W[2,0]:.5f}')+','+str(f'{W[0,1]:.5f}')+','+str(f'{W[1,1]:.5f}')+','+str(f'{W[2,1]:.5f}')+','+str(f'{W[0,2]:.5f}')+','+str(f'{W[1,2]:.5f}')+','+str(f'{W[2,2]:.5f}')+','+str(f'{Wob:.5f}')+','+str(f'{WHo[0]:.5f}')+','+str(f'{WHo[1]:.5f}')+','+str(f'{WHo[2]:.5f}'))
		#print(ji)
		j+=1
	mi+=1		





 