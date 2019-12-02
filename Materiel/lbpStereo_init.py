"""
author: Lily Amsellem
"""

import numpy as np
import scipy.ndimage
import imageio
import matplotlib.pyplot as plt

def compute_data_cost(I1, I2, num_disp_values, Tau):
    """data_cost: a 3D array of sixe height x width x num_disp_value;
    data_cost(y,x,l) is the cost of assigning the label l to pixel (y,x).
    The cost is min(1/3 ||I1[y,x]-I2[y,x-l]||_1, Tau)."""
    h,w,_ = I1.shape
    dataCost=np.zeros((h,w,num_disp_values))
    for l in range(num_disp_values):
        dataCost[:,l:w,l]=np.minimum(np.linalg.norm(I1[:,l:w]-I2[:,0:w-l],ord=1,axis=2),Tau*np.ones((h,w-l)));
        dataCost[:,0:l,l]=np.minimum(np.linalg.norm(I1[:,0:l]-I2[:,w-l:w],ord=1,axis=2),Tau*np.ones((h,l)));   
    return dataCost



def compute_energy(dataCost,disparity,Lambda):
    """dataCost: a 3D array of sixe height x width x num_disp_values;
    dataCost(y,x,l) is the cost of assigning the label l to pixel (y,x).
    disparity: array of size height x width containing disparity of each pixel.
    (an integer between 0 and num_disp_values-1)
    Lambda: a scalar value.
    Return total energy, a scalar value"""
    
    h,w,_= dataCost.shape
    yy,xx = np.meshgrid(range(h),range(w),indexing='ij')
    local_costs=dataCost[yy,xx,disparity]
    energy=np.sum(local_costs)
    
    pottsEnergyR=(disparity-np.roll(disparity,1,axis=1))!=0 #comparison with right pixel
    pottsEnergyL=(disparity-np.roll(disparity,-1,axis=1))!=0 #comparison with left pixel
    pottsEnergyU=(disparity-np.roll(disparity,-1,axis=0))!=0 #Comparison with above pixel
    pottsEnergyD=(disparity-np.roll(disparity,1,axis=0))!=0 #Comparison with down pixel
    
    #Set energy to 0 on borders for pixels that have no neighbours
    pottsEnergyR[:,w-1]=0
    pottsEnergyL[:,0]=0
    pottsEnergyU[0,:]=0
    pottsEnergyD[h-1,:]=0    
    pottsEnergy=pottsEnergyL+pottsEnergyR+pottsEnergyU+pottsEnergyD
    
    energy+=Lambda*np.sum(pottsEnergy)
    return energy

def update_msg(msgUPrev,msgDPrev,msgLPrev,msgRPrev,dataCost,Lambda):
    """Update message maps.
    dataCost: 3D array, depth=label number.
    msgUPrev,msgDPrev,msgLPrev,msgRPrev: 3D arrays (same dims) of old messages.
    Lambda: scalar value
    Return msgU,msgD,msgL,msgR: updated messages"""
    msgU=np.zeros(dataCost.shape)
    msgD=np.zeros(dataCost.shape)
    msgL=np.zeros(dataCost.shape)
    msgR=np.zeros(dataCost.shape)
    
    h,w,_=dataCost.shape

    #The messages are considered as OUTGOING
    #We need to compute INCOMING messages
    #For instance, incoming message from above pixel = outgoing message from pixel below shifted of 1
    #upward
    incomingMsgU=np.roll(msgDPrev,-1,axis=0)
    incomingMsgD=np.roll(msgUPrev,1,axis=0)
    incomingMsgL=np.roll(msgRPrev,-1,axis=1)
    incomingMsgR=np.roll(msgLPrev,1,axis=1)
    
    #Messages from the pixels on the boundaries are not computed
    incomingMsgU[h-1,:,:]=0
    incomingMsgD[0,:,:]=0
    incomingMsgL[:,w-1,:]=0
    incomingMsgR[:,0,:]=0
    
    #Compute outgoing messages
    NU=dataCost+incomingMsgD+incomingMsgL+incomingMsgR
    ND=dataCost+incomingMsgU+incomingMsgL+incomingMsgR
    NL=dataCost+incomingMsgU+incomingMsgD+incomingMsgR
    NR=dataCost+incomingMsgU+incomingMsgD+incomingMsgL
    
    #Use question 2) Formula
    SU=np.amin(NU,axis=2)
    SD=np.amin(ND,axis=2)
    SL=np.amin(NL,axis=2)
    SR=np.amin(NR,axis=2)
    
    for l in range(num_disp_values):
        msgU[:,:,l]=np.minimum(NU[:,:,l],Lambda+SU)
        msgD[:,:,l]=np.minimum(ND[:,:,l],Lambda+SD)
        msgL[:,:,l]=np.minimum(NL[:,:,l],Lambda+SL)
        msgR[:,:,l]=np.minimum(NR[:,:,l],Lambda+SR)
    return msgU,msgD,msgL,msgR

def normalize_msg(msgU,msgD,msgL,msgR):
    """Subtract mean along depth dimension from each message"""
    avg=np.mean(msgU,axis=2)
    msgU -= avg[:,:,np.newaxis]
    avg=np.mean(msgD,axis=2)
    msgD -= avg[:,:,np.newaxis]
    avg=np.mean(msgL,axis=2)
    msgL -= avg[:,:,np.newaxis]
    avg=np.mean(msgR,axis=2)
    msgR -= avg[:,:,np.newaxis]
    return msgU,msgD,msgL,msgR

def compute_belief(dataCost,msgU,msgD,msgL,msgR):
    """Compute beliefs, sum of data cost and messages from all neighbors"""
    beliefs=dataCost.copy()
    beliefs+=msgU+msgD+msgL+msgR
    return beliefs

def MAP_labeling(beliefs):
    """Return a 2D array assigning to each pixel its best label from beliefs
    computed so far"""
    MAPs=np.zeros((beliefs.shape[0],beliefs.shape[1]))
    #get indices of labels which minimize the cost
    MAPs=np.argmin(beliefs,axis=2)  
    return MAPs

def stereo_bp(I1,I2,num_disp_values,Lambda,Tau=15,num_iterations=60):
    """The main function"""
    dataCost = compute_data_cost(I1, I2, num_disp_values, Tau)
    energy = np.zeros((num_iterations)) # storing energy at each iteration
    # The messages sent to neighbors in each direction (up,down,left,right)
    h,w,_ = I1.shape
    msgU=np.zeros((h, w, num_disp_values))
    msgD=np.zeros((h, w, num_disp_values))
    msgL=np.zeros((h, w, num_disp_values))
    msgR=np.zeros((h, w, num_disp_values))

    for iter in range(num_iterations):
        msgU,msgD,msgL,msgR = update_msg(msgU,msgD,msgL,msgR,dataCost,Lambda)
        msgU,msgD,msgL,msgR = normalize_msg(msgU,msgD,msgL,msgR)
        # Next lines unused for next iteration, could be done only at the end
        beliefs = compute_belief(dataCost,msgU,msgD,msgL,msgR)
        disparity = MAP_labeling(beliefs)
        energy[iter] = compute_energy(dataCost,disparity,Lambda)
    return disparity,energy

# Input
img_left =imageio.imread('imL.png')
img_right=imageio.imread('imR.png')


fig=plt.figure(figsize=(10,10))
fig.add_subplot(121)
plt.imshow(img_left)
fig.add_subplot(122)
plt.imshow(img_right)
plt.show()

# Convert as float gray images
img_left=img_left.astype(float)
img_right=img_right.astype(float)

# Parameters
num_disp_values=16 # these images have disparity between 0 and 15. 
Lambda=10.0

# Gaussian filtering
I1=scipy.ndimage.filters.gaussian_filter(img_left, 0.6)
I2=scipy.ndimage.filters.gaussian_filter(img_right,0.6)

disparity,energy = stereo_bp(I1,I2,num_disp_values,Lambda)
imageio.imwrite('disparity_{:g}.png'.format(Lambda),disparity)

# Plot results
fig=plt.figure(figsize=(6,6))
plt.subplot(121)
plt.plot(energy)
plt.subplot(122)
plt.imshow(disparity,cmap='gray',vmin=0,vmax=num_disp_values-1)
plt.show()
