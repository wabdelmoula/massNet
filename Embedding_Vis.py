# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:48:44 2020

@author: Admin
"""
import numpy as np
np.random.seed(0)

class Embedding_Vis(object):
    
    def __init__ (self, Y_embd, xLocation,  yLocation):
        self.Y_embd = Y_embd
        self.xLocation = xLocation
        self.yLocation = yLocation
        
                                     
# ======= Embedding To L*a*b* ===========
    def EmbeddingToLAB(self):
        maxV = np.max(abs(self.Y_embd),axis=0)
        minV = np.min(self.Y_embd,axis=0)
        rangeV = np.ptp(self.Y_embd,axis=0)
        color3D_LAB = np.zeros((self.Y_embd.shape[0],self.Y_embd.shape[1]))
        color3D_LAB[:,0] = 100*(self.Y_embd[:,0] - minV[0])/rangeV[0]
        color3D_LAB[:,1] = 127*(self.Y_embd[:,1])/maxV[1]
        color3D_LAB[:,2] = 127*(self.Y_embd[:,2])/maxV[2]
        return color3D_LAB
        
# ========= Spatial mapped embedding ============== 
    def EmbeddingToImage(self, LAB_Data):
        col = max(np.unique(self.xLocation))
        row = max(np.unique(self.yLocation))
        RGB_im =  np.zeros((col,row,3))
        for i in range(len(self.xLocation)):
            RGB_im[ np.asscalar(self.xLocation[i])-1, np.asscalar(self.yLocation[i])-1,0] = LAB_Data[i,0]
            RGB_im[ np.asscalar(self.xLocation[i])-1, np.asscalar(self.yLocation[i])-1,1] = LAB_Data[i,1]
            RGB_im[ np.asscalar(self.xLocation[i])-1, np.asscalar(self.yLocation[i])-1,2] = LAB_Data[i,2]
        return RGB_im
        
