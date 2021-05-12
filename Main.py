# -*- coding: utf-8 -*-
"""

@author: Walid abdelmoula et al.
"""

# -*- coding: utf-8 -*-
"""
- Implementation of massNet (Abdelmoula et al, bioRxiv, 2021)
- This is a follow-up development that benefited from our previous methods of msiPL (Abdelmoula et al, bioRxiv 2020) 
- The massNet software is shared under the 3D Slicer Software License agreement. 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(1337)
from tensorflow import set_random_seed
set_random_seed(2)

import os
import h5py
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, plot_confusion_matrix, accuracy_score
from keras.losses import  categorical_crossentropy, binary_crossentropy
from keras.utils import plot_model
from scipy import stats
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
from ROC_Figure import *
import forcelayout as fl # Force Directed Layout Algorithms
import umap
import hdbscan
import nibabel as nib
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import time

# ========= Color Map ==============                                      
def discrete_cmap(N, base_cmap):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    MyCmap = LinearSegmentedColormap.from_list(cmap_name, color_list, N)
    
    cmaplist = [MyCmap(i) for i in range(MyCmap.N)]
    cmaplist[0] = (0,0,0) # First Entry Black background
    MyCmapF = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, MyCmap.N)
    return MyCmapF

# ====== Visualize Image: From 1D vector to Image ==============
def Image_Distribution(V,xLoc,yLoc):
    col = max(np.unique(xLoc))
    row = max(np.unique(yLoc))
    Myimg = np.zeros((col,row))
    for i in range(len(xLoc)):
        Myimg[np.asscalar(xLoc[i])-1, np.asscalar(yLoc[i])-1] = V[i]
    return Myimg

# ============ Scatter Plot Embedding =================
def Plot_Embedding(Data_LowDim, myColors,figS):
    plt.figure(figsize=(figS, figS)); 
    g = sns.scatterplot(x="umap1", y="umap2",  data=Data_LowDim, hue="Label",
                        palette=myColors, legend='full')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig = g.get_figure()

#============= Spatial Distribution Encoded Fetaures =============
def get_EncFeatures(Latent_z,myZCoord,xLocation,yLocation,directory):
    myzSections = np.unique(myZCoord)
    ndim = Latent_z.shape[1]
    for zr in range(len(myzSections)):
        ij_r = np.argwhere(myZCoord == myzSections[zr])
        indx = ij_r[:,0]
        xLoc = xLocation[indx]
        yLoc = yLocation[indx]
        zSection_Latent_z = np.squeeze(Latent_z[indx,])        
        plt.figure(figsize=(10, 10))
        for j in range(ndim):
            EncFeat = zSection_Latent_z[:,j] #encoded_imgs[i,0] #image index starts at 0 not 1 
            im = Image_Distribution(EncFeat,xLoc,yLoc);
            ax = plt.subplot(1, ndim, j + 1)    
            plt.imshow(im,cmap="hot");   #plt.colorbar()   
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
        directory_Latz = directory+'//Latent'
        if not os.path.exists(directory_Latz):
            os.makedirs(directory_Latz)
        plt.savefig(directory_Latz + '\\EncFetaures_Tissue'+str(myzSections[zr])+'.png',bbox_inches='tight')

# ========================== 3D mz image ============================
def Save_3D_nifti(myData,ClassID,XCoord,YCoord,ZCoord,s,directory):
    mzSections = np.unique(ZCoord)
    Vol_mz = np.zeros((s,s,len(mzSections)))
    nSections = len(mzSections)
    directory_NIFT = directory 
    if not os.path.exists(directory_NIFT):
        os.makedirs(directory_NIFT)
    for Zsec in range(len(mzSections)):
        ij_r = np.argwhere(ZCoord == mzSections[Zsec])
        indx = ij_r[:,0]
        xLoc = XCoord[indx]
        yLoc = YCoord[indx]
        myDataV = myData[:,ClassID]
        MSI_2D = np.squeeze(myDataV[indx])
        for idx in range(len(xLoc)):
            Vol_mz[np.asscalar(xLoc[idx])-1, np.asscalar(yLoc[idx])-1,Zsec] = MSI_2D[idx]  

    I_nii = nib.Nifti1Image(Vol_mz,affine=np.eye(4))
    nib.save(I_nii,directory_NIFT +'\\3D_Image_Class' + str(ClassID) + '.nii')
# ================== Correlate Cluster with m.z ==============
def Correlate_cluster_mz(ClassID, myData, All_mz, Learned_mzPeaks):
    Kimg = myData[:,ClassID]
    Peaks_ID = [np.argmin(np.abs(All_mz[:] - Learned_mzPeaks[i])) for i in  range(len(Learned_mzPeaks))]
    MSI_PeakList = Tst_MSI[:,Peaks_ID[:]] # get only MSI data only for the shotlisted learned m/z peaks
    Corr_Val =  np.zeros(len(Learned_mzPeaks))
    for i in range(len(Learned_mzPeaks)):
        Corr_Val[i] = stats.pearsonr(Kimg,MSI_PeakList[:,i])[0]
    id_mzCorr = np.argmax(Corr_Val)
    rank_ij =  np.argsort(Corr_Val)[::-1]
    return MSI_PeakList,Corr_Val, rank_ij
    
# =================== Display mz Image =====================
def Display_mzImage(MSI_D,myZCoord,xLocation,yLocation):
    myzSections = np.unique(myZCoord)
    for zr in range(len(myzSections)):
        ij_r = np.argwhere(myZCoord == myzSections[zr])
        indx = ij_r[:,0]
        xLoc = xLocation[indx]
        yLoc = yLocation[indx]
        zSection_MSI_V = MSI_D[indx]    
        im = Image_Distribution(zSection_MSI_V,xLoc,yLoc)
        ax = plt.subplot(1, len(myzSections), zr + 1)    
        plt.imshow(im,cmap="hot");   #plt.colorbar()   
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        
# =================== Load MSI Data  ========================
def Load_MSI(sub_ij, Data_idx):
    Combined_MSI = []; XCoord = []; YCoord = []; ZCoord = []; Labels =[]
    for id in range(0,len(sub_ij)):
        ij = sub_ij[id]
        f =  h5py.File('MSI_imzml/hd5/' + Data_idx[ij] + '.h5','r')
        MSI_train = np.transpose(f["Data"])
        mzList = f["mzArray"]
        Class_Label = f["Class_Label"] 
        Class_Label = np.asarray(Class_Label)
        nSpecFeatures = len(mzList)
        xLocation = np.array(f["xLocation"]).astype(int)
        yLocation = np.array(f["yLocation"]).astype(int)
        zLocation = np.full(len(yLocation),id)
        col = max(np.unique(xLocation))
        row = max(np.unique(yLocation))
        im = np.zeros((col,row))
        if id==0:
            Combined_MSI = MSI_train
            XCoord = xLocation
            YCoord = yLocation
            ZCoord = zLocation
            Labels = Class_Label
        else:
            Combined_MSI = np.concatenate((MSI_train,Combined_MSI), axis=0)
            XCoord = np.concatenate((xLocation,XCoord))
            YCoord = np.concatenate((yLocation,YCoord))
            ZCoord = np.concatenate((zLocation,ZCoord))
            Labels = np.concatenate((Class_Label,Labels)) 
    return Combined_MSI, XCoord, YCoord, ZCoord, Labels, mzList
        
# ========= msiPL Prediction: encoded features =============
def msiPL_Predict(myModel,encoder,Data):
    encoded_imgs = encoder.predict(Data) # Learned non-linear manifold
    decoded_imgs = myModel.predict(Data) # Reconstructed Data
    dec_TIC = np.sum(decoded_imgs, axis=-1)
    return encoded_imgs, decoded_imgs, dec_TIC

  
# =================== Get Training and Testing Data  ======================
# Training: odd index               # Testing: even index    
# =================== Load MSI Data  ======================================
Data_idx = ['GBM12_2', 'GBM12_1', 'GBM22_1','GBM22_2','GBM39_1','GBM39_2','GBM108_negative','GBM108_positive']    
#Train_Data = {'GBM12_2','GBM22_1','GBM39_1','GBM108_negative'};
# Load Training:
Training_id = [0,2,4,6]
Tr_MSI, Tr_XCoord, Tr_YCoord, Tr_ZCoord, Tr_Labels, All_mz = Load_MSI(Training_id, Data_idx)
All_mz = np.array(All_mz)
directory_tr = 'msiPL_Results/' +'Training'
if not os.path.exists(directory_tr):
    os.makedirs(directory_tr) 
# ======= Display mz Image ==========
mzId = np.argmin(np.abs(All_mz[:] - 400.9546))
mz_I = Tr_MSI[:,mzId]
Display_mzImage(mz_I,Tr_ZCoord,Tr_XCoord,Tr_YCoord)

# ====== Average Spectrum: Normal Vs Tumor
ij_T = np.argwhere(Tr_Labels==2);
Spec_Tumor = np.squeeze(Tr_MSI[ij_T,:])

AvgSpec_Tumor = np.mean(Spec_Tumor,axis=0)
plt.plot(All_mz,AvgSpec_Tumor,color = 'm'); 
plt.title('Average Spectrum: Tumor Only')

ij_N = np.argwhere(Tr_Labels==1); 
Spec_Normal = np.squeeze(Tr_MSI[ij_N,:])
AvgSpec_Normal = np.mean(Spec_Normal,axis=0)
plt.plot(All_mz,AvgSpec_Normal,color = 'c'); 
plt.title('Average Spectrum: Normal Only')

# ===================== PART#1 ===========================

# ================= msiPL: Load Model ======================
from Computational_Model import *
nSpecFeatures = len(All_mz)
input_shape = (nSpecFeatures, )
intermediate_dim = 512
latent_dim = 5
VAE_BN_Model = VAE_BN(nSpecFeatures,  intermediate_dim, latent_dim)
myModel, encoder = VAE_BN_Model.get_architecture()
myModel.summary()

# ============= msiPL: Model Training =================
start_time = time.time()
history = myModel.fit(Tr_MSI, epochs=100, batch_size=100, shuffle="batch")   
plt.plot(history.history['loss'])
plt.ylabel('loss'); plt.xlabel('epoch')
print("--- %s seconds ---" % (time.time() - start_time))
myModel.save_weights('Trained_4GBM.h5')

# ============= Load rained msiPL Model ============
myModel.load_weights('Trained_4GBM.h5')

# ================= Model Predictions ===================
encoded_imgs, decoded_imgs, dec_TIC = msiPL_Predict(myModel,encoder,Tr_MSI)
Latent_mean, Latent_var, Latent_z = encoded_imgs
get_EncFeatures(Latent_z,Tr_ZCoord,Tr_XCoord, Tr_YCoord,directory_tr)   

# ============= MSE: Original & Reconstructed Spectra ========
mse = mean_squared_error(Tr_MSI,decoded_imgs)
meanSpec_Rec = np.mean(decoded_imgs,axis=0) 
print('mean squared error(mse)  = ', mse)
meanSpec_Orig = np.mean(Tr_MSI,axis=0) # TIC-norm original MSI Data
N_DecImg = decoded_imgs/dec_TIC[:,None]  # TIC-norm reconstructed MSI  Data
meanSpec_RecTIC = np.mean(N_DecImg,axis=0)
plt.plot(All_mz,meanSpec_Orig,color = [0,0,0,1]); plt.plot(All_mz,meanSpec_RecTIC,color = [0,0.65,0.576,0.6]); 
plt.title('TIC-norm distribution of average spectrum: Original and Predicted')

# ========= Predicted m/z images
mzId = np.argmin(np.abs(All_mz[:] -616.17))
mz_I = N_DecImg[:,mzId]
Display_mzImage(mz_I,Tr_ZCoord,Tr_XCoord,Tr_YCoord)

#********************* Peak Learning ********************    
from LearnPeaks import *
meanSpec_Orig = np.mean(Tr_MSI,axis=0); std_spectra = np.std(Tr_MSI, axis=0) 
W_enc = encoder.get_weights()
# Normalize Weights by multiplying it with std of original data variables
Beta = 2.5
Learned_mzBins, Learned_mzPeaks, mzBin_Indx, Real_PeakIdx = LearnPeaks(All_mz, W_enc,std_spectra,latent_dim,Beta,meanSpec_Orig)

df_1 = pd.DataFrame({'mz Peaks': Learned_mzPeaks})
df_1.to_excel(directory_tr+'/'+'Learned_Peaks.xlsx', engine='xlsxwriter' , sheet_name='Sheet1')

# ====================== Tumor Spec Only: Umap ===================================
ij_Tr = np.argwhere(Tr_Labels==2)
z_Tr_section = np.squeeze(Tr_ZCoord[ij_Tr])
Spec_Tr_Tumor = np.squeeze(Tr_MSI[ij_Tr,:])
Latent_z_Tr_Tumor = np.squeeze(Latent_z[ij_Tr,:])

# ************************ Dimensionality Reduction **************************
Low_dim=2
embd_Y = umap.UMAP(n_components=Low_dim,n_neighbors=20,min_dist=0.0,random_state=0,).fit_transform(Latent_z)

my_Tissue_Label = Tr_ZCoord; myColors = ['grey','orange','magenta','dodgerblue'];
# my_Tissue_Label = Tr_Labels; myColors = ['green','red']

d = {'umap1': embd_Y[:,0], 'umap2': embd_Y[:,1], 'Label':my_Tissue_Label}
my_df = pd.DataFrame(data=d)
Plot_Embedding(my_df,myColors,figS=10)


# hf_tr = h5py.File('latentZ_TrainGBM.h5', 'w')
# hf_tr.create_dataset('Latent_z', data=Latent_z)
# hf_tr.create_dataset('TissueID', data=Tr_ZCoord)
# hf_tr.create_dataset('ClassID', data=Tr_Labels)
# hf_tr.close()

#df_umap = pd.DataFrame({'umap1': embd_Y[:,0], 'umap2': embd_Y[:,1], 'Latent_z':Latent_z, 'TissueID':Tr_ZCoord, 'ClassID':Tr_Labels})
#df_umap.to_hdf('umap_TrainingGBM.h5', key='df_umap', mode='w')

# ******* Density Clustering of Umap:
cluster_labels = hdbscan.HDBSCAN(min_samples=4, min_cluster_size=200,).fit_predict(embd_Y)
clustered = (cluster_labels >= 0)
plt.scatter(embd_Y[~clustered, 0],
            embd_Y[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(embd_Y[clustered, 0],
            embd_Y[clustered, 1],
            c=cluster_labels[clustered],
            s=0.1,
            cmap='Spectral');

# ===================== PCA =======================
from sklearn.decomposition import PCA
myPCA = PCA(n_components=2, svd_solver='full')
X_pca = myPCA.fit_transform(Tr_MSI)
# X_pca_test = myPCA.transform(Tst_MSI)  # FIt on New unseen Data

my_Tissue_Label = Tr_Labels; myColors = ['green','red']
d = {'umap1': X_pca[:,0], 'umap2': X_pca[:,1], 'Label':my_Tissue_Label}
my_df_pca = pd.DataFrame(data=d)
Plot_Embedding(my_df_pca,myColors,figS=10)

# =============================== PART#2 ================================
# ******************* Downstream Data Analysis **************************
# Fully Connected Neural Networks
# ***************** Multi-class classification *************************
from MultiClass_Classifier import * #Two fully connected layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

# ***************************************************************************
# Compile the model
def compile_FC(MC_Model,myModel):
    #1: Load the trained VAE weights to the classification model
    for l1,l2 in zip(MC_Model.layers[0:1], myModel.layers[0:1]):
        l1.set_weights(l2.get_weights())
    #2: check if the weights loaded correctly to the fc model:
    myModel.get_weights()[0][1]
    MC_Model.get_weights()[0][1]
    #3: No need to train the encoder-part since it was already trained:
    for layer in MC_Model.layers[0:1]: 
        layer.trainable = False
    # MC_Model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy']) #multi-class
    MC_Model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy']) #Binary Classification
    return MC_Model

# Evaluate cross validation mode
def fc_evaluate(ModelFc_KF, train_X, train_label,valid_X,valid_label):
    #CV_history = ModelFc_KF.fit(train_X, train_label, batch_size=100, epochs=30, shuffle = True, verbose=1,validation_data=(valid_X,valid_label))
    ModelFc_KF.fit(train_X, train_label, batch_size=100, epochs=30, shuffle = True, verbose=1,validation_data=(valid_X,valid_label))
    train_loss, train_acc = ModelFc_KF.evaluate(train_X,train_label, verbose=0)
    val_loss, val_acc = ModelFc_KF.evaluate(valid_X,valid_label, verbose=0)  
    
    # plt.figure()
    # plt.plot(CV_history.history['acc'],  '.', label='Training'); plt.legend()
    # plt.plot(CV_history.history['val_acc'], '-', label='Validation'); plt.legend()
    # plt.xlabel('Iterations'); plt.ylabel('Accuracy');
    # plt.figure()
    # plt.plot(CV_history.history['loss'],label='Training loss'); plt.legend()
    # plt.plot(CV_history.history['val_loss'],label='Validation loss'); plt.legend()
    
    return ModelFc_KF, train_loss, train_acc, val_loss, val_acc

# Split Data using Cross Validation:
def Apply_KFOLD_CrossValid(KF_Model,myModel,n_folds,nClasses,Data_X, Data_Y):
    train_scores, val_scores, All_fc_Models = list(), list(), list()
    train_losses, val_losses = list(), list()
    cv_Tr_MSI, cv_Tr_Labels, cv_val_MSI, cv_val_Labels = list(), list(), list(), list()
    kfold = KFold(n_folds, random_state=1, shuffle=True)   
    for train_idx, val_idx in kfold.split(Data_X):
        train_X, train_label = Data_X[train_idx], Data_Y[train_idx]
        valid_X, valid_label = Data_X[val_idx], Data_Y[val_idx]
        # Evaluate model:
        KF_FC_Model = KF_Model.fc()
        KF_FC_Model = compile_FC(KF_FC_Model,myModel)
        fc_model, train_loss, train_acc, val_loss, val_acc = fc_evaluate(KF_FC_Model,train_X, train_label, valid_X, valid_label)
        print('Train: %.3f, Validation: %.3f' % (train_acc, val_acc))
        train_scores.append(train_acc); train_losses.append(train_loss)
        val_scores.append(val_acc); val_losses.append(val_loss)
        All_fc_Models.append(fc_model)
        #Save Cross Validation Data
        cv_Tr_MSI.append(train_X); cv_Tr_Labels.append(train_label);
        cv_val_MSI.append(valid_X); cv_val_Labels.append(valid_label);
    return All_fc_Models, train_scores, train_losses, val_scores, val_losses, cv_Tr_MSI, cv_Tr_Labels, cv_val_MSI, cv_val_Labels

# ***************************************************************************
    
# 1. Fully connected classifier
Class_Label = Tr_Labels - 1
nClasses = len(np.unique(Class_Label)); nHidden = 128; 
trainY_One_Hot = to_categorical(Class_Label)
DL_CLassifier = MultiClass_Classifier(nSpecFeatures,nClasses,nHidden,encoder)


# 2. Run Classifier:
CV_Status =  int(input("Would you like to use 5-fold cross validation? Yes=1; No=0 ... :"))
if CV_Status== 0:
    start_time = time.time()
    FC_Model = DL_CLassifier.fc()
    #2: Compile the fully-connected classifier:
    FC_Model = compile_FC(FC_Model,myModel)
    FC_Model.summary()
    # Split Train/Validation:
    train_X,valid_X,train_label,valid_label = train_test_split(Tr_MSI,trainY_One_Hot,test_size=0.2,random_state=0)
    classify_train = FC_Model.fit(train_X, train_label, batch_size=100, epochs=30, verbose=1, shuffle="batch", validation_data=(valid_X,valid_label))
    print("Classification RUnning Time = %s seconds"   % (time.time() - start_time))
    plt.plot(classify_train.history['acc'],  '.', label='Training'); plt.legend()
    plt.plot(classify_train.history['val_acc'], '-', label='Validation'); plt.legend()
    plt.xlabel('Iterations'); plt.ylabel('Accuracy');
    plt.figure()
    plt.plot(classify_train.history['loss'],label='Training loss'); plt.legend()
    plt.plot(classify_train.history['val_loss'],label='Validation loss'); plt.legend()

else:
    #  5-fold cross validation
    n_folds = 5
    All_fc_Models, train_scores, train_losses, val_scores, val_losses,cv_Tr_MSI, cv_Tr_Labels, cv_val_MSI, cv_val_Labels= Apply_KFOLD_CrossValid(DL_CLassifier,myModel,n_folds,nClasses,Tr_MSI, trainY_One_Hot)   
    print('Training Accuracy = ',  (train_scores)) 
    print('Validation Accuracy = ',  (val_scores))    
    print('Training Loss = ',  (train_losses)) 
    print('Validation Loss = ',  (val_losses))  
    ID_BestModel = int (input('Enter index of the best K-fold model = '))
    del FC_Model
    FC_Model = All_fc_Models[ID_BestModel]
    
FC_Model.save_weights('FC_Model.h5')
# FC_Model.load_weights('FC_Model.h5')

# ----- Classification Report on Validation Data:
Pred_Val_One_Hot = FC_Model.predict(cv_val_MSI[ID_BestModel])
Valid_Labels = np.array([np.argmax(r) for r in cv_val_Labels[ID_BestModel]]); Valid_Labels += 1;
Pred_Labels_Val = np.array([np.argmax(r) for r in Pred_Val_One_Hot]); Pred_Labels_Val += 1;
ConfMtx_Tr = confusion_matrix(Valid_Labels,Pred_Labels_Val, normalize='true')
Annot_kws = {'ha':'center', 'va':'center'}
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(ConfMtx_Tr,cmap='Blues',annot=True,ax=ax) 
plt.ylabel('Actural')
plt.xlabel('Predicted')
plt.title('Validation Data')
plt.show()
print("[Validation] Classification report for classifier %s" %  classification_report(Valid_Labels, Pred_Labels_Val))

myROC = ROC_Figure(cv_val_Labels[ID_BestModel],Pred_Val_One_Hot)
myROC.getFigures()
myROC.get_AllFiguresROC()

# ----- Classification Report on Training Data:
Pred_Tr_One_Hot = FC_Model.predict(cv_Tr_MSI[ID_BestModel])
cv_training_Labels = np.array([np.argmax(r) for r in cv_Tr_Labels[ID_BestModel]]); cv_training_Labels += 1;
Pred_Labels_Tr = np.array([np.argmax(r) for r in Pred_Tr_One_Hot]); Pred_Labels_Tr += 1;
ConfMtx_Tr = confusion_matrix(cv_training_Labels,Pred_Labels_Tr, normalize='true')
Annot_kws = {'ha':'center', 'va':'center'}
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(ConfMtx_Tr,cmap='Blues',annot=True,ax=ax) 
plt.ylabel('Actural')
plt.xlabel('Predicted')
plt.title('Training Data')
plt.show()
print("[Training] Classification report for classifier %s" %  classification_report(cv_training_Labels, Pred_Labels_Tr))

myROC = ROC_Figure(cv_Tr_Labels[ID_BestModel],Pred_Tr_One_Hot)
myROC.getFigures()
myROC.get_AllFiguresROC()

# ===========================================================================
#                 Apply DL Classifier on Test Data
#                 Test Test Test Test Test Test Test    
# ***************************************************************************
# =============== Test DL Classifier on New Unseen Data ====================
# 1. Load MSI Testing:
 #TestData = {'GBM12_1', 'GBM22_2','GBM39_2','GBM108_positive'};


Test_id = [1,3,5,7]
Tst_MSI, Tst_XCoord, Tst_YCoord, Tst_ZCoord, Tst_Labels, All_mz = Load_MSI(Test_id, Data_idx)
directory_tst = directory_tr + '/' + 'Testing'
if not os.path.exists(directory_tst):
    os.makedirs(directory_tst) 

mzId = np.argmin(np.abs(All_mz[:] - 400.9546))
mz_I = Tst_MSI[:,mzId]
Display_mzImage(mz_I,Tst_ZCoord,Tst_XCoord,Tst_YCoord)
#  Ion_mz  = np.empty((Tst_MSI.shape[0], 2), type(Tst_MSI))
Image_size = 100;ClassID=1;
Save_3D_nifti(Ion_mz,ClassID,Tst_XCoord,Tst_YCoord,Tst_ZCoord,Image_size,directory_tst)


# 1.1. Average Spectrum: Normal vs Tumor:
ij_T = np.argwhere(Tst_Labels==2);
Spec_Tumor = np.squeeze(Tst_MSI[ij_T,:])

AvgSpec_Tumor = np.mean(Spec_Tumor,axis=0)
plt.plot(All_mz,AvgSpec_Tumor,color = 'm'); 
plt.title('Average Spectrum: Tumor Only')

ij_N = np.argwhere(Tst_Labels==1); 
Spec_Normal = np.squeeze(Tst_MSI[ij_N,:])
AvgSpec_Normal = np.mean(Spec_Normal,axis=0)
plt.plot(All_mz,AvgSpec_Normal,color = 'c'); 
plt.title('Average Spectrum: Normal Only')


# 2. Test DL CLassifier
start_time = time.time();
TestY_One_Hot = to_categorical(Tst_Labels - 1)
test_loss,test_acc = FC_Model.evaluate(Tst_MSI,TestY_One_Hot,verbose=0)
print('Test Acuuracy: %.3f' % test_acc)
print("Classification Running Time = %s seconds"   % (time.time() - start_time))
# print('Train Accuracy: %.3f, Validation Accuracy: %.3f' % (train_scores[ID_BestModel], val_scores[ID_BestModel]))

# 2.1. Probabilistic Prediction:
Tst_Pred = FC_Model.predict(Tst_MSI)
directory_tst_Pred = directory_tr + '/' + 'Testing' + '/' + 'Prediction_Images'
if not os.path.exists(directory_tst_Pred):
    os.makedirs(directory_tst_Pred) 
get_EncFeatures(Tst_Pred,Tst_ZCoord,Tst_XCoord, Tst_YCoord,directory_tst_Pred) 

Image_size = 100;ClassID=0;
Save_3D_nifti(Tst_Pred,ClassID,Tst_XCoord,Tst_YCoord,Tst_ZCoord,Image_size,directory_tst_Pred)

# 3. ROC Analysis on Test Data:s
pred_labels_One_Hot = FC_Model.predict(Tst_MSI)
myROC = ROC_Figure(TestY_One_Hot,pred_labels_One_Hot)
myROC.getFigures()
myROC.get_AllFiguresROC()

# 4. Confusion Matrix:
Pred_Labels = np.array([np.argmax(r) for r in pred_labels_One_Hot]); Pred_Labels += 1;
ConfMtx = confusion_matrix(Tst_Labels,Pred_Labels, normalize='true')
Annot_kws = {'ha':'center', 'va':'center'}
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(ConfMtx,cmap='Blues',annot=True,ax=ax) 
plt.ylabel('Actural')
plt.xlabel('Predicted')
plt.show()

# 2.2. Classification Report:
print("Classification report for classifier %s" %  classification_report(Tst_Labels, Pred_Labels))

# 2.3. Correlate Clusters with m/z
ClassID=1
MSI_PeakList, Corr_Val, rank_ij = Correlate_cluster_mz(ClassID, Tst_Pred, All_mz, Learned_mzPeaks)
id_mzCorr = np.argmax(Corr_Val)
Top_10mz = [i for i in Learned_mzPeaks[rank_ij[0:10]]]
Top_10Corr = [j for j in Corr_Val[rank_ij[0:10]]]


# TOp correlated m/z
rankID = 0
mz_I = MSI_PeakList[:,rank_ij[rankID]]
Display_mzImage(mz_I,Tst_ZCoord,Tst_XCoord,Tst_YCoord)
print('mz=',Learned_mzPeaks[rank_ij[rankID]])
print('Corr. Value=',Corr_Val[rank_ij[rankID]])

# plt.plot(Learned_mzPeaks,Corr_Val)
# print(['%0.4f' % i for i in Learned_mzPeaks[rank_ij[0:10]]])
# print('Correlation Top Ranked peaks:', end='')
# print(['%0.4f' % i for i in Corr_Val[rank_ij[0:10]]])

# ================= Model Predictions ===================
encoded_imgs_tst, decoded_imgs_tst, dec_TIC_tst = msiPL_Predict(myModel,encoder,Tst_MSI)
Latent_mean_tst, Latent_var_tst, Latent_z_tst = encoded_imgs_tst
get_EncFeatures(Latent_z_tst,Tst_ZCoord,Tst_XCoord, Tst_YCoord,directory_tst)   

# ============= MSE: Original & Reconstructed Spectra ========
mse = mean_squared_error(Tst_MSI,decoded_imgs_tst)
meanSpec_Rec = np.mean(decoded_imgs_tst,axis=0) 
print('mean squared error(mse)  = ', mse)
meanSpec_Orig_Tst = np.mean(Tst_MSI,axis=0) # TIC-norm original MSI Data
N_DecImg_Tst = decoded_imgs_tst/dec_TIC_tst[:,None]  # TIC-norm reconstructed MSI  Data
meanSpec_RecTIC_Tst = np.mean(N_DecImg_Tst,axis=0)
plt.plot(All_mz,meanSpec_Orig_Tst,color = [0, 1, 0,1]); plt.plot(All_mz,meanSpec_RecTIC_Tst,color = [1, 0, 0,0.6]); 
plt.title('TIC-norm distribution of average spectrum: Original and Predicted')

# ================ Umap of Tumor Only ========================
ij = np.argwhere(Tst_Labels==2)
z_section_Tst = np.squeeze(Tst_ZCoord[ij])
Spec_Tumor = np.squeeze(Tst_MSI[ij,:])
Latent_z_tst_Tumor = np.squeeze(Latent_z_tst[ij,:])
myColors = ['grey','orange','magenta','dodgerblue'];

AvgSpec_Tumor = np.mean(Spec_Tumor,axis=0)
plt.plot(All_mz,AvgSpec_Tumor,color = [1, 0, 0,1]); 
plt.title('Average Spectrum: Tumor Only')

umapY_T = umap.UMAP(n_components=2,n_neighbors=20,random_state=0,).fit_transform(Latent_z_tst_Tumor)
d = {'umap1': umapY_T[:,0], 'umap2': umapY_T[:,1], 'Label':z_section_Tst}
my_df_T = pd.DataFrame(data=d)
Plot_Embedding(my_df_T,myColors,figS=10)

# ================== Umap on Full Tissue ===========================
Low_dim=2
umapY_T = umap.UMAP(n_components=Low_dim,n_neighbors=20,min_dist=0.0,random_state=0,).fit_transform(Latent_z_tst)

my_Umap_Label = Tst_ZCoord; myColors = ['grey','orange','magenta','dodgerblue'];
# my_Umap_Label = Tst_Labels; myColors = ['green','red']

d_T = {'umap1': umapY_T[:,0], 'umap2': umapY_T[:,1], 'Label':my_Umap_Label}
my_df_T = pd.DataFrame(data=d_T)
Plot_Embedding(my_df_T,myColors,figS=10)

hf_tst = h5py.File('latentZ_TestGBM.h5', 'w')
hf_tst.create_dataset('Latent_z', data=Latent_z_tst)
hf_tst.create_dataset('TissueID', data=Tst_ZCoord)
hf_tst.create_dataset('ClassID', data=Tst_Labels)
hf_tst.close()

# hf_tst = h5py.File('MSI_TestData.h5', 'w')
# hf_tst.create_dataset('MSI_Data', data=Tst_MSI)
# hf_tst.create_dataset('mzList', data=All_mz)
# hf_tst.close()

# ================== Color umap using m/z values =======================
ClassID =  int(input("Compute Top10 mz in: Normal(0) or Tumor(1)? Your Selection="))
MSI_PeakList, Corr_Val, rank_ij = Correlate_cluster_mz(ClassID, Tst_Pred, All_mz, Learned_mzPeaks)
id_mzCorr = np.argmax(Corr_Val)
Top_10mz = [i for i in Learned_mzPeaks[rank_ij[0:10]]]
Top_10Corr = [j for j in Corr_Val[rank_ij[0:10]]]

#mydf =  pd.read_hdf(directory_tst+'/umap_TestGBM.h5'); myMSI = Tst_MSI;
mydf =  pd.read_hdf(directory_tr+'/umap_TrainingGBM.h5');  myMSI = Tr_MSI;

# Create directory:
mydir = ['Training', 'Testing']   
ij_dir = int(input("Select a number 0:Training or 1:Testing? Your answer = ")) 
directory_mz_umap = directory_tr + '/' + 'umap_mz' + '/' + mydir[ij_dir]
if not os.path.exists(directory_mz_umap):
    os.makedirs(directory_mz_umap) 
    
# Color Umap: Choose Tst_MSI or Tr_MSI
for indx in range(0,len(Top_10mz)):
    mzId = np.argmin(np.abs(All_mz[:] - Top_10mz[indx]))
    plt.figure(figsize=(10, 10));    
    #plt.figure()
    plt.scatter(mydf.umap1,mydf.umap2,c=myMSI[:,mzId])
    plt.jet(); #plt.colorbar()
    plt.xlabel('umap1'); plt.ylabel('umap2')
    plt.title('mz = ' + str(All_mz[mzId]))
    plt.savefig(directory_mz_umap + '\\mz_' + str(All_mz[mzId]) + '.png')

# ===============================SVM=======================================
# -------------------------- SVM Classification ---------------------------
#==========================================================================

# ======================= SVM Classification ==============================
# SVM Feature Importance (Linear Kernel Only)
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
    
from sklearn import  svm
from sklearn.model_selection import train_test_split

# a. Create a classifier:
Myclassifier = svm.SVC(gamma='scale', kernel='linear',random_state=0) 
# Myclassifier = svm.SVC(gamma='scale', kernel='rbf') 

# Grid search:
# from sklearn.model_selection import GridSearchCV
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5], 'gamma':[0.1,1]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)
# clf.fit(X_train, y_train)
# search.cv_results_['params'][search.best_index_]

# b. Split Data into training/testing:
# X_train, X_test, y_train, y_test = train_test_split(
#     Tr_MSI, Tr_Labels, test_size=0.3, shuffle=True)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn');

# c. Train the svm:
start_tr = time.time()
Myclassifier.fit(Tr_MSI, Tr_Labels)
end_tr = time.time()
print('Time Training SVM (seconds) = ', end_tr - start_tr)

# Classification Report Trainsing Data:
Pred_training = Myclassifier.predict(Tr_MSI)
print("[Traning Data] Classification report for classifier %s:\n%s\n"
      % (Myclassifier, classification_report(Tr_Labels, Pred_training)))
plot_confusion_matrix(Myclassifier, Tr_MSI, Tr_Labels,cmap='Blues') 
plt.show()

predicted_tr_One_hot = to_categorical(Pred_training-1)
y_test_One_hot = to_categorical(Tr_Labels-1)
myROC = ROC_Figure(y_test_One_hot,predicted_tr_One_hot)
myROC.get_AllFiguresROC()

# c.1 Fetaure importance (svm linear kernel)
svm_Coeff = Myclassifier.coef_
Ranked_Features = np.squeeze((-svm_Coeff).argsort())
Top100_mz_svm = [All_mz[i] for i in Ranked_Features[0:100]]

# d. Apply on Test Data:
start_tst = time.time()
predicted = Myclassifier.predict(Tst_MSI)
end_tst = time.time()
print('Time SVM on Test Data (seconds) = ', end_tst - start_tst)
print('Test Accuracy = ', accuracy_score(Tst_Labels, predicted))



# e. Confusion Matrix:
print("[Test Data] Classification report for classifier %s:\n%s\n"
      % (Myclassifier, classification_report(Tst_Labels, predicted)))
plot_confusion_matrix(Myclassifier, Tst_MSI, Tst_Labels,cmap='Blues') 
plt.show()

# Normalized Confusion Matrix
start_tst = time.time()
plot_confusion_matrix(Myclassifier, Tst_MSI, Tst_Labels,cmap='Blues',normalize='true') 
plt.show()
end_tst = time.time()
print('Time Training SVM (seconds) = ', end_tst - start_tst)
#print("Confusion matrix:\n%s" % disp.confusion_matrix)

# f. ROC Distribution:
from sklearn.metrics import roc_auc_score
from ROC_Figure import *
from keras.utils import to_categorical
predicted_One_hot = to_categorical(predicted-1)
y_test_One_hot = to_categorical(Tst_Labels-1)
myROC = ROC_Figure(y_test_One_hot,predicted_One_hot)
# myROC.getFigures()
myROC.get_AllFiguresROC()

#fpr, tpr,_ = roc_curve(y_test_One_hot[:,1],predicted_One_hot[:,1])
#roc_auc = auc(fpr, tpr); print(roc_auc)

    

