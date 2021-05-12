# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:46:02 2020
          A fully connected logistic classifier (probabilistic multi-class)

@author: Admin
"""

import numpy as np
from keras.layers import Lambda, Input, Dense, ReLU, BatchNormalization, Dropout
from keras.constraints import maxnorm
from keras.models import Model
from keras import backend as K


class MultiClass_Classifier(object):
    
    def __init__ (self, nSpecFeatures, nClasses,  nHidden, Enc_Features):
        self.nClasses = nClasses
        self.nHidden = nHidden
        self.Enc_Features = Enc_Features
        self.nSpecFeatures = nSpecFeatures
        
    def fc(self):
        input_shape = (self.nSpecFeatures, )
        inputs = Input(shape=input_shape, name='encoder_input')
        den_1 = Dense(self.nHidden,activation='relu', kernel_constraint=maxnorm(3))(self.Enc_Features(inputs)[2])
        den_1 = Dropout(0.2)(den_1)
        den_1 = BatchNormalization()(den_1)
        
        den_2 = Dense(self.nHidden,activation='relu', kernel_constraint=maxnorm(3))(den_1)
        den_2 = Dropout(0.2)(den_2)
        den_2 = BatchNormalization()(den_2)
        
        out = Dense(self.nClasses, activation='softmax')(den_2)
        fc_model = Model(inputs, out)
        fc_model.summary()
        return fc_model
                                         