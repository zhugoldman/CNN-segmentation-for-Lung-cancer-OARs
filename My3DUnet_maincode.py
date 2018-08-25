import tensorflow as tf
from keras import backend as keras
from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Dropout, normalization,Activation,Add,Conv3DTranspose
from keras.layers import Dense, core, Conv2D,MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import *
import PlotResult
import matplotlib.pyplot as plt
from functools import partial,update_wrapper
import numpy as np

from ZhuAutoSegModel import MyDefineLoss
from ZhuAutoSegModel import load_data_preprocess


class My3DUnet():
    def __init__(self, label_num, checkpoint_dir=default_checkpoint_dir,img_frames=4,img_rows=72, img_cols=72):
        self.label_num = label_num
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_frames = img_frames
        self.checkpoint_dir = checkpoint_dir
        self.class_weights = np.ones(self.label_num)

    def masks_3d(self,masks):
        assert (len(masks.shape) == 5)  # 5D arrays
        assert (masks.shape[4] == 1)  # check the channel is 1
        im_fram = masks.shape[1]
        im_h = masks.shape[2]
        im_w = masks.shape[3]

        masks = np.reshape(masks, (masks.shape[0], im_h * im_w * im_fram))
        new_masks = np.zeros((masks.shape[0], im_h * im_w * im_fram, self.label_num))
        for i in range(masks.shape[0]):
            for j in range(im_h * im_w * im_fram):
                new_masks[i, j, masks[i, j]] = 1

        return new_masks

    def combine_image_label(self,old_image,old_labels,label_num):
        combine_mask = old_labels > label_num
        combine_labels = old_labels * combine_mask
        output_image = np.concatenate((old_image,combine_labels),axis=-1)
        output_labels = old_labels * np.logical_not(combine_mask)
        output_labels[combine_mask] = 1
        return output_image,output_labels

    def load_train_data(self,folderpath,Is3d=True):
        temp_train_imgs,temp_train_labels = load_data_preprocess.load_binary_data(folderpath)
        self.train_imgs,self.train_labels = self.combine_image_label(temp_train_imgs,temp_train_labels,self.label_num)
        for i in range(0, self.label_num):
            print("%d weight: %f" % (i, self.class_weights[i]))


        if Is3d == True:
            self.train_mask_labels = self.masks_3d(self.train_labels)
        else:
            self.train_mask_labels = self.masks_2d(self.train_labels)


    def normalization_data(self,data):
        data = load_data_preprocess.imagedata_preprocess_chest(data)
        return data

    def conv3D_and_bn(self,input_tensor,filters, kernel_size, padding_method='same', activation_method=None, kernel_initializer_method='he_normal'):
        output_tensor = Conv3D(filters, kernel_size, padding=padding_method, activation = activation_method,kernel_initializer=kernel_initializer_method)(input_tensor)
        output_tensor = normalization.BatchNormalization(axis=-1)(output_tensor)
        output_tensor = Activation('relu')(output_tensor)
        return output_tensor

    def Residual3d_x3(self,input_tensor, filters,kernel_size):
        skip = self.conv3D_and_bn(input_tensor,filters, kernel_size, padding_method='same', kernel_initializer_method='he_normal')
        conv = self.conv3D_and_bn(skip, filters, kernel_size, padding_method='same', kernel_initializer_method='he_normal')
        conv = self.conv3D_and_bn(conv, filters, kernel_size, padding_method='same', kernel_initializer_method='he_normal')
        return Add()([skip, conv])  # the residual connection

    def conv3d_as_pool(self,input_tensor,filters,kernel_size,strides):
        return Conv3D(filters, kernel_size, padding='same', activation = 'relu',kernel_initializer='he_normal',strides=strides)(input_tensor)

    def deconv3d_as_up(self,input_tensor,filters,kernel_size,strides):
        return Conv3DTranspose(filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal',
                      strides=strides)(input_tensor)

    def deconv3d_x3(self,lhs, rhs,filters,kernel_size,strides):
        rhs_up = self.deconv3d_as_up(rhs, filters, kernel_size, strides)
        lhs_conv = Conv3D(filters, kernel_size, padding='same', activation = 'relu',kernel_initializer='he_normal')(lhs)
        rhs_add = Add()([rhs_up, lhs_conv])
        conv = self.Residual3d_x3(rhs_add,filters,kernel_size)
        return Add()([conv, rhs_add])

    def build_model_8x96x96(self,label_num):
        inputs = Input((self.img_frames,self.img_rows, self.img_cols, 5))

        # conv_1 = self.Residual3d_x3(inputs, 16, (1,3,3))
        # pool = self.conv3d_as_pool(conv_1,32,(1,3,3),(1,2,2))
        conv_1 = self.Residual3d_x3(inputs, 16, (3,3,3))
        pool = self.conv3d_as_pool(conv_1,32,(3,3,3),(2,2,2))
        print("conv1 shape:", pool.shape)

        conv_2 = self.Residual3d_x3(pool, 32, (3,3,3))
        pool = self.conv3d_as_pool(conv_2,64,(3,3,3),(2,2,2))
        print("conv2 shape:", pool.shape)

        conv_3 = self.Residual3d_x3(pool, 64, (1,3,3))
        pool = self.conv3d_as_pool(conv_3,128,(1,3,3),(2,2,2))
        print("conv3 shape:", pool.shape)

        # conv_4 = self.Residual3d_x3(pool, 128, (1,5,5))
        # pool = self.conv3d_as_pool(conv_4,256,(1,5,5),(1,3,3))
        conv_4 = self.Residual3d_x3(pool, 128, (1,3,3))
        pool = self.conv3d_as_pool(conv_4,256,(1,3,3),(1,2,2))
        print("conv4 shape:", pool.shape)

        conv_5 = self.Residual3d_x3(pool, 256, (1,3,3))
        pool = self.conv3d_as_pool(conv_5,512,(1,3,3),(1,2,2))
        print("conv5 shape:", pool.shape)

        bottom = self.Residual3d_x3(pool, 512, (1,3,3))
        print("bottom shape:", bottom.shape)

        # deconv_4 = self.deconv3d_x3(conv_4, bottom, 128, (1,3,3), (1, 2, 2))
        deconv_5 = self.deconv3d_x3(conv_5, bottom, 256, (1, 3, 3), (1, 2, 2))
        deconv_4 = self.deconv3d_x3(conv_4, deconv_5, 128, (1, 3, 3), (1, 2, 2))
        deconv_3 = self.deconv3d_x3(conv_3, deconv_4, 64,(1,3,3),(2,2,2))
        deconv_2 = self.deconv3d_x3(conv_2, deconv_3, 32,(3,3,3),(2,2,2))
        deconv_1 = self.deconv3d_x3(conv_1, deconv_2, 16, (3, 3, 3), (2, 2, 2))


        output_layer = Conv3D(label_num, 1, activation='relu', padding='same', kernel_initializer='he_normal')(deconv_1)
        output_layer = core.Reshape((label_num, self.img_rows * self.img_cols * self.img_frames))(output_layer)
        output_layer = core.Permute((2, 1))(output_layer)
        output_layer = core.Activation('softmax')(output_layer)
        print("output_layer shape:", output_layer.shape)
        model = Model(inputs=inputs, outputs=output_layer)

        # initiate  optimizer
        opt = optimizers.Adam(lr=1e-4)
        self.lossfn = MyDefineLoss.wrapped_partial(MyDefineLoss.combine_wieght_ce_dice_loss, weights=self.class_weights)
        model.compile(opt, loss=self.lossfn, metrics=['accuracy'])
        return model

    def train3dUnet(self,folderpath = default_train_folderpath,data_augmentation = True):
        print("loading data")
        self.load_train_data(folderpath,True)
        model = self.build_model_8x96x96(self.label_num)
        # model = self.build_model_48x48x32(self.label_num)
        print("build model done")
        ckpfile = self.checkpoint_dir + '/My3dUnet.hdf5'
        model_checkpoint = ModelCheckpoint(ckpfile, monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        print('No use data augmentation.')
        nor_train_data = self.normalization_data(self.train_imgs)
        model.fit(nor_train_data, self.train_mask_labels, batch_size=8, epochs=16, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])
