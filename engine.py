import os
import sys

import numpy as np
import pycuda.autoinit
import tensorrt as trt

import common
from model import *

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def populate_encoder1(network, weights_encoder1, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_encoder1['conv1.weight'].numpy()
    conv1_b = weights_encoder1['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=3, kernel_shape=(1, 1), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    

    conv2_w = weights_encoder1['conv2.weight'].numpy()
    conv2_b = weights_encoder1['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=conv1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    relu1.get_output(0).name = output_name
    network.mark_output(tensor=relu1.get_output(0))

def populate_encoder2(network, weights_encoder2, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_encoder2['conv1.weight'].numpy()
    conv1_b = weights_encoder2['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=3, kernel_shape=(1, 1), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    
    conv2_w = weights_encoder2['conv2.weight'].numpy()
    conv2_b = weights_encoder2['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=conv1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights_encoder2['conv3.weight'].numpy()
    conv3_b = weights_encoder2['conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
    conv3.stride = (1, 1)
    conv3.padding = (1, 1)
    conv3.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu2 = network.add_activation(input=conv3.get_output(0), type=trt.ActivationType.RELU)

    pool1 = network.add_pooling(input=relu2.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv4_w = weights_encoder2['conv4.weight'].numpy()
    conv4_b = weights_encoder2['conv4.bias'].numpy()
    conv4 = network.add_convolution(input=pool1.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
    conv4.stride = (1, 1)
    conv4.padding = (1, 1)
    conv4.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu3 = network.add_activation(input=conv4.get_output(0), type=trt.ActivationType.RELU)

    relu3.get_output(0).name = output_name
    network.mark_output(tensor=relu3.get_output(0))

def populate_encoder3(network, weights_encoder3, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_encoder3['conv1.weight'].numpy()
    conv1_b = weights_encoder3['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=3, kernel_shape=(1, 1), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    
    conv2_w = weights_encoder3['conv2.weight'].numpy()
    conv2_b = weights_encoder3['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=conv1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights_encoder3['conv3.weight'].numpy()
    conv3_b = weights_encoder3['conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
    conv3.stride = (1, 1)
    conv3.padding = (1, 1)
    conv3.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu2 = network.add_activation(input=conv3.get_output(0), type=trt.ActivationType.RELU)

    pool1 = network.add_pooling(input=relu2.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv4_w = weights_encoder3['conv4.weight'].numpy()
    conv4_b = weights_encoder3['conv4.bias'].numpy()
    conv4 = network.add_convolution(input=pool1.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
    conv4.stride = (1, 1)
    conv4.padding = (1, 1)
    conv4.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu3 = network.add_activation(input=conv4.get_output(0), type=trt.ActivationType.RELU)

    conv5_w = weights_encoder3['conv5.weight'].numpy()
    conv5_b = weights_encoder3['conv5.bias'].numpy()
    conv5 = network.add_convolution(input=relu3.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv5_w, bias=conv5_b)
    conv5.stride = (1, 1)
    conv5.padding = (1, 1)
    conv5.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu4 = network.add_activation(input=conv5.get_output(0), type=trt.ActivationType.RELU)

    pool2 = network.add_pooling(input=relu4.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool2.stride = (2, 2)

    conv6_w = weights_encoder3['conv6.weight'].numpy()
    conv6_b = weights_encoder3['conv6.bias'].numpy()
    conv6 = network.add_convolution(input=pool2.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv6_w, bias=conv6_b)
    conv6.stride = (1, 1)
    conv6.padding = (1, 1)
    conv6.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu5 = network.add_activation(input=conv6.get_output(0), type=trt.ActivationType.RELU)

    relu5.get_output(0).name = output_name
    network.mark_output(tensor=relu5.get_output(0))

def populate_encoder4(network, weights_encoder4, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_encoder4['conv1.weight'].numpy()
    conv1_b = weights_encoder4['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=3, kernel_shape=(1, 1), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    
    conv2_w = weights_encoder4['conv2.weight'].numpy()
    conv2_b = weights_encoder4['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=conv1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights_encoder4['conv3.weight'].numpy()
    conv3_b = weights_encoder4['conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
    conv3.stride = (1, 1)
    conv3.padding = (1, 1)
    conv3.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu2 = network.add_activation(input=conv3.get_output(0), type=trt.ActivationType.RELU)

    pool1 = network.add_pooling(input=relu2.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv4_w = weights_encoder4['conv4.weight'].numpy()
    conv4_b = weights_encoder4['conv4.bias'].numpy()
    conv4 = network.add_convolution(input=pool1.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
    conv4.stride = (1, 1)
    conv4.padding = (1, 1)
    conv4.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu3 = network.add_activation(input=conv4.get_output(0), type=trt.ActivationType.RELU)

    conv5_w = weights_encoder4['conv5.weight'].numpy()
    conv5_b = weights_encoder4['conv5.bias'].numpy()
    conv5 = network.add_convolution(input=relu3.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv5_w, bias=conv5_b)
    conv5.stride = (1, 1)
    conv5.padding = (1, 1)
    conv5.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu4 = network.add_activation(input=conv5.get_output(0), type=trt.ActivationType.RELU)

    pool2 = network.add_pooling(input=relu4.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool2.stride = (2, 2)

    conv6_w = weights_encoder4['conv6.weight'].numpy()
    conv6_b = weights_encoder4['conv6.bias'].numpy()
    conv6 = network.add_convolution(input=pool2.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv6_w, bias=conv6_b)
    conv6.stride = (1, 1)
    conv6.padding = (1, 1)
    conv6.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu5 = network.add_activation(input=conv6.get_output(0), type=trt.ActivationType.RELU)

    conv7_w = weights_encoder4['conv7.weight'].numpy()
    conv7_b = weights_encoder4['conv7.bias'].numpy()
    conv7 = network.add_convolution(input=relu5.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv7_w, bias=conv7_b)
    conv7.stride = (1, 1)
    conv7.padding = (1, 1)
    conv7.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu6 = network.add_activation(input=conv7.get_output(0), type=trt.ActivationType.RELU)

    conv8_w = weights_encoder4['conv8.weight'].numpy()
    conv8_b = weights_encoder4['conv8.bias'].numpy()
    conv8 = network.add_convolution(input=relu6.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv8_w, bias=conv8_b)
    conv8.stride = (1, 1)
    conv8.padding = (1, 1)
    conv8.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu7 = network.add_activation(input=conv8.get_output(0), type=trt.ActivationType.RELU)

    conv9_w = weights_encoder4['conv9.weight'].numpy()
    conv9_b = weights_encoder4['conv9.bias'].numpy()
    conv9 = network.add_convolution(input=relu7.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv9_w, bias=conv9_b)
    conv9.stride = (1, 1)
    conv9.padding = (1, 1)
    conv9.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu8 = network.add_activation(input=conv9.get_output(0), type=trt.ActivationType.RELU)

    pool3 = network.add_pooling(input=relu8.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool3.stride = (2, 2)

    conv10_w = weights_encoder4['conv10.weight'].numpy()
    conv10_b = weights_encoder4['conv10.bias'].numpy()
    conv10 = network.add_convolution(input=pool3.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv10_w, bias=conv10_b)
    conv10.stride = (1, 1)
    conv10.padding = (1, 1)
    conv10.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu9 = network.add_activation(input=conv10.get_output(0), type=trt.ActivationType.RELU)

    relu9.get_output(0).name = output_name
    network.mark_output(tensor=relu9.get_output(0))

def populate_encoder5(network, weights_encoder5, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_encoder5['conv1.weight'].numpy()
    conv1_b = weights_encoder5['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=3, kernel_shape=(1, 1), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    
    conv2_w = weights_encoder5['conv2.weight'].numpy()
    conv2_b = weights_encoder5['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=conv1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights_encoder5['conv3.weight'].numpy()
    conv3_b = weights_encoder5['conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
    conv3.stride = (1, 1)
    conv3.padding = (1, 1)
    conv3.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu2 = network.add_activation(input=conv3.get_output(0), type=trt.ActivationType.RELU)

    pool1 = network.add_pooling(input=relu2.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv4_w = weights_encoder5['conv4.weight'].numpy()
    conv4_b = weights_encoder5['conv4.bias'].numpy()
    conv4 = network.add_convolution(input=pool1.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
    conv4.stride = (1, 1)
    conv4.padding = (1, 1)
    conv4.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu3 = network.add_activation(input=conv4.get_output(0), type=trt.ActivationType.RELU)

    conv5_w = weights_encoder5['conv5.weight'].numpy()
    conv5_b = weights_encoder5['conv5.bias'].numpy()
    conv5 = network.add_convolution(input=relu3.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv5_w, bias=conv5_b)
    conv5.stride = (1, 1)
    conv5.padding = (1, 1)
    conv5.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu4 = network.add_activation(input=conv5.get_output(0), type=trt.ActivationType.RELU)

    pool2 = network.add_pooling(input=relu4.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool2.stride = (2, 2)

    conv6_w = weights_encoder5['conv6.weight'].numpy()
    conv6_b = weights_encoder5['conv6.bias'].numpy()
    conv6 = network.add_convolution(input=pool2.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv6_w, bias=conv6_b)
    conv6.stride = (1, 1)
    conv6.padding = (1, 1)
    conv6.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu5 = network.add_activation(input=conv6.get_output(0), type=trt.ActivationType.RELU)

    conv7_w = weights_encoder5['conv7.weight'].numpy()
    conv7_b = weights_encoder5['conv7.bias'].numpy()
    conv7 = network.add_convolution(input=relu5.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv7_w, bias=conv7_b)
    conv7.stride = (1, 1)
    conv7.padding = (1, 1)
    conv7.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu6 = network.add_activation(input=conv7.get_output(0), type=trt.ActivationType.RELU)

    conv8_w = weights_encoder5['conv8.weight'].numpy()
    conv8_b = weights_encoder5['conv8.bias'].numpy()
    conv8 = network.add_convolution(input=relu6.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv8_w, bias=conv8_b)
    conv8.stride = (1, 1)
    conv8.padding = (1, 1)
    conv8.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu7 = network.add_activation(input=conv8.get_output(0), type=trt.ActivationType.RELU)

    conv9_w = weights_encoder5['conv9.weight'].numpy()
    conv9_b = weights_encoder5['conv9.bias'].numpy()
    conv9 = network.add_convolution(input=relu7.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv9_w, bias=conv9_b)
    conv9.stride = (1, 1)
    conv9.padding = (1, 1)
    conv9.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu8 = network.add_activation(input=conv9.get_output(0), type=trt.ActivationType.RELU)

    pool3 = network.add_pooling(input=relu8.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool3.stride = (2, 2)

    conv10_w = weights_encoder5['conv10.weight'].numpy()
    conv10_b = weights_encoder5['conv10.bias'].numpy()
    conv10 = network.add_convolution(input=pool3.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv10_w, bias=conv10_b)
    conv10.stride = (1, 1)
    conv10.padding = (1, 1)
    conv10.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu9 = network.add_activation(input=conv10.get_output(0), type=trt.ActivationType.RELU)

    conv11_w = weights_encoder5['conv11.weight'].numpy()
    conv11_b = weights_encoder5['conv11.bias'].numpy()
    conv11 = network.add_convolution(input=relu9.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv11_w, bias=conv11_b)
    conv11.stride = (1, 1)
    conv11.padding = (1, 1)
    conv11.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu10 = network.add_activation(input=conv11.get_output(0), type=trt.ActivationType.RELU)

    conv12_w = weights_encoder5['conv12.weight'].numpy()
    conv12_b = weights_encoder5['conv12.bias'].numpy()
    conv12 = network.add_convolution(input=relu10.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv12_w, bias=conv12_b)
    conv12.stride = (1, 1)
    conv12.padding = (1, 1)
    conv12.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu11 = network.add_activation(input=conv12.get_output(0), type=trt.ActivationType.RELU)

    conv13_w = weights_encoder5['conv13.weight'].numpy()
    conv13_b = weights_encoder5['conv13.bias'].numpy()
    conv13 = network.add_convolution(input=relu11.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv13_w, bias=conv13_b)
    conv13.stride = (1, 1)
    conv13.padding = (1, 1)
    conv13.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu12 = network.add_activation(input=conv13.get_output(0), type=trt.ActivationType.RELU)

    pool4 = network.add_pooling(input=relu12.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool4.stride = (2, 2)

    conv14_w = weights_encoder5['conv14.weight'].numpy()
    conv14_b = weights_encoder5['conv14.bias'].numpy()
    conv14 = network.add_convolution(input=pool4.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv14_w, bias=conv14_b)
    conv14.stride = (1, 1)
    conv14.padding = (1, 1)
    conv14.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu13 = network.add_activation(input=conv14.get_output(0), type=trt.ActivationType.RELU)

    relu13.get_output(0).name = output_name
    network.mark_output(tensor=relu13.get_output(0))

def populate_decoder1(network, weights_decoder1, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_decoder1['conv1.weight'].numpy()
    conv1_b = weights_decoder1['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=3, kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    conv1.padding = (1, 1)
    conv1.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    conv1.get_output(0).name = output_name
    network.mark_output(tensor=conv1.get_output(0))

def populate_decoder2(network, weights_decoder2, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_decoder2['conv1.weight'].numpy()
    conv1_b = weights_decoder2['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=64, kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    conv1.padding = (1, 1)
    conv1.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.RELU)

    # shape1 = network.add_shape(input=relu1.get_output(0))
    # scale1 = network.add_constant(shape=(4, ), weights=trt.Weights(np.ascontiguousarray([1, 1, 2, 2], dtype=np.int32)))
    # newshape1 = network.add_elementwise(input1=shape1.get_output(0), input2=scale1.get_output(0), op=trt.ElementWiseOperation.PROD)
    resize1 = network.add_resize(input=relu1.get_output(0))
    resize1.scales = (1, 1, 2, 2)
    resize1.resize_mode = trt.ResizeMode.LINEAR

    conv2_w = weights_decoder2['conv2.weight'].numpy()
    conv2_b = weights_decoder2['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=resize1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu2 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights_decoder2['conv3.weight'].numpy()
    conv3_b = weights_decoder2['conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu2.get_output(0), num_output_maps=3, kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
    conv3.stride = (1, 1)
    conv3.padding = (1, 1)
    conv3.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    conv3.get_output(0).name = output_name
    network.mark_output(tensor=conv3.get_output(0))

def populate_decoder3(network, weights_decoder3, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_decoder3['conv1.weight'].numpy()
    conv1_b = weights_decoder3['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=128, kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    conv1.padding = (1, 1)
    conv1.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.RELU)

    resize1 = network.add_resize(input=relu1.get_output(0))
    resize1.scales = (1, 1, 2, 2)
    resize1.resize_mode = trt.ResizeMode.LINEAR

    conv2_w = weights_decoder3['conv2.weight'].numpy()
    conv2_b = weights_decoder3['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=resize1.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu2 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights_decoder3['conv3.weight'].numpy()
    conv3_b = weights_decoder3['conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu2.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
    conv3.stride = (1, 1)
    conv3.padding = (1, 1)
    conv3.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu3 = network.add_activation(input=conv3.get_output(0), type=trt.ActivationType.RELU)

    resize2 = network.add_resize(input=relu3.get_output(0))
    resize2.scales = (1, 1, 2, 2)
    resize2.resize_mode = trt.ResizeMode.LINEAR

    conv4_w = weights_decoder3['conv4.weight'].numpy()
    conv4_b = weights_decoder3['conv4.bias'].numpy()
    conv4 = network.add_convolution(input=resize2.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
    conv4.stride = (1, 1)
    conv4.padding = (1, 1)
    conv4.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu4 = network.add_activation(input=conv4.get_output(0), type=trt.ActivationType.RELU)

    conv5_w = weights_decoder3['conv5.weight'].numpy()
    conv5_b = weights_decoder3['conv5.bias'].numpy()
    conv5 = network.add_convolution(input=relu4.get_output(0), num_output_maps=3, kernel_shape=(3, 3), kernel=conv5_w, bias=conv5_b)
    conv5.stride = (1, 1)
    conv5.padding = (1, 1)
    conv5.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    conv5.get_output(0).name = output_name
    network.mark_output(tensor=conv5.get_output(0))

def populate_decoder4(network, weights_decoder4, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_decoder4['conv1.weight'].numpy()
    conv1_b = weights_decoder4['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=256, kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    conv1.padding = (1, 1)
    conv1.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.RELU)

    resize1 = network.add_resize(input=relu1.get_output(0))
    resize1.scales = (1, 1, 2, 2)
    resize1.resize_mode = trt.ResizeMode.LINEAR

    conv2_w = weights_decoder4['conv2.weight'].numpy()
    conv2_b = weights_decoder4['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=resize1.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu2 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights_decoder4['conv3.weight'].numpy()
    conv3_b = weights_decoder4['conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu2.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
    conv3.stride = (1, 1)
    conv3.padding = (1, 1)
    conv3.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu3 = network.add_activation(input=conv3.get_output(0), type=trt.ActivationType.RELU)

    conv4_w = weights_decoder4['conv4.weight'].numpy()
    conv4_b = weights_decoder4['conv4.bias'].numpy()
    conv4 = network.add_convolution(input=relu3.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
    conv4.stride = (1, 1)
    conv4.padding = (1, 1)
    conv4.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu4 = network.add_activation(input=conv4.get_output(0), type=trt.ActivationType.RELU)

    conv5_w = weights_decoder4['conv5.weight'].numpy()
    conv5_b = weights_decoder4['conv5.bias'].numpy()
    conv5 = network.add_convolution(input=relu4.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv5_w, bias=conv5_b)
    conv5.stride = (1, 1)
    conv5.padding = (1, 1)
    conv5.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu5 = network.add_activation(input=conv5.get_output(0), type=trt.ActivationType.RELU)

    resize2 = network.add_resize(input=relu5.get_output(0))
    resize2.scales = (1, 1, 2, 2)
    resize2.resize_mode = trt.ResizeMode.LINEAR

    conv6_w = weights_decoder4['conv6.weight'].numpy()
    conv6_b = weights_decoder4['conv6.bias'].numpy()
    conv6 = network.add_convolution(input=resize2.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv6_w, bias=conv6_b)
    conv6.stride = (1, 1)
    conv6.padding = (1, 1)
    conv6.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu6 = network.add_activation(input=conv6.get_output(0), type=trt.ActivationType.RELU)

    conv7_w = weights_decoder4['conv7.weight'].numpy()
    conv7_b = weights_decoder4['conv7.bias'].numpy()
    conv7 = network.add_convolution(input=relu6.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv7_w, bias=conv7_b)
    conv7.stride = (1, 1)
    conv7.padding = (1, 1)
    conv7.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu7 = network.add_activation(input=conv7.get_output(0), type=trt.ActivationType.RELU)

    resize3 = network.add_resize(input=relu7.get_output(0))
    resize3.scales = (1, 1, 2, 2)
    resize3.resize_mode = trt.ResizeMode.LINEAR

    conv8_w = weights_decoder4['conv8.weight'].numpy()
    conv8_b = weights_decoder4['conv8.bias'].numpy()
    conv8 = network.add_convolution(input=resize3.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv8_w, bias=conv8_b)
    conv8.stride = (1, 1)
    conv8.padding = (1, 1)
    conv8.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu8 = network.add_activation(input=conv8.get_output(0), type=trt.ActivationType.RELU)

    conv9_w = weights_decoder4['conv9.weight'].numpy()
    conv9_b = weights_decoder4['conv9.bias'].numpy()
    conv9 = network.add_convolution(input=relu8.get_output(0), num_output_maps=3, kernel_shape=(3, 3), kernel=conv9_w, bias=conv9_b)
    conv9.stride = (1, 1)
    conv9.padding = (1, 1)
    conv9.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    conv9.get_output(0).name = output_name
    network.mark_output(tensor=conv9.get_output(0))

def populate_decoder5(network, weights_decoder5, input_name, input_shape, output_name):
    input_tensor = network.add_input(name=input_name, dtype=trt.float32, shape=input_shape)

    conv1_w = weights_decoder5['conv1.weight'].numpy()
    conv1_b = weights_decoder5['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=512, kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    conv1.padding = (1, 1)
    conv1.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu1 = network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.RELU)

    resize1 = network.add_resize(input=relu1.get_output(0))
    resize1.scales = (1, 1, 2, 2)
    resize1.resize_mode = trt.ResizeMode.LINEAR

    conv2_w = weights_decoder5['conv2.weight'].numpy()
    conv2_b = weights_decoder5['conv2.bias'].numpy()
    conv2 = network.add_convolution(input=resize1.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)
    conv2.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu2 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights_decoder5['conv3.weight'].numpy()
    conv3_b = weights_decoder5['conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu2.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
    conv3.stride = (1, 1)
    conv3.padding = (1, 1)
    conv3.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu3 = network.add_activation(input=conv3.get_output(0), type=trt.ActivationType.RELU)

    conv4_w = weights_decoder5['conv4.weight'].numpy()
    conv4_b = weights_decoder5['conv4.bias'].numpy()
    conv4 = network.add_convolution(input=relu3.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
    conv4.stride = (1, 1)
    conv4.padding = (1, 1)
    conv4.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu4 = network.add_activation(input=conv4.get_output(0), type=trt.ActivationType.RELU)

    conv5_w = weights_decoder5['conv5.weight'].numpy()
    conv5_b = weights_decoder5['conv5.bias'].numpy()
    conv5 = network.add_convolution(input=relu4.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv5_w, bias=conv5_b)
    conv5.stride = (1, 1)
    conv5.padding = (1, 1)
    conv5.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu5 = network.add_activation(input=conv5.get_output(0), type=trt.ActivationType.RELU)

    resize2 = network.add_resize(input=relu5.get_output(0))
    resize2.scales = (1, 1, 2, 2)
    resize2.resize_mode = trt.ResizeMode.LINEAR

    conv6_w = weights_decoder5['conv6.weight'].numpy()
    conv6_b = weights_decoder5['conv6.bias'].numpy()
    conv6 = network.add_convolution(input=resize2.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv6_w, bias=conv6_b)
    conv6.stride = (1, 1)
    conv6.padding = (1, 1)
    conv6.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu6 = network.add_activation(input=conv6.get_output(0), type=trt.ActivationType.RELU)

    conv7_w = weights_decoder5['conv7.weight'].numpy()
    conv7_b = weights_decoder5['conv7.bias'].numpy()
    conv7 = network.add_convolution(input=relu6.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv7_w, bias=conv7_b)
    conv7.stride = (1, 1)
    conv7.padding = (1, 1)
    conv7.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu7 = network.add_activation(input=conv7.get_output(0), type=trt.ActivationType.RELU)

    conv8_w = weights_decoder5['conv8.weight'].numpy()
    conv8_b = weights_decoder5['conv8.bias'].numpy()
    conv8 = network.add_convolution(input=relu7.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=conv8_w, bias=conv8_b)
    conv8.stride = (1, 1)
    conv8.padding = (1, 1)
    conv8.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu8 = network.add_activation(input=conv8.get_output(0), type=trt.ActivationType.RELU)

    conv9_w = weights_decoder5['conv9.weight'].numpy()
    conv9_b = weights_decoder5['conv9.bias'].numpy()
    conv9 = network.add_convolution(input=relu8.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv9_w, bias=conv9_b)
    conv9.stride = (1, 1)
    conv9.padding = (1, 1)
    conv9.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu9 = network.add_activation(input=conv9.get_output(0), type=trt.ActivationType.RELU)

    resize3 = network.add_resize(input=relu9.get_output(0))
    resize3.scales = (1, 1, 2, 2)
    resize3.resize_mode = trt.ResizeMode.LINEAR

    conv10_w = weights_decoder5['conv10.weight'].numpy()
    conv10_b = weights_decoder5['conv10.bias'].numpy()
    conv10 = network.add_convolution(input=resize3.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=conv10_w, bias=conv10_b)
    conv10.stride = (1, 1)
    conv10.padding = (1, 1)
    conv10.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu10 = network.add_activation(input=conv10.get_output(0), type=trt.ActivationType.RELU)

    conv11_w = weights_decoder5['conv11.weight'].numpy()
    conv11_b = weights_decoder5['conv11.bias'].numpy()
    conv11 = network.add_convolution(input=relu10.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv11_w, bias=conv11_b)
    conv11.stride = (1, 1)
    conv11.padding = (1, 1)
    conv11.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu11 = network.add_activation(input=conv11.get_output(0), type=trt.ActivationType.RELU)

    resize4 = network.add_resize(input=relu11.get_output(0))
    resize4.scales = (1, 1, 2, 2)
    resize4.resize_mode = trt.ResizeMode.LINEAR

    conv12_w = weights_decoder5['conv12.weight'].numpy()
    conv12_b = weights_decoder5['conv12.bias'].numpy()
    conv12 = network.add_convolution(input=resize4.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=conv12_w, bias=conv12_b)
    conv12.stride = (1, 1)
    conv12.padding = (1, 1)
    conv12.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    relu12 = network.add_activation(input=conv12.get_output(0), type=trt.ActivationType.RELU)

    conv13_w = weights_decoder5['conv13.weight'].numpy()
    conv13_b = weights_decoder5['conv13.bias'].numpy()
    conv13 = network.add_convolution(input=relu12.get_output(0), num_output_maps=3, kernel_shape=(3, 3), kernel=conv13_w, bias=conv13_b)
    conv13.stride = (1, 1)
    conv13.padding = (1, 1)
    conv13.padding_mode = trt.PaddingMode.CAFFE_ROUND_UP

    conv13.get_output(0).name = output_name
    network.mark_output(tensor=conv13.get_output(0))

def build_engine(populate_func, weights, input_shape, input_name, output_name):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)
    config.max_workspace_size = common.GiB(1)
    populate_func(network, weights, input_name, input_shape, output_name)
    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)

def main():
    dummy_input = np.random.random((1, 3, 512, 512)).ravel().astype(np.float32)

    e1 = encoder1()
    engine = build_engine(populate_encoder1, e1.state_dict(), (1, 3, 512, 512), 'in_e1', 'out_e1')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    e2 = encoder2()
    engine = build_engine(populate_encoder2, e2.state_dict(), (1, 3, 512, 512), 'in_e2', 'out_e2')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    e3 = encoder3()
    engine = build_engine(populate_encoder3, e3.state_dict(), (1, 3, 512, 512), 'in_e3', 'out_e3')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    e4 = encoder4()
    engine = build_engine(populate_encoder4, e4.state_dict(), (1, 3, 512, 512), 'in_e4', 'out_e4')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    e5 = encoder5()
    engine = build_engine(populate_encoder5, e5.state_dict(), (1, 3, 512, 512), 'in_e5', 'out_e5')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    dummy_input1 = np.random.random((1, 64, 512, 512)).ravel().astype(np.float32)
    dummy_input2 = np.random.random((1, 128, 256, 256)).ravel().astype(np.float32)
    dummy_input3 = np.random.random((1, 256, 128, 128)).ravel().astype(np.float32)
    dummy_input4 = np.random.random((1, 512, 64, 64)).ravel().astype(np.float32)
    dummy_input5 = np.random.random((1, 512, 32, 32)).ravel().astype(np.float32)

    d1 = decoder1()
    engine = build_engine(populate_decoder1, d1.state_dict(), (1, 64, 512, 512), 'in_d1', 'out_d1')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input1)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    d2 = decoder2()
    engine = build_engine(populate_decoder2, d2.state_dict(), (1, 128, 256, 256), 'in_d2', 'out_d2')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input2)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    d3 = decoder3()
    engine = build_engine(populate_decoder3, d3.state_dict(), (1, 256, 128, 128), 'in_d3', 'out_d3')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input3)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    d4 = decoder4()
    engine = build_engine(populate_decoder4, d4.state_dict(), (1, 512, 64, 64), 'in_d4', 'out_d4')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input4)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

    d5 = decoder5()
    engine = build_engine(populate_decoder5, d5.state_dict(), (1, 512, 32, 32), 'in_d5', 'out_d5')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    np.copyto(inputs[0].host, dummy_input5)
    [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(output.shape)

if __name__ == '__main__':
    main()
