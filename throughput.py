import argparse
import time

import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import common
from model import *
from engine import *
from engine_server import feature_transfer

def single_style_transfer(content, style, order): # order:0, 1, 2, 3, 4 BCHW
    content = content.ravel().astype(np.float32)
    style = style.ravel().astype(np.float32)
    
    encoder_engine = encoder_engines[order]
    decoder_engine = decoder_engines[order]
    inputs, outputs, bindings, stream = common.allocate_buffers(encoder_engine)
    context = encoder_engine.create_execution_context()
    np.copyto(inputs[0].host, content)
    [content_feature] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    content_feature = content_feature.reshape(DECODER_INPUT_SHAPE[order])
    inputs, outputs, bindings, stream = common.allocate_buffers(encoder_engine)
    context = encoder_engine.create_execution_context()
    np.copyto(inputs[0].host, style)
    [style_feature] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    style_feature = style_feature.reshape(DECODER_INPUT_SHAPE[order])
    target = torch.cat([feature_transfer(c_f, s_f) for c_f, s_f in zip(content_feature, style_feature)]).cpu().numpy().ravel().astype(np.float32)
    inputs, outputs, bindings, stream = common.allocate_buffers(decoder_engine)
    context = decoder_engine.create_execution_context()
    np.copyto(inputs[0].host, target)
    [result] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    result = result.reshape(ENCODER_INPUT_SHAPE)
    return result

def chained_style_transfer(content, style, order): #BCHW
    new_content = content
    for i in range(order, -1, -1):
        new_content = single_style_transfer(new_content, style, i)
    return new_content

def torch_single_style_transfer(content, style, order):
    encoder = encoders[order].cuda()
    decoder = decoders[order].cuda()
    content_feature = encoder(content)
    style_feature = encoder(style)
    target = torch.cat([feature_transfer(c_f, s_f) for c_f, s_f in zip(content_feature, style_feature)])
    return decoder(target)

def torch_chained_style_transfer(content, style, order):
    new_content = content
    for i in range(order, -1, -1):
        new_content = torch_single_style_transfer(new_content, style, i)
    return new_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='torch vs trt throughput testing')
    parser.add_argument('--batch', type=int, default=4, help='batch size to test, default=4')
    parser.add_argument('--engine', type=str, help='which engine to test, trt or torch')
    parser.add_argument('--iters', type=int, default=100, help='iterations to run, default=100')
    parser.add_argument('--layers', type=int, help='numbers of encoder-decoder pairs to use')
    parser.add_argument('--method', type=str, help='transfer method, single or chained')
    args = parser.parse_args()

    e1 = encoder1()
    e2 = encoder2()
    e3 = encoder3()
    e4 = encoder4()
    e5 = encoder5()
    d1 = decoder1()
    d2 = decoder2()
    d3 = decoder3()
    d4 = decoder4()
    d5 = decoder5()
    e1.load_state_dict(torch.load('encoder1.pth'))
    e2.load_state_dict(torch.load('encoder2.pth'))
    e3.load_state_dict(torch.load('encoder3.pth'))
    e4.load_state_dict(torch.load('encoder4.pth'))
    e5.load_state_dict(torch.load('encoder5.pth'))
    d1.load_state_dict(torch.load('decoder1.pth'))
    d2.load_state_dict(torch.load('decoder2.pth'))
    d3.load_state_dict(torch.load('decoder3.pth'))
    d4.load_state_dict(torch.load('decoder4.pth'))
    d5.load_state_dict(torch.load('decoder5.pth'))

    content = torch.rand(args.batch, 3, 1024, 1024)
    style = torch.rand(args.batch, 3, 1024, 1024)

    ENCODER_INPUT_SHAPE = (args.batch, 3, 1024, 1024)
    encoders = [e1, e2, e3, e4, e5]
    encoder_pop_funcs = [populate_encoder1, populate_encoder2, populate_encoder3, populate_encoder4, populate_encoder5]

    DECODER_INPUT_SHAPE = [(args.batch, 64, 1024, 1024),
                                  (args.batch, 128, 512, 512),
                                  (args.batch, 256, 256, 256),
                                  (args.batch, 512, 128, 128),
                                  (args.batch, 512, 64, 64)]
    decoders = [d1, d2, d3, d4, d5]
    decoder_pop_funcs = [populate_decoder1, populate_decoder2, populate_decoder3, populate_decoder4, populate_decoder5]

    if args.engine == 'torch':
        content = content.detach().cuda()
        style = style.detach().cuda()
        with torch.no_grad():
            start = time.time()
            for i in range(args.iters):
                if args.method == 'single':
                    result = torch_single_style_transfer(content, style, args.layers - 1)
                else:
                    result = torch_chained_style_transfer(content, style, args.layers - 1)
            stop = time.time()
    else:
        encoder_engines = [build_engine(encoder_pop_funcs[i], encoders[i].state_dict(), ENCODER_INPUT_SHAPE, 'in_e%d'%(i+1), 'out_e%d'%(i+1)) for i in range(5)]
        decoder_engines = [build_engine(decoder_pop_funcs[i], decoders[i].state_dict(), DECODER_INPUT_SHAPE[i], 'in_d%d'%(i+1), 'out_d%d'%(i+1)) for i in range(5)]
        content = content.detach().numpy()
        style = style.detach().numpy()
        start = time.time()
        for i in range(args.iters):
            if args.method == 'single':
                result = single_style_transfer(content, style, args.layers - 1)
            else:
                result = chained_style_transfer(content, style, args.layers - 1)
        stop = time.time()
    print(args)
    print('time: %f'%(stop - start))
    print('total images: %d'%(args.batch * args.iters))
    print('throughput: %f'%(args.batch * args.iters / (stop - start)))

