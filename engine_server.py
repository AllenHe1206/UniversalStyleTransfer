import base64
import io
import os
import sys
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import tensorrt as trt
from torchvision import transforms, utils

import common
from model import *
from engine import *

from flask import Flask, render_template, request, make_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/inference", methods=['GET', 'POST'])
def inference():
    c = request.files.get('content')
    c.seek(0)
    f = c.read()
    content = np.frombuffer(f, dtype=np.uint8)
    content = cv2.imdecode(content, -1)
    ORIGINAL_SHAPE = content.shape
    content = cv2.resize(content, ENCODER_INPUT_SHAPE[-2:])
    content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
    content = transforms.ToTensor()(content).unsqueeze(0).float()
    
    s = request.files.get('style')
    s.seek(0)
    f = s.read()
    style = np.frombuffer(f, dtype=np.uint8)
    style = cv2.imdecode(style, -1)
    style = cv2.resize(style, ENCODER_INPUT_SHAPE[-2:])
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
    style = transforms.ToTensor()(style).unsqueeze(0).float()

    layer_order = int(request.form.get('layer_order')) #int 1 2 3 4 5
    transfer_type = request.form.get('transfer_type') #string chained/single
    model_runtime = request.form.get('model_runtime') #string torch/trt
    
    print(model_runtime)
    print(transfer_type)
    print('layers: %d'%layer_order)
    start = time.time()
    if model_runtime == 'trt':
        content = content.detach().numpy()
        style = style.detach().numpy()
        if transfer_type == 'single':
            result = single_style_transfer(content, style, layer_order - 1)
        else:
            result = chained_style_transfer(content, style, layer_order - 1)
    else:
        content = content.detach().cuda()
        style = style.detach().cuda()
        with torch.no_grad():
            if transfer_type == 'single':
                result = torch_single_style_transfer(content, style, layer_order - 1)
            else:
                result = torch_chained_style_transfer(content, style, layer_order - 1)
            result = result.detach().to('cpu').numpy()
    
    print('time:')
    print(time.time() - start)
    # print(result.shape)
    print('max:', end=' ')
    print(result.max())
    print('min:', end=' ')
    print(result.min())
    result = result.squeeze(0)
    
    result = (result - result.min()) / (result.max() - result.min()) * 255
    result = result.transpose(1, 2, 0)
    PIL_image = Image.fromarray(result.astype(np.uint8), 'RGB')
    PIL_image = PIL_image.resize(ORIGINAL_SHAPE[:-1])
    b_img = io.BytesIO()
    PIL_image.save(b_img, 'png')
    PIL_image.save('test.png')
    encoded_img = base64.b64encode(b_img.getvalue())
    decoded_img = encoded_img.decode('utf-8')
    img_data = f'data:image/png;base64,{decoded_img}'
    return render_template('index.html', img_data=img_data)

def feature_transfer(ori_image,style):
    ori_image = torch.tensor(ori_image).double().cuda()
    style = torch.tensor(style).double().cuda()
    channel = ori_image.size(0)
    imagev = ori_image.view(channel, -1)
    stylev = style.view(channel, -1)
    imagev = imagev - torch.mean(imagev, 1).unsqueeze(1).expand_as(imagev)
    conv = torch.mm(imagev, imagev.t()).div(imagev.size()[1] - 1) + torch.eye(imagev.size()[0]).double().cuda()
    cu, ce, cv = torch.svd(conv, some=False)
    c = imagev.size()[0]

    c = len(ce[ce>=1e-5])
    s_mean = torch.mean(stylev,1)
    stylev = stylev - torch.mean(stylev,1).unsqueeze(1).expand_as(stylev)
    conv = torch.mm(stylev, stylev.t()).div(stylev.size()[1] - 1)
    su, se, sv = torch.svd(conv, some=False)

    s = len(se[se>=1e-5])
                
    #whiten
    cd = (ce[0:c]).pow(-0.5)
    cd = torch.mm(cv[:,0:c], torch.diag(cd).cuda())
    cd = torch.mm(cd, (cv[:,0:c].t()))
    whiten = torch.mm(cd, imagev)

    sd = (se[0:s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(sv[:,0:s], torch.diag(sd).cuda()), (sv[:,0:s].t())), whiten)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        
    targetFeature = targetFeature.view_as(ori_image).float().unsqueeze(0)
    return torch.clone(targetFeature)

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

    target = feature_transfer(content_feature.squeeze(0), style_feature.squeeze(0)).cpu().numpy().ravel().astype(np.float32)
    
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
    content_feature = encoder(content).squeeze(0)
    style_feature = encoder(style).squeeze(0)
    target = feature_transfer(content_feature, style_feature)
    return decoder(target)

def torch_chained_style_transfer(content, style, order):
    new_content = content
    for i in range(order, -1, -1):
        new_content = torch_single_style_transfer(new_content, style, i)
    return new_content

if __name__ == '__main__':
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

    ENCODER_INPUT_SHAPE = (1, 3, 1024, 1024)
    # dummy_input = np.random.random(INPUT_SHAPE).ravel().astype(np.float32)
    encoders = [e1, e2, e3, e4, e5]
    encoder_pop_funcs = [populate_encoder1, populate_encoder2, populate_encoder3, populate_encoder4, populate_encoder5]
    encoder_engines = [build_engine(encoder_pop_funcs[i], encoders[i].state_dict(), ENCODER_INPUT_SHAPE, 'in_e%d'%(i+1), 'out_e%d'%(i+1)) for i in range(5)]
    DECODER_INPUT_SHAPE = [(1, 64, 1024, 1024),
                           (1, 128, 512, 512),
                           (1, 256, 256, 256),
                           (1, 512, 128, 128),
                           (1, 512, 64, 64)]
    decoders = [d1, d2, d3, d4, d5]
    decoder_pop_funcs = [populate_decoder1, populate_decoder2, populate_decoder3, populate_decoder4, populate_decoder5]
    decoder_engines = [build_engine(decoder_pop_funcs[i], decoders[i].state_dict(), DECODER_INPUT_SHAPE[i], 'in_d%d'%(i+1), 'out_d%d'%(i+1)) for i in range(5)]
    app.run(host='0.0.0.0', port=8800, debug=True, threaded=False)
