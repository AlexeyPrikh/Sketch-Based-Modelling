import torch
from torchvision.transforms import Normalize, ToPILImage
import numpy as np
import cv2
import argparse
import json

from spin.models import hmr, SMPL
from spin.utils.imutils import crop
from spin.utils.renderer import Renderer
import spin.config
import spin.constants

import urllib
from pathlib import Path
from textwrap import dedent

import cv2
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from g2pe.src.utils import get_final_preds, get_input, put_kps, KPS, SKELETON

from utils import ROOT_PATH, ASSETS_PATH, DESIRED_SIZE, DEFAULT_IMG_FILEPATH, DEFAULT_IMG_URLPATH, MODEL_FILEPATH, MODEL_URLPATH, KPS_SPIN_MAP

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from os import listdir
import glob
import os
from random import randint 
from utils import MODEL_HMR_PATH, LOAD_DIR, SAVE_DIR, IDXS_MAP

def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

def process_image_spin(img_file, input_res=224):
    normalize_img = Normalize(mean=spin.constants.IMG_NORM_MEAN, std=spin.constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() 
    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

def g2pe_coordinate_transformation(img, predicted_keypoints):
    img = Image.fromarray(img[..., ::-1])
    original_img_size = img.size
    ratio = float(DESIRED_SIZE) / max(original_img_size)
    resized_img_size = tuple([int(x * ratio) for x in original_img_size])
    img = img.resize(resized_img_size, Image.ANTIALIAS)
    original_img_size = max(original_img_size)
    predicted_keypoints *= (DESIRED_SIZE - 1) / (original_img_size - 1)
    return predicted_keypoints, img, resized_img_size


def process_image(img_raw, model):

    pose_input, img, center, scale = get_input(img_raw)
    model.setInput(pose_input[None])
    predicted_heatmap = model.forward()
    predicted_keypoints, confidence = get_final_preds(
        predicted_heatmap, center[None], scale[None], post_process=True
    )
    predicted_keypoints, confidence, predicted_heatmap = (
        predicted_keypoints[0],
        confidence[0],
        predicted_heatmap[0],
    )
    predicted_keypoints, img, resized_img_size = g2pe_coordinate_transformation(img, predicted_keypoints)

    predicted_heatmap = predicted_heatmap.sum(0)
    predicted_heatmap_min = predicted_heatmap.min()
    predicted_heatmap_max = predicted_heatmap.max()
    predicted_heatmap = (predicted_heatmap - predicted_heatmap_min) / (
        predicted_heatmap_max - predicted_heatmap_min
    )
    predicted_heatmap = Image.fromarray((predicted_heatmap * 255).astype("uint8"))
    predicted_heatmap = predicted_heatmap.resize(resized_img_size, Image.ANTIALIAS)

    return img, predicted_keypoints, confidence, predicted_heatmap

def scale_g2pe(predicted_keypoints, size_sp, size_kp):
    # scale to img_sp
    m1,n1 = size_kp
    if (m1>n1): # vertically
        ax1=1
        ax2=0
    else: # horizontal
        ax1=0
        ax2=1

    shift = (size_kp[ax1]-size_kp[ax2])/2
    scale = size_sp[ax2]/size_kp[ax2]
    predicted_keypoints_scaled = np.copy(predicted_keypoints)
    predicted_keypoints_scaled[:,ax2] -= shift
    predicted_keypoints_scaled *= scale
    
    return predicted_keypoints_scaled
    
    
def get_g2pe_estim(image_path, model_g2pe, size_sp) :
    # use 2d keypoints detection
    with open(image_path, "br") as inp:
        img_raw = inp.read()
    img_kp, predicted_keypoints, confidence, predicted_heatmap = process_image(img_raw, model_g2pe)

    # scale to img_sp
    size_kp = np.asarray(img_kp).shape[:2]
    predicted_keypoints_scaled = scale_g2pe(predicted_keypoints, size_sp, size_kp)
    y_g2pe_estim = torch.tensor(predicted_keypoints_scaled)
    return y_g2pe_estim

def scale_spin(pred_output, pred_camera,size_sp):
    pred_vertices = pred_output.vertices
    joints_pred_3d = pred_output.joints
    joints_pred_2d = joints_pred_3d[0,:,:2]
    pred_camera = pred_camera.detach()
    coef = pred_camera[:,0]
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*spin.constants.FOCAL_LENGTH/(spin.constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()

    #scale to img_sp
    (m, n) = size_sp
    joints_pred_2d_scaled = joints_pred_2d
    joints_pred_2d_scaled[:,0] = m/2 * (coef*(joints_pred_2d_scaled[:,0]+camera_translation[0]) + 1)
    joints_pred_2d_scaled[:,1] = n/2 * (coef*(joints_pred_2d_scaled[:,1]+camera_translation[1]) + 1)
    return joints_pred_2d_scaled
    
def get_pred(img_data, model_hmr, smpl):

    # use spin model => joints_pred_2d
    img_sp, norm_img, size_sp =  img_data
    pred_rotmat, pred_betas, pred_camera = model_hmr(norm_img.to(get_device()))
    pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], 
                       global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    
    y_pred = scale_spin(pred_output, pred_camera, size_sp)
    y_pred = y_pred[IDXS_MAP]
    
    return y_pred

def plot_pred_and_data_skeleton(y_pred, y_g2pe_estim, img_data, filename,
                                keypoints=KPS, skeleton=SKELETON, width=1, rad=2):
    img_sp, norm_img, size_sp = img_data
    
    trans = ToPILImage(mode='RGB')
    img = trans(img_sp.squeeze())
    draw = ImageDraw.Draw(img)
    
    kps = y_g2pe_estim
    for name, (x, y) in zip(keypoints, kps):
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill="red", width=width)
    for src, dst in skeleton:
        x1, y1 = kps[src - 1]
        x2, y2 = kps[dst - 1]
        draw.line([x1, y1, x2, y2], fill="red", width=width)
        
        
    kps = y_pred.detach()
    for name, (x, y) in zip(keypoints, kps):
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill="blue", width=width)
    for src, dst in skeleton:
        x1, y1 = kps[src - 1]
        x2, y2 = kps[dst - 1]
        draw.line([x1, y1, x2, y2], fill="blue", width=width)
    
    
    fig, ax = plt.subplots(facecolor='lightgray',figsize=(16,10))
    ax.imshow(img)
    plt.title('Before : pred->blue; g2pe_estim->red')
    path = SAVE_DIR+'/' + filename + '.png'
    plt.savefig(path)
    plt.show()
    return img


def save_3d_model_on_img(img_data, model_hmr, filename):
    
    img_sp, norm_img, size_sp = img_data
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model_hmr(norm_img.to(device))
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], 
                           global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
    
    renderer = Renderer(focal_length=spin.constants.FOCAL_LENGTH, img_res=spin.constants.IMG_RES, faces=smpl.faces)
    
    pred_v = pred_vertices
    img_ = img
    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*spin.constants.FOCAL_LENGTH/(spin.constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_v[0].cpu().numpy()
    img = img_.permute(1,2,0).cpu().numpy()
    # Render parametric shape
    img_shape = renderer(pred_vertices, camera_translation, img)

    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center

    # Render non-parametric shape
    img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))
    plt.imshow(img_shape_side)
    plt.savefig(save_path+'/shape__rot_'+filename)
    plt.imshow(img_shape)
    plt.savefig(save_path+'/shape_'+filename)
    
def loss_parallel(y_pred, y_data):
    L = 0
    for src, dst in SKELETON:
        v1 = y_pred[dst - 1]- y_pred[src - 1]
        v2 = y_data[dst - 1]- y_data[src - 1]
        cos = 1. - torch.sum(v1*v2)/(torch.norm(v1)*torch.norm(v2))
        L += cos
    return L

def freeze_layer(model_hmr):
    for module in model_hmr.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval() 
        if isinstance(module, torch.nn.modules.Dropout):
            module.eval()  
    
    
def main():
    #Fix device
    print('Calculations on device ', get_device())
    
    args = parser.parse_args()
    
# get img_name
    dataset_names = glob.glob(LOAD_DIR + '*.jpg')
    print('Number of sketches : ', len(dataset_names))
    if args.img ==None:
        num_of_im = randint(0, len(dataset_names)-1)
        # !!!
        img_name = dataset_names[num_of_im]
    else:
        img_name = args.img
    print('Path to image : ', img_name)
    
# Load g2pe model
    if not MODEL_FILEPATH.exists():
        urllib.request.urlretrieve(MODEL_URLPATH, MODEL_FILEPATH)

    model_g2pe = cv2.dnn.readNetFromONNX(str(MODEL_FILEPATH))

    
# Load img_sp
    img, norm_img = process_image_spin(img_name, input_res=spin.constants.IMG_RES)
    img_array = np.asarray(img)
    size_sp = (img_array.shape[1],img_array.shape[2])
    
    img_data = img, norm_img, size_sp  
    
# get y_data
    y_g2pe_estim = get_g2pe_estim(img_name, model_g2pe, size_sp)
    
    
# Load pretrained model
    model_hmr = hmr(spin.config.SMPL_MEAN_PARAMS).to(get_device())
    checkpoint = torch.load(MODEL_HMR_PATH, map_location='cpu')
    model_hmr.load_state_dict(checkpoint['model'], strict=False)
# Load SMPL model
    smpl = SMPL(spin.config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(get_device())

    
# Freeze layers
    model_hmr.train()
    freeze_layer(model_hmr)
     
    with torch.no_grad():
        y_pred_before = get_pred(img_data, model_hmr, smpl)
        
# Save image
    img_before = plot_pred_and_data_skeleton(y_pred_before, y_g2pe_estim, img_data,filename='skeleton_before')
    
# Start optimization
    loss_mse = torch.nn.MSELoss()

    C = 1000
    lr = 1e-6
    EPOCHS = 50

    optimizer = optim.SGD(model_hmr.parameters(), lr=lr) # 6
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.5, 
                                  verbose=True, patience=2)

    print('Start training')
    y_g2pe_estim.to(device=get_device())
    list_loss=[]
    for epoch in range(EPOCHS):  
        optimizer.zero_grad()
        y_pred = get_pred(img_data, model_hmr, smpl)
        loss_value = loss_mse(y_pred, y_g2pe_estim) + C*loss_parallel(y_pred, y_g2pe_estim)
        epoch_loss = loss_value.item()
        loss_value.backward() 
        optimizer.step()
        scheduler.step(epoch_loss)
        print('Epoch_loss ', epoch, ' = ', epoch_loss)
        list_loss.append(epoch_loss)

    print('Finished training: ','loss before = ', list_loss[0],'; loss after = ', list_loss[-1])
    y_g2pe_estim.cpu()
    with torch.no_grad():
        y_pred_after = get_pred(img_data, model_hmr, smpl).cpu()
    
        
# Save image
    img_after = plot_pred_and_data_skeleton(y_pred_after, y_g2pe_estim, img_data, filename='skeleton_after')
    
    print('The end')

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default=None, help='Path to input image')
 
if __name__ == '__main__':
    main()