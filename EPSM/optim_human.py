import torch
import drjit as dr
import mitsuba as mi
import sys,os,json
sys.path.append(".")
import cv2
import numpy as np
import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
from utils.logger import Logger
from utils.matcher import Matcher
from mitsuba.scalar_rgb import Transform4f as T
from tqdm.std import tqdm
import importlib
mi.set_variant('cuda_ad_rgb')
    
if __name__=="__main__":
    method = sys.argv[1]
    config = sys.argv[2]
    Logger.init(exp_name=config+"/"+method, show=False, debug=False, path="results/",add_time=False)
    tasks = importlib.import_module(f'exp.{config}') # import specific task
    
    resolution = tasks.resolution
    spp = tasks.spp
    scene = tasks.scene
    thres = tasks.thres
    max_depth = tasks.max_depth
    match_res = tasks.match_res

    denoiser = mi.OptixDenoiser((resolution,resolution))
    if hasattr(tasks,"gt_img")==True:
        gt_img = torch.from_numpy(cv2.cvtColor(cv2.imread(tasks.gt_img),cv2.COLOR_BGR2RGB)).to(device)/255.0
        img_ref = mi.TensorXf(gt_img.reshape(-1,3))
    else:
        if hasattr(tasks,"gt_scene")==True:
            img_ref = mi.render(tasks.gt_scene, seed=0, spp=512, sensor=0)
        else:
            img_ref = mi.render(scene, seed=0, spp=512, sensor=0)
        img_ref = img_ref[...,:3]
        img_np = np.array(mi.util.convert_to_bitmap(img_ref))
        #img_ref = denoiser(img_ref)
        gt_img = torch.from_numpy(img_np).to(device)/255.0
    Logger.save_img("gt_img.png",gt_img)
    #exit()
    gt_img_low= torch.from_numpy(cv2.resize(np.array(mi.util.convert_to_bitmap(img_ref)),(256,256))).to(device)/255.0

    matcher = Matcher(match_res, device)


    model = tasks.model

    pose_params_opt = ((torch.rand(1, 72) - 0.5) * 0.0).cuda()
    pose_params_opt.requires_grad = True
    shape_params = (torch.rand(1, 10) * 0.0).cuda()

    optimizer = torch.optim.Adam([pose_params_opt], lr=0.005)
    init_verts = model.gen_mesh(pose_params_opt,shape_params)[0]
    params = mi.traverse(scene)
    optim_vert = mi.Point3f(init_verts)
    trafo = mi.Transform4f.translate([0.05,30.0,15]).scale(10).rotate([1,0,0],90)
    params["human.vertex_positions"] = dr.ravel(trafo@optim_vert)
    params.update()
    img_init = mi.render(scene, params, seed=0, spp=512, sensor=0)
    init_img = torch.from_numpy(np.array( mi.util.convert_to_bitmap(img_init[...,:3]))).to(device)/255.0
    Logger.save_img("init_img.png",init_img)
    

    if method.endswith("hybrid"):
        method = method[:-7]
        integrator2 = mi.load_dict({
            'type': "prb_reparam",
            'max_depth': max_depth
        })
    else:
        thres = 10000
    
    integrator1 = mi.load_dict({
        'type': method,
        'max_depth': max_depth
    })

    if method.startswith("manifold"):
        sensor_id = 1
    else:
        sensor_id = 0
    
    loop = tqdm(range(tasks.it))
    for it in loop:
        optimizer.zero_grad()
        with torch.no_grad():
            pose_params_opt.clamp_(-0.1,0.1)
        verts = model.gen_mesh(pose_params_opt,shape_params)[0]
        params = mi.traverse(scene)
        optim_vert = mi.Point3f(verts)
        dr.enable_grad(optim_vert)
        trafo = mi.Transform4f.translate([0.05,30.0,15]).scale(10).rotate([1,0,0],90)
        params["human.vertex_positions"] = dr.ravel(trafo@optim_vert)
        params.update()

        if it<thres:
            img = mi.render(scene, params, seed=it, spp=spp, integrator=integrator1, sensor=sensor_id)
        else:
            img = mi.render(scene, params, seed=it, spp=spp, integrator=integrator2, sensor=0)

        imgs = np.array(mi.util.convert_to_bitmap(img[...,:3]))
        Logger.add_image(f"optim",imgs/255.0,flip=False)
        Logger.save_img(f"optim.png",imgs/255.0,flip=False)
        Logger.save_img_2(f"optim{it}.png",imgs/255.0,flip=False)
        if img.shape[-1]==5:
            render_img = torch.from_numpy(cv2.resize(imgs,(match_res,match_res))).to(device)/255.0
            grad_ = matcher.match_Sinkhorn(render_img[...,:3].reshape(-1,3), gt_img_low[...,:3].reshape(-1,3))
            grad_ = grad_.reshape(match_res,match_res,5)
            grad_ = grad_.repeat(resolution//match_res,resolution//match_res,1)
            grad = mi.TensorXf(grad_)
            dr.backward(img*grad)
        else:
            loss = dr.sum(dr.sqr(img - img_ref[...,:3])) / len(img)
            dr.backward(loss)

        x = dr.grad(optim_vert)
        x[dr.isnan(x)] = 0
        grad = x.torch()
        dr.set_grad(optim_vert,0)
        loss = torch.sum(verts*grad)
        loss.backward()
        optimizer.step()
    Logger.exit()
    img_final = mi.render(scene, params, seed=0, spp=8192, sensor=0)
    img_final = torch.from_numpy(np.array( mi.util.convert_to_bitmap(img_final[...,:3]))).to(device)/255.0
    Logger.save_img(f"{sys.argv[1]}.png",img_final)
    print("finish optim")


    