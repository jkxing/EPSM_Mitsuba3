import torch
import drjit as dr
import mitsuba as mi
import sys,os,json
import importlib
sys.path.append(".")
import cv2
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
mi.set_variant('cuda_ad_rgb')

log_level = 1

Pooler = torch.nn.AvgPool2d(kernel_size=2)
@dr.wrap_ad(source='drjit', target='torch')
def down_res_loss(st, img, img_ref):
    img = img[None,...].permute(0,3,1,2)
    img_ref = img_ref[None,...].permute(0,3,1,2)
    while st>0:
        img = Pooler(img)
        img_ref = Pooler(img_ref)
        st = st-1
    if log_level>0:
        Logger.save_img("down_res.png",img.permute(0,2,3,1)[0])
    return torch.mean((img-img_ref)**2)


if __name__=="__main__":

    method = sys.argv[1]
    config = sys.argv[2]
    Logger.init(exp_name=config+"/"+method, show=False, debug=False, path="results/",add_time=False)

    tasks = importlib.import_module(f'exp.{config}') # import specific task
    resolution = tasks.resolution #resolution
    spp = tasks.spp # spp
    scene = tasks.scene # scene
    thres = tasks.thres # for hybrid scheme
    max_depth = tasks.max_depth
    match_res = tasks.match_res

    # get target image
    if hasattr(tasks,"gt_img")==True:
        gt_img = torch.from_numpy(cv2.cvtColor(cv2.imread(tasks.gt_img),cv2.COLOR_BGR2RGB)).to(device)/255.0
        img_ref = mi.TensorXf(gt_img.reshape(-1,3))
    else:
        if hasattr(tasks,"gt_scene")==True:
            img_ref = mi.render(tasks.gt_scene, seed=0, spp=8192, sensor=0)
        else:
            img_ref = mi.render(scene, seed=0, spp=8192, sensor=0)
        img_ref = img_ref[...,:3]
        img_np = np.array(mi.util.convert_to_bitmap(img_ref))
        gt_img = torch.from_numpy(img_np).to(device)/255.0

    if log_level>0:
        Logger.save_img("gt_img.png",gt_img)

    gt_img_low= torch.from_numpy(cv2.resize(np.array(mi.util.convert_to_bitmap(img_ref)),(match_res,match_res))).to(device)/255.0

    # pixel matcher using optimal transport(Sinkhorn)
    matcher = Matcher(match_res, device)

    # get optimized parameter and transformation
    opt, apply_transformation, output, params = tasks.optim_settings()
    apply_transformation(params, opt)
    
    for key in opt.keys():
        dr.enable_grad(opt[key])
    params = mi.traverse(scene)

    # get init image
    img_init = mi.render(scene, params, seed=0, spp=512, sensor=0)
    init_img = torch.from_numpy(np.array( mi.util.convert_to_bitmap(img_init[...,:3]))).to(device)/255.0

    if log_level>0:
        Logger.save_img("init_img.png",init_img)

    # deal with hybrid scheme
    if method.endswith("hybrid"):
        method = method[:-7]
        integrator2 = mi.load_dict({
            'type': "prb_reparam",
            'max_depth': max_depth
        })
    else:
        thres = 10000
    
    # define integrator
    integrator1 = mi.load_dict({
        'type': method,
        'max_depth': max_depth
    })

    # camera settings are slightly different between EPSM and PRB.
    if method.startswith("manifold"):
        sensor_id = 1
    else:
        sensor_id = 0
    
    loop = tqdm(range(tasks.it))
        
    for it in loop:

        apply_transformation(params, opt)
        if it<thres:
            img = mi.render(scene, params, seed=it, spp=spp, integrator=integrator1, sensor=sensor_id)
        else:
            if it==thres:
                for key in opt.keys():
                    opt.reset(key)
            img = mi.render(scene, params, seed=it, spp=spp, integrator=integrator2, sensor=0)
        
        imgs = np.array(mi.util.convert_to_bitmap(img[...,:3]))

        
        if log_level>0:
            Logger.save_img(f"optim.png",imgs/255.0,flip=False)
            Logger.add_image(f"optim",imgs/255.0,flip=False)
        if log_level>1:
            Logger.save_img_2(f"optim{it}.png",imgs/255.0,flip=False)

        if img.shape[-1]==5:
            render_img = torch.from_numpy(cv2.resize(imgs,(match_res,match_res))).to(device)/255.0
            grad_ = matcher.match_Sinkhorn(render_img[...,:3].reshape(-1,3), gt_img_low[...,:3].reshape(-1,3))
            grad_ = grad_.reshape(match_res,match_res,5)
            grad_ = grad_.repeat(resolution//match_res,resolution//match_res,1)
            grad = mi.TensorXf(grad_)
            dr.backward(img*grad)
        else:
            # whether using multi-resolution loss
            # loss = down_res_loss(6-((7*it)//tasks.it),img,img_ref[...,:3])
            loss = dr.sum(dr.sqr(img - img_ref[...,:3])) / len(img)
            dr.backward(loss)
        
        try:
            # remove nan in grad
            dic = {}
            for key in opt.keys():
                x = dr.grad(opt[key])
                x[dr.isnan(x)] = 0
                dr.set_grad(opt[key],x)
                dic[key] = float(opt[key].torch().item())#.item()
            if log_level>1:
                Logger.save_param(f"param{it}.npy",dic)
        except:
            pass
        
        opt.step()
        loop.set_description(f"Iteration {it:02d}: error={output(opt)}")
    
    Logger.exit()
    img_final = mi.render(scene, params, seed=0, spp=8192, sensor=0)
    img_final = torch.from_numpy(np.array( mi.util.convert_to_bitmap(img_final[...,:3]))).to(device)/255.0
    if log_level>0:
        Logger.save_img(f"{sys.argv[1]}.png",img_final)
    
    print("finish optim")
