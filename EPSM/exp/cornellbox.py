import mitsuba as mi
import drjit as dr
import os,sys
import torch
import numpy as np
import math,random
it=500 # 500
spp=256
resolution=512
thres = 375
max_depth = 6
match_res = 128

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
from mitsuba.scalar_rgb import Transform4f as T

num = 6
rgb = [[100,0,0], [100,100,0],[0,100,0],[0,100,100],[0,0,100],[100,0,100]]
angle  = [math.pi*2*i/num-math.pi/2 for i in range(num)]
init_rot = math.pi/3

def load_light():
    base={"type":"scene"}
    for i in range(num):
        x = 0.5*math.sin(angle[i])
        y = 0.5*math.cos(angle[i])
        x1 = 0.51*math.sin(angle[i])
        y1 = 0.51*math.cos(angle[i])
        base["light"+str(i)] =  {
            "type": "obj",
            "filename": "data/meshes/rectangle.obj",
            "to_world": mi.ScalarTransform4f.look_at(origin=[x,1.0+y,0.1],
                                                    target=[0,1.0,-0.3],
                                                    up=[0, 0, 1]).scale(0.05),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': rgb[i],
                }
            }
        }
        base["lightbar"+str(i)] =  {
            "type": "obj",
            "filename": "data/meshes/bar2.obj",
            "to_world": mi.ScalarTransform4f.look_at(origin=[x,1.0+y,0.1],
                                                    target=[0,1.0,-0.3],
                                                    up=[0, 0, 1]).scale(0.05),
            'bsdf': {
                'type': 'twosided',
                'material': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': 0.4
                    }
                }
            }
        }
    return base

def load_light2():
    base={"type":"scene"}
    for i in range(num):
        x = 0.5*math.sin(angle[i])
        y = 0.5*math.cos(angle[i])
        base["light"+str(i)] =  {
            "type": "obj",
            "filename": "data/meshes/rectangle.obj",
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': rgb[i],
                }
            }
        }
        base["lightbar"+str(i)] =  {
            "type": "obj",
            "filename": "data/meshes/bar2.obj",
            'bsdf': {
                'type': 'twosided',
                'material': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': 0.4
                    }
                }
            }
        }
    return base

scene_file = "data/scenes/cornellbox.xml"
gt_scene = mi.load_file(scene_file)
params = mi.traverse(gt_scene)
scene_file = "data/scenes/cornellbox2.xml"
scene = mi.load_file(scene_file)
params = mi.traverse(scene)
def optim_settings():
    opt = mi.ad.Adam(lr=0.01)
    init = {}
    for i in range(num):
        ori_rot = init_rot
        opt[f"rot{i}"] = mi.Float(ori_rot)
        init[f"light{i}"] = dr.unravel(mi.Point3f,params[f'light{i}.vertex_positions'])
        init[f"lightbar{i}"] = dr.unravel(mi.Point3f,params[f'lightbar{i}.vertex_positions'])
    def apply_transformation(params, opt):
        for i in range(num):
            x = 0.5*dr.sin(opt[f"rot{i}"]+angle[i])
            y = 0.5*dr.cos(opt[f"rot{i}"]+angle[i])
            trafo = mi.Transform4f.look_at(origin=[x,1.0+y,0.1],
                                                    target=[0,1.0,-0.3],
                                                    up=[0, 0, 1]).scale(0.05),
            params[f'light{i}.vertex_positions'] = dr.ravel(trafo[0]@init[f"light{i}"])
            x1 = dr.detach(0.51*dr.sin(opt[f"rot{i}"]+angle[i]))
            y1 = dr.detach(0.51*dr.cos(opt[f"rot{i}"]+angle[i]))
            trafo = mi.Transform4f.look_at(origin=[x1,1.0+y1,0.1],
                                                    target=[0,1.0,-0.3],
                                                    up=[0, 0, 1]).scale(0.05),
            params[f'lightbar{i}.vertex_positions'] = dr.ravel(trafo[0]@init[f"lightbar{i}"])
        params.update()
    
    def output(opt):
        return ",".join([str(opt['rot'+str(i)]) for i in range(num)])
    return opt, apply_transformation, output, params