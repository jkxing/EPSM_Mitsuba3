import mitsuba as mi
import drjit as dr
import os,sys
import torch
import numpy as np
it=600 # 500
spp=64
resolution=512
thres = 250
max_depth = 2
match_res = 128

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
from mitsuba.scalar_rgb import Transform4f as T

def load_scene():
    base = {
        "type": "scene",
        "integrator":{
            'type': 'prb_reparam',
            "max_depth": max_depth,
        },
        "sensor0": {
            "type": "perspective",
            "fov": 90,
            "near_clip": 1.0,
            "far_clip": 1000.0,
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.1,3.5,0.1],
                                                    target=[0.1, 0, 0.1],
                                                    up=[0, 0, 1]),
            "myfilm": {
                "type": "hdrfilm",
                'rfilter': { 'type': 'gaussian' },
                'sample_border': True,
                "width": resolution,
                "height": resolution,
            }, "mysampler": {
                "type": "independent",
                "sample_count": 4,
            },
        },
        "DiffuseBSDF": {
            "type": "diffuse",
            "reflectance": {
                    "type": "rgb",
                    "value": [0.5, 0.5, 0.5],
                }
        },
        "DiffuseBSDF2": {
            "type": "diffuse",
            "reflectance": {
                    "type": "rgb",
                    "value": [0.8, 0.5, 0.5],
                }
        },
        "myemitter": {
            "type": "rectangle",
            "to_world": T.translate([0.0,8.0,0.0]).rotate([1, 0, 0], 90.0).scale(0.01),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 300000.0,
                }
            }
        },
        "Floor": {
            "type": "rectangle",
            "to_world": T.rotate([1, 0, 0], -90.0).scale(10),
            "mybsdf": {
                "type": "ref",
                "id": "DiffuseBSDF",
            }
        },
        "text":{
                "type":"obj",
                "filename": "data/meshes/ShadowText.obj",
                "to_world": T.translate([0.05,6.0,0.05]).scale(0.7).rotate([0,1,0],180),
                "mybsdf": {
                    "type": "ref",
                    "id": "DiffuseBSDF2",
                }
            }
    }
    return mi.load_dict(base)

def load_scene2():
    base = {
        "type": "scene",
        "integrator":{
            'type': 'prb_reparam',
            "max_depth": max_depth,
        },
        "sensor0": {
            "type": "perspective",
            "fov": 90,
            "near_clip": 1.0,
            "far_clip": 1000.0,
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.1,3.5,0.1],
                                                    target=[0.1, 0, 0.1],
                                                    up=[0, 0, 1]),
            "myfilm": {
                "type": "hdrfilm",
                'rfilter': { 'type': 'gaussian' },
                'sample_border': True,
                "width": resolution,
                "height": resolution,
            }, "mysampler": {
                "type": "independent",
                "sample_count": 4,
            },
        },
        "sensor1": {
            "type": "perspective",
            "fov": 90,
            "near_clip": 1.0,
            "far_clip": 1000.0,
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.1,3.5,0.1],
                                                    target=[0.1, 0, 0.1],
                                                    up=[0, 0, 1]),
            "myfilm": {
                "type": "hdrfilm",
                'rfilter': { 'type': 'gaussian' },
                'sample_border': False,
                "width": resolution,
                "height": resolution,
            }, "mysampler": {
                "type": "independent",
                "sample_count": 4,
            },
        },
        "sensor2": {
            "type": "perspective",
            "fov": 90,
            "near_clip": 1.0,
            "far_clip": 1000.0,
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.1,3.5,0.1],
                                                    target=[0.1, 0, 0.1],
                                                    up=[0, 0, 1]),
            "myfilm": {
                "type": "hdrfilm",
                'rfilter': { 'type': 'gaussian' },
                'sample_border': False,
                "width": 128,
                "height": 128,
            }, "mysampler": {
                "type": "independent",
                "sample_count": 8,
            },
        },
        "DiffuseBSDF": {
            "type": "diffuse",
            "reflectance": {
                    "type": "rgb",
                    "value": [0.5, 0.5, 0.5],
                }
        },
        "DiffuseBSDF2": {
            "type": "diffuse",
            "reflectance": {
                    "type": "rgb",
                    "value": [0.8, 0.5, 0.5],
                }
        },
        "myemitter": {
            "type": "rectangle",
            "to_world": T.translate([0.0,8.0,0.0]).rotate([1, 0, 0], 90.0).scale(0.01),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 300000.0,
                }
            }
        },
        "Floor": {
            "type": "rectangle",
            "to_world": T.rotate([1, 0, 0], -90.0).scale(10),
            "mybsdf": {
                "type": "ref",
                "id": "DiffuseBSDF",
            }
        },
    }
    import random,math
    for i in range(400):
        x = math.sin(i*1.0/400*math.pi*2)*0.8+0.03
        z = math.cos(i*1.0/400*math.pi*2)*0.8+0.03
        base[f"sphere{i}"] = {
            "type":"obj",
            "filename": "data/meshes/sphere.obj",
            "to_world": T.translate([x,6.0,z]).scale(0.025),
            "mybsdf": {
                "type": "ref",
                "id": "DiffuseBSDF2",
            }
        }
    return mi.load_dict(base)

gt_scene = load_scene()
scene = load_scene2()
params = mi.traverse(scene)
def optim_settings():
    objlist=[f"sphere{i}" for i in range(400)]
    opt = mi.ad.Adam(lr=0.005)
    init={}
    for obj in objlist:
        opt[obj] = mi.Vector3f([0.0,0.0,0.0])
        init[obj] = dr.unravel(mi.Point3f, params[f'{obj}.vertex_positions'])
    def apply_transformation(params, opt):

        for obj in objlist:
            opt[obj][1] = 0
            trans_mat = mi.Transform4f.translate(opt[obj])
            params[f'{obj}.vertex_positions'] = dr.ravel(trans_mat@init[obj])
        params.update()
    def output(opt):
        return opt["sphere0"]
    return opt, apply_transformation, output, params

