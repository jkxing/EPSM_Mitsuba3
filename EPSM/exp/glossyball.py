import mitsuba as mi
import drjit as dr
import os,sys
import torch
import numpy as np
it=200 # 500
spp=32
resolution=512
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
            "max_depth": 2,
        },
        "mysensor": {
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
        "myemitterr": {
            'type': 'obj',
            'filename': 'data/meshes/rectangle.obj',
            "to_world": T.translate([-1.0,5.0,-1.0]).rotate([1, 0, 0], 90.0).scale(0.05),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [10000.0,0.0,0.0],
                }
            }
        },
        "myemitterg": {
            'type': 'obj',
            'filename': 'data/meshes/rectangle.obj',
            "to_world": T.translate([-1.0,5.0,1.0]).rotate([1, 0, 0], 90.0).scale(0.05),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0.0,10000.0,0.0],
                }
            }
        },
        "myemitterb": {
            'type': 'obj',
            'filename': 'data/meshes/rectangle.obj',
            "to_world": T.translate([1.0,5.0,0.0]).rotate([1, 0, 0], 90.0).scale(0.05),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0.0,0.0,10000.0],
                }
            }
        },
        "Floor": {
            'type': 'obj',
            'filename': 'data/meshes/plate.obj',
            "to_world": T.scale(1.2).rotate([1, 0, 0], 180.0).scale(3),
            "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.01,
                 }
            }
        },

        "Floor2": {
            'type': 'obj',
            'filename': 'data/meshes/rectangle.obj',
            "to_world": T.translate([0.0,-0.1,0.0]).rotate([1, 0, 0], -90.0).scale(4),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.5, 0.5, 0.5],
                }
            }
        },
        'light2': {
            'type': 'envmap',
            'filename': 'data/envmap/cyclorama_hard_light_1k.exr'
        }
    }
    return mi.load_dict(base)

def load_scene2():
    base = {
        "type": "scene",
        "integrator":{
            'type': 'prb_reparam',
            "max_depth": 2,
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
        "myemitterr": {
            'type': 'obj',
            'filename': 'data/meshes/rectangle.obj',
            "to_world": T.translate([-4.0,5.0,-4.0]).rotate([1, 0, 0], 90.0).scale(0.05),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [10000.0,0.0,0.0],
                }
            }
        },
        "myemitterg": {
            'type': 'obj',
            'filename': 'data/meshes/rectangle.obj',
            "to_world": T.translate([-4.0,5.0,4.0]).rotate([1, 0, 0], 90.0).scale(0.05),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0.0,10000.0,0.0],
                }
            }
        },
        "myemitterb": {
            'type': 'obj',
            'filename': 'data/meshes/rectangle.obj',
            "to_world": T.translate([5.0,5.0,0.0]).rotate([1, 0, 0], 90.0).scale(0.05),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0.0,0.0,10000.0],
                }
            }
        },
        "Floor": {
            'type': 'obj',
            'filename': 'data/meshes/plate.obj',
            "to_world": T.scale(1.2).rotate([1, 0, 0], 180.0).scale(3),
            "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.01,
                 }
            }
        },

        "Floor2": {
            'type': 'obj',
            'filename': 'data/meshes/rectangle.obj',
            "to_world": T.translate([0.0,-0.1,0.0]).rotate([1, 0, 0], -90.0).scale(4),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.5, 0.5, 0.5],
                }
            }
        },
        'light2': {
            'type': 'envmap',
            'filename': 'data/envmap/cyclorama_hard_light_1k.exr'
        }
    }
    return mi.load_dict(base)

gt_scene = load_scene()
scene = load_scene2()
params = mi.traverse(scene)
thres = 10000
max_depth = 2

def optim_settings():
    objlist=[f"myemitter{i}" for i in "rgb"]
    opt = mi.ad.Adam(lr=0.1)
    init={}
    for obj in objlist:
        opt[obj] = mi.Vector3f([0.0,0.0,0.0])
        init[obj] = dr.unravel(mi.Point3f, params[f'{obj}.vertex_positions'])
    
    opt["alpha"]=params["Floor.bsdf.brdf_0.alpha.value"]
    def apply_transformation(params, opt):
        opt["alpha"] = dr.clamp(opt["alpha"],0.001,0.5)
        for obj in objlist:
            opt[obj][1] = 0
            trans_mat = mi.Transform4f.translate(opt[obj])
            params[f'{obj}.vertex_positions'] = dr.ravel(trans_mat@init[obj])
        #params["Floor.bsdf.alpha.value"] = opt["alpha"]
        params.update()
    def output(opt):
        return f'{opt["myemitterr"]},{opt["myemitterg"]},{opt["myemitterb"]}'
    return opt, apply_transformation, output, params
