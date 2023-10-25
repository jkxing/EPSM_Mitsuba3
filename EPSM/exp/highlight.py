import os
from os.path import realpath, join

import drjit as dr
import mitsuba as mi
import torch

mi.set_variant('cuda_ad_rgb')
it=500 # 500
spp=64
resolution=512
thres = 375
max_depth = 2
match_res = 128

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
from mitsuba.scalar_rgb import Transform4f as T

gt_scene = {
    'type': 'scene',
    'integrator': {'type': 'prb_reparam','max_depth':max_depth},
    'sensor0': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(0, 0, 2),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': resolution,
            'height': resolution,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': True
        },
    },

    'sensor1': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(0, 0, 2),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': resolution,
            'height': resolution,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': False
        },
    },

    'sensor2': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(0, 0, 2),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': 128,
            'height': 128,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': False
        },
    },
    
    'wall0': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        'to_world': T.translate([1.8,0, -8]).scale(4.0).rotate([1,0,0],-30).rotate([0,0,1],90).rotate([1,0,0],90),
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

    'wall1': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        'to_world': T.translate([0,0, -8]).scale(4.0).rotate([1,0,0],-58).rotate([0,0,1],90).rotate([1,0,0],90),
        "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.02,
                 }
            }
    },

    'wall2': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        'to_world': T.translate([-1.8,0, -8]).scale(4.0).rotate([1,0,0],-30).rotate([0,0,1],90).rotate([1,0,0],90),
        "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.03,
                 }
            }
    },

    'wall3': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        'to_world': T.translate([-3.5,0, -8]).scale(4.0).rotate([1,0,0],-52).rotate([0,0,1],90).rotate([1,0,0],90),
        "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.04,
                 }
            }
    },

    'wall4': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        'to_world': T.translate([3.5,0, -8]).scale(4.0).rotate([1,0,0],-56).rotate([0,0,1],90).rotate([1,0,0],90),
        "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.005,
                 }
            }
    },

    "emitter0": {
            "type": "obj",
            "filename": "data/meshes/sphere.obj",
            "to_world": T.translate([0,8,-7]).scale(0.5),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0, 200, 0],
                }
            }
        },



    "emitter1": {
        "type": "obj",
            "filename": "data/meshes/sphere.obj",
            "to_world": T.translate([-3,8,-7]).scale(0.5),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [200, 200, 0],
                }
            }
    },

    "emitter2": {
        "type": "obj",
            "filename": "data/meshes/sphere.obj",
            "to_world": T.translate([2.8,8,-7]).scale(0.5),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0, 200, 200],
                }
            }
    },

    "emitter3": {
        "type": "obj",
            "filename": "data/meshes/sphere.obj",
            "to_world": T.translate([-7,8,-7]).scale(0.5),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [200, 0, 0],
                }
            }
    },

    "emitter4": {
        "type": "obj",
            "filename": "data/meshes/sphere.obj",
            "to_world": T.translate([7.9,8,-7]).scale(0.5),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0, 0, 200],
                }
            }
    },

    'light2': {
        'type': 'envmap',
        'filename': 'data/envmap/drachenfels_cellar_1k.exr'
    }
   
}
gt_scene = mi.load_dict(gt_scene)
scene = {
    'type': 'scene',

    'integrator': {'type': 'prb_reparam','max_depth':max_depth},
    'sensor0': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(0, 0, 2),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': resolution,
            'height': resolution,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': True
        },
    },

    'sensor1': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(0, 0, 2),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': resolution,
            'height': resolution,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': False
        },
    },

    'sensor2': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(0, 0, 2),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': 128,
            'height': 128,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': False
        },
    },
    
    'wall0': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
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

    'wall1': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.02,
                 }
            }
    },

    'wall2': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.03,
                 }
            }
    },
    'wall3': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.04,
                 }
            }
    },

    'wall4': {
        'type': 'obj',
        'filename': 'data/meshes/rec.obj',
        "bsdf": {
                 'type': "twosided",
                 "bsdf":{
                    'type': 'roughconductor',
                    'material': 'Al',
                    'distribution': 'ggx',
                    'alpha': 0.005,
                 }
            }
    },

    "emitter0": {
            "type": "obj",
            "filename": "data/meshes/sphere.obj",
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0, 200, 0],
                }
            }
        },



    "emitter1": {
        "type": "obj",
            "filename": "data/meshes/sphere.obj",
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [200, 200, 0],
                }
            }
    },

    "emitter2": {
        "type": "obj",
            "filename": "data/meshes/sphere.obj",
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0, 200, 200],
                }
            }
    },

    "emitter3": {
        "type": "obj",
            "filename": "data/meshes/sphere.obj",
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [200, 0, 0],
                }
            }
    },

    "emitter4": {
        "type": "obj",
            "filename": "data/meshes/sphere.obj",
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [0, 0, 200],
                }
            }
    },


    'light2': {
        'type': 'envmap',
        'filename': 'data/envmap/drachenfels_cellar_1k.exr'
    }

}
scene = mi.load_dict(scene)
params = mi.traverse(scene)
print(params)
def optim_settings():
    opt = mi.ad.Adam(lr=0.1)
    init = {}
    num = 5
    transl = [0,-3,3.4,-7,6.8]
    transw = [1.8,0,-1.8,-3.5,3.5]
    for i in range(num):
        opt[f"rot{i}"] = mi.Float(45)
        opt[f"li{i}"] = mi.Float(transl[i])
        init[f"rot{i}"] = dr.unravel(mi.Point3f,params[f'wall{i}.vertex_positions'])
        init[f"li{i}"] = dr.unravel(mi.Point3f,params[f'emitter{i}.vertex_positions'])
    def apply_transformation(params, opt):
        for i in range(num):
            x = opt[f"rot{i}"]
            trafo = mi.Transform4f.translate([transw[i],0, -8]).scale(4.0).rotate([1,0,0],-x).rotate([0,0,1],90).rotate([1,0,0],90)
            params[f'wall{i}.vertex_positions'] = dr.ravel(trafo@init[f"rot{i}"])
            trafo = mi.Transform4f.translate([opt[f"li{i}"],8,-7]).scale(0.5)
            params[f'emitter{i}.vertex_positions'] = dr.ravel(trafo@init[f"li{i}"])
        params.update()
    
    def output(opt):
        return f"{opt['li0']},{opt['li1']},{opt['li2']}"
    return opt, apply_transformation, output, params
