import mitsuba as mi
import drjit as dr
import os,sys
import torch
import numpy as np
it=1000 # 500
spp=64
resolution=512
thres = 1200
max_depth = 3
match_res = 256

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
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.1,25,18],
                                                    target=[0.1, 0, 0],
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
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.1,25,18],
                                                    target=[0.1, 0, 0],
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
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.1,25,18],
                                                    target=[0.1, 0, 0],
                                                    up=[0, 0, 1]),
            "myfilm": {
                "type": "hdrfilm",
                'rfilter': { 'type': 'gaussian' },
                'sample_border': False,
                "width": 256,
                "height": 256,
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

        "Floor2": {
            "type": "obj",
            "filename": "data/meshes/rectangle.obj",
            "to_world": T.translate([0.0,8.0,-5.0]).scale(30),
            "mybsdf": {
                'type': "twosided",
                "bsdf":{
                    "type": "ref",
                    "id":  "DiffuseBSDF"
                }
            }
        },

        "Floor3": {
            "type": "obj",
            "filename": "data/meshes/rectangle.obj",
            "to_world": T.translate([-15,-20, 0]).scale(50).rotate([0,1,0],90),
            "bsdf":{
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "filename" : 'data/texture/wood2.jpg',
                    "filter_type" : 'nearest',
                    "wrap_mode" : 'repeat'
                },
                # "type": "ref",
                # "id":  "DiffuseBSDF"
            }
        },



        "Floor4": {
            "type": "obj",
            "filename": "data/meshes/rectangle.obj",
            "to_world": T.translate([15,-20, 0]).scale(50).rotate([0,1,0],90),
            "mybsdf": {
                'type': "twosided",
                "bsdf":{
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "filename" : 'data/texture/wood2.jpg',
                        "filter_type" : 'nearest',
                        "wrap_mode" : 'repeat'
                    },
                    # "type": "ref",
                    # "id":  "DiffuseBSDF"
                }
            }
        },


        "human":{
                "type":"obj",
                "filename": "data/human/0.obj",
                "mybsdf": {
                    "type": "ref",
                    "id": "DiffuseBSDF2",
                }
            },
        "stair":{
                "type":"obj",
                "to_world": T.translate([0, -9.0, -9]).rotate([1, 0, 0], 90).scale(3),
                "filename": "data/meshes/stair.obj",
                "bsdf":{
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "filename" : 'data/texture/WoodPanel.jpg',
                        "filter_type" : 'nearest',
                        "wrap_mode" : 'repeat'
                    },
                    # "type": "ref",
                    # "id":  "DiffuseBSDF"
                }
            },
        "myemitter": {
            "type": "obj",
            "filename": "data/meshes/sphere.obj",
            "to_world": T.translate([0.0,50.0,30.0]).rotate([1, 0, 0], 90.0).scale(0.1),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 300000.0,
                }
            }
        }
    }
    return mi.load_dict(base)

# gt_scene = load_scene("shadow_art/scene/00001")
# img = mi.render(gt_scene,spp=1024)
# np.save("shadow_art/gt_hdr.npy",np.array(img))
# mi.util.convert_to_bitmap(img[...,:3]).write("shadow_art/gt.png")
# exit()

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from PIL import Image
from scipy.io import loadmat

class SMPL():
    def  __init__(self):
        DATA_DIR = "data/human"
        data_filename = os.path.join(DATA_DIR, "UV_Processed.mat")
        tex_filename = os.path.join(DATA_DIR, "tex.png")
        ALP_UV = loadmat(data_filename)

        with Image.open(tex_filename) as image:
            np_image = np.asarray(image.convert("RGB")).astype(np.float32)
        tex = torch.from_numpy(np_image / 255.)[None].to(device).contiguous()

        verts_temp = torch.from_numpy((ALP_UV["All_vertices"]).astype(
            int)).squeeze().to(device)  # (7829,)
        U = torch.Tensor(ALP_UV['All_U_norm']).to(device)  # (7829, 1)
        V = torch.Tensor(ALP_UV['All_V_norm']).to(device)  # (7829, 1)
        faces = torch.from_numpy(
            (ALP_UV['All_Faces'] - 1).astype(int)).to(device)  # (13774, 3)
        face_indices = torch.Tensor(
            ALP_UV['All_FaceIndices']).squeeze()  # (13774,)
        
        offset_per_part = {}
        already_offset = set()
        cols, rows = 4, 6
        for i, u in enumerate(np.linspace(0, 1, cols, endpoint = False)):
            for j, v in enumerate(np.linspace(0, 1, rows, endpoint = False)):
                part = rows * i + j + 1  # parts are 1-indexed in face_indices
                offset_per_part[part] = (u, v)

        U_norm = U.clone()
        V_norm = V.clone()
        
        # iterate over faces and offset the corresponding vertex u and v values
        for i in range(len(faces)):
            face_vert_idxs = faces[i]
            part = face_indices[i]
            offset_u, offset_v = offset_per_part[int(part.item())]
            for vert_idx in face_vert_idxs:
                if vert_idx.item() not in already_offset:
                    U_norm[vert_idx] = U[vert_idx] / cols + offset_u
                    V_norm[vert_idx] = (1 - V[vert_idx]) / rows + offset_v
                    already_offset.add(vert_idx.item())

        smpl_layer = SMPL_Layer(
                center_idx = 0,
                gender = 'male',
                model_root = DATA_DIR)
        self.smpl_layer = smpl_layer.to(device)
        self.faces = faces
        self.verts_temp=verts_temp

    def gen_mesh(self,pose_params,shape_params):
        verts, _ = self.smpl_layer(pose_params, th_betas = shape_params)
        verts = verts[:, self.verts_temp.long()-1]  # (1, 7829, 3)
        return verts

gt_scene = load_scene()
params = mi.traverse(gt_scene)

torch.manual_seed(0)
pose_params = (torch.rand(1, 72) - 0.5) * 0.2
pose_params[0,16*3+2] = 0.8
pose_params[0,17*3+2] = -0.8
shape_params = torch.rand(1, 10) * 0.0
position = (torch.rand(3) - 0.5) * 0.8
model = SMPL()
pose_params = pose_params.to(device)
shape_params = shape_params.to(device)
position = position.to(device)
gt_verts = model.gen_mesh(pose_params,shape_params)[0]
verts_mi = mi.Point3f(gt_verts)
trafo = mi.Transform4f.translate([0.05,30.0,15]).scale(10).rotate([1,0,0],90)
params["human.vertex_positions"] = dr.ravel(trafo@verts_mi)
params.update()
scene = load_scene()

