import os
from os.path import realpath, join

import drjit as dr
import mitsuba as mi
import torch

mi.set_variant('cuda_ad_rgb')
it=1000 # 500
spp=64
resolution=512
thres = 1000
max_depth = 4
match_res = 256

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
from mitsuba.scalar_rgb import Transform4f as T

def create_flat_lens_mesh(resolution):
    # Generate UV coordinates
    U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, resolution[0]),
        dr.linspace(mi.Float, 0, 1, resolution[1]),
        indexing='ij'
    )
    texcoords = mi.Vector2f(U, V)
    
    # Generate vertex coordinates
    X = 2.0 * (U - 0.5)
    Y = 2.0 * (V - 0.5)
    vertices = mi.Vector3f(X, Y, 0.0)

    # Create two triangles per grid cell
    faces_x, faces_y, faces_z = [], [], []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1
            faces_x.extend([v00, v01])
            faces_y.extend([v10, v10])
            faces_z.extend([v01, v11])

    # Assemble face buffer 
    faces = mi.Vector3u(faces_x, faces_y, faces_z)

    # Instantiate the mesh object
    mesh = mi.Mesh("lens-mesh", resolution[0] * resolution[1], len(faces_x), has_vertex_texcoords=True)
    
    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['vertex_texcoords'] = dr.ravel(texcoords)
    mesh_params['faces'] = dr.ravel(faces)
    mesh_params.update()

    return mesh

lenres = 32
lim=0.05
lr = 0.001
import random
random.seed(2134)
lens_res = (lenres,lenres)
lens_fname = join("data/meshes", 'lens_{}_{}.ply'.format(*lens_res))
if not os.path.isfile(lens_fname):
    m = create_flat_lens_mesh(lens_res)
    m.write_ply(lens_fname)
    print('[+] Wrote lens mesh ({}x{} tesselation) file to: {}'.format(*lens_res, lens_fname))



integrator = {
    'type': 'prb_reparam',
    'max_depth': max_depth,
}
res = 512

base = {
    'type': 'scene',
    'sensor0': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(-1, 2, 2),
                        target=(-1, 2, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 50,
        'film': {
            'type': 'hdrfilm',
            'width': res,
            'height': res,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': True
        },
    },

    'sensor1': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(-1, 2, 2),
                        target=(-1, 2, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 50,
        'film': {
            'type': 'hdrfilm',
            'width': res,
            'height': res,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': False
        },
    },

    'sensor2': {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(-1, 2, 2),
                        target=(-1, 2, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 50,
        'film': {
            'type': 'hdrfilm',
            'width': 256,
            'height': 256,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': False
        },
    },
    'integrator': integrator,
    # Glass BSDF
    'white-bsdf': {
        'type': 'diffuse',
        'id': 'white-bsdf',
        'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
    },
    'black-bsdf': {
        'type': 'diffuse',
        'id': 'black-bsdf',
        'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
    },
    # Glass rectangle, to be optimized
    'lens': {
        'type': 'ply',
        'id': 'lens',
        'filename': lens_fname,
        'to_world': T.translate([-1, 2, -2]).scale(1.5).rotate([1,0,0],0),
        "bsdf": {
                'type': 'dielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0
            }
    },

    'lens1': {
        'type': 'ply',
        'id': 'lens1',
        'filename': lens_fname,
        'to_world': T.translate([-1, 2, -1.6]).scale(1.5).rotate([1,0,0],0),
        "bsdf": {
                'type': 'dielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0
            }
    },


    "DiffuseBSDF": {
        "type": "diffuse",
        "reflectance": {
                "type": "rgb",
                "value": [1.0, 1.0, 0.0],
            }
    },
    
    'wall': {
        'type': 'obj',
        'filename': 'data/meshes/rectangle.obj',
        'to_world': T.translate([3,1, -7.5]).scale(9).rotate([1,0,0],-0),
        'bsdf':{
            'type': "twosided",
            "bsdf":{
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "filename" : 'data/texture/texball.png',
                    "filter_type" : 'nearest',
                    "wrap_mode" : 'repeat'
                },
                # "type": "ref",
                # "id":  "DiffuseBSDF"
            }
        },
    },


    
    # 'wall1': {
    #     'type': 'obj',
    #     'filename': 'data/meshes/rectangle.obj',
    #     'to_world': T.translate([0,0, -7.51]).scale(5).rotate([1,0,0],-0),
    #     'bsdf':{
    #         'type': "twosided",
    #         "bsdf":{
    #             "type": "diffuse",
    #             "id":  "black-bsdf"
    #         }
    #     },
    # },
    

    "emitter": {
            "type": "point",
            "intensity": {
                "type": "rgb",
                "value": [600, 600, 600],
            },
            "to_world": T.translate([0,8,0])
    },

    # 'light2': {
    #     'type': 'envmap',
    #     'filename': '../scenes/textures/envmap.exr'
    # }
}

gt_scene = mi.load_dict(base)
params = mi.traverse(gt_scene)

U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, lenres),
        dr.linspace(mi.Float, 0, 1, lenres),
        indexing='ij'
    )
init_height0 = -(1-((V*2-1)**2+(U*2-1)**2)**0.5)*0.5
limit_height1 = -(1-((V*2-1)**2+(U*2-1)**2)**0.5)*0.5
for i in range(lenres*lenres):
    if i%lenres==0 or i//lenres==0 or i//lenres==lenres-1 or i%lenres==lenres-1:
        init_height0[i] = 0
        limit_height1[i] = 0
    else:
        init_height0[i] = (random.random()-0.5)*lim
        limit_height1[i] = lim

positions_initial = dr.unravel(mi.Vector3f, params['lens.vertex_positions'])
normals_initial   = dr.unravel(mi.Vector3f, params['lens.vertex_normals'])
normals_pertube = dr.zeros(mi.cuda_ad_rgb.Vector3f, (lenres*lenres))
for i in range(lenres*lenres):
    normals_pertube[0,i] = random.random()*lim
    normals_pertube[1,i] = random.random()*lim
    normals_pertube[2,i] = random.random()*lim
new_normals = (normals_initial + normals_pertube)
params['lens.vertex_normals'] = dr.ravel(new_normals)
params.update()
scene = mi.load_dict(base)
def optim_settings():
    params = mi.traverse(scene) 
    normals_pertube = dr.zeros(mi.cuda_ad_rgb.Vector3f, (lenres*lenres))
    normals_initial   = dr.unravel(mi.Vector3f, params['lens.vertex_normals'])

    opt = mi.ad.Adam(lr=lr)
    opt['trans'] = normals_pertube
    def apply_transformation(params, opt):
        opt['trans'] = dr.clamp(opt['trans'], -lim, lim)
        new_normals = (opt['trans'] + normals_initial)
        params['lens.vertex_normals'] = dr.ravel(new_normals)
        params.update()
    
    def output(opt):
        return f"{torch.max(abs(opt['trans'].torch()))},{torch.mean(abs(opt['trans'].torch()))}"
    
    return opt, apply_transformation, output, params
