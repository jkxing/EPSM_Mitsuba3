import mitsuba as mi
import drjit as dr
import numpy as np
it=200
spp=256
resolution=512
thres = 1000
max_depth = 8
match_res = 128
scene_file = "data/bedroom/scene_v3.xml"
scene = mi.load_file(scene_file)
params = mi.traverse(scene)
to_world = params["PerspectiveCamera.to_world"]
trans_mat = mi.Transform4f.translate([0,0.0,0.0])@to_world@mi.Transform4f.rotate([0,1,0],5)
params["PerspectiveCamera.to_world"] = dr.ravel(trans_mat)
params.update()
print(params)
def optim_settings():
    opt = mi.ad.Adam(lr=0.02)
    init_toworld = mi.Transform4f(params["PerspectiveCamera.to_world"]) 
    opt['trans']=mi.Float(2)
    opt['trans2']=mi.Vector3f([-0.6,0,0.0])
    def apply_transformation(params, opt):
        opt['trans'] = dr.clamp(opt['trans'],-50,50)
        opt['trans2'] = dr.clamp(opt['trans2'],-3,3)
        #opt['trans2'][1:] = 0
        trafo = mi.Transform4f.rotate([0,1,0], opt[f'trans']*10)
        for i in range(3):
            params[f'PerspectiveCamera.to_world'] = dr.ravel(init_toworld@trafo) 
            params[f'PerspectiveCamera_1.to_world'] = dr.ravel(init_toworld@trafo) 
            params[f'PerspectiveCamera_2.to_world'] = dr.ravel(init_toworld@trafo) 
        params.update()
    
    def output(opt):
        return f"trans = {opt['trans']},{opt['trans2']}"
    return opt, apply_transformation, output, params
