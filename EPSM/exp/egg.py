import mitsuba as mi
import drjit as dr
it=200
spp=256
resolution=512
thres = 375
max_depth = 6
match_res = 128

scene_file = "data/caustic-egg/caustic_egg.xml"
scene = mi.load_file(scene_file)
params = mi.traverse(scene)
to_world = params["PerspectiveCamera.to_world"]
trans_mat = mi.Transform4f.translate([-12.5,1,-1])@to_world@mi.Transform4f.rotate([1,0,0],25)
params["PerspectiveCamera.to_world"] = dr.ravel(trans_mat)
params["PerspectiveCamera_1.to_world"] = dr.ravel(trans_mat)
params["PerspectiveCamera_2.to_world"] = dr.ravel(trans_mat)
def optim_settings():
    initpos = dr.unravel(mi.Point3f, params[f'Diffuse_0003.vertex_positions'])
    opt = mi.ad.Adam(lr=0.03)
    opt['trans'] = mi.Vector3f([1, -1.0, 1])
    def apply_transformation(params, opt):
        opt['trans'][::2] = 0
        trafo = mi.Transform4f.translate(opt['trans'])
        params['Diffuse_0003.vertex_positions'] =  dr.ravel(trafo@initpos)
        params.update()
    
    def output(opt):
        return f"trans=[{opt['trans']}]"
    return opt, apply_transformation, output, params