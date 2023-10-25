import mitsuba as mi
import drjit as dr
it=200
spp=64
resolution=512
thres = 375
max_depth = 6
match_res = 128
scene_file = "data/scenes/bunny.xml"
scene = mi.load_file(scene_file)
params = mi.traverse(scene)

print(params)

def optim_settings():
    initial_positions = dr.unravel(mi.Point3f, params['sphere.vertex_positions'])
    # initial_trans = mi.Transform4f(params["sphere.to_world"])
    opt = mi.ad.Adam(lr=0.02)
    opt['trans'] = mi.Vector3f([0, 2.2, 1.7])
    def apply_transformation(params, opt):
        trafo = mi.Transform4f.translate(opt['trans'])
        params['sphere.vertex_positions'] = dr.ravel(trafo @ initial_positions)
        # params['sphere.to_world'] = trafo @ mi.Transform4f(initial_trans)
        params.update()
    
    def output(opt):
        return f"trans=[{opt['trans']}]"
    return opt, apply_transformation, output, params