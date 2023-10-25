import mitsuba as mi
import drjit as dr
import numpy as np
it=600
spp=64
resolution=512
thres = 500
max_depth = 8
match_res = 128
scene_file = "data/bathroom2/scene.xml"
scene = mi.load_file(scene_file)
params = mi.traverse(scene)

def optim_settings():
    objects = ["butterfly8", "leaf2", "leaf", "sunflower3",
               "butterfly4", "leaf4", "leaf3", "sunflower"]

    init_trans = [(1.2, 0.0), (0.0, -0.33), (0.0, -0.33), (-1.2, 0.0),
                  (1.2, 0.0), (0.0, 0.33), (0.0, 0.33), (-1.2, 0.0)]
    # init_trans = [(0.1, 0.0), (0.0, -0.1), (0.0, -0.1), (-0.1, 0.0),
    #               (0.1, 0.0), (0.0, 0.1), (0.0, 0.1), (-0.1, 0.0)]

    opt = mi.ad.Adam(lr=0.01)
    initial_positions = {}
    for obj, trans in zip(objects, init_trans):
        initial_positions[obj] = dr.unravel(mi.Point3f, params[f"{obj}.vertex_positions"])
        opt[f"trans_{obj}_x"] = mi.Float(trans[0])
        opt[f"trans_{obj}_y"] = mi.Float(trans[1])
    def apply_transformation(params, opt):
        for obj in objects:
            trafo = mi.Transform4f.translate([opt[f'trans_{obj}_x'], opt[f'trans_{obj}_y'], 0.0])
            params[f'{obj}.vertex_positions'] = dr.ravel(trafo @ initial_positions[obj])
        params.update()
    
    def output(opt):
        l2 = 0
        for obj in objects:
            x = float(opt[f"trans_{obj}_x"][0])
            y = float(opt[f"trans_{obj}_y"][0])
            l2 += x * x + y * y
        return f"sum_L2={l2}"
    return opt, apply_transformation, output, params