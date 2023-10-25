import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
  mi.set_variant("cuda_ad_rgb")

def so3_exp(w : mi.Vector3f) -> mi.Matrix4f:
  x = w.x
  y = w.y
  z = w.z
  norm = dr.sqrt(x * x + y * y + z * z)
  x = x / norm
  y = y / norm
  z = z / norm

  skew_operator = mi.Matrix4f(0, -z, y, 0, z, 0, -x, 0, -y, x, 0, 0, 0, 0, 0, 0)
  return mi.Matrix4f(1) + skew_operator * dr.sin(norm) + (skew_operator @ skew_operator) * (1 - dr.cos(norm))

def se3_exp(r : mi.Vector3f, t : mi.Vector3f) -> mi.Matrix4f:
  x = r.x
  y = r.y
  z = r.z
  norm = dr.sqrt(x * x + y * y + z * z)
  x = x / norm
  y = y / norm
  z = z / norm
  skew_operator = mi.Matrix4f(0, -z, y, 0, z, 0, -x, 0, -y, x, 0, 0, 0, 0, 0, 0)
  skew_square = skew_operator @ skew_operator
  R = mi.Matrix4f(1) + skew_operator * dr.sin(norm) + skew_square * (1 - dr.cos(norm))
  G = mi.Matrix3f(1) * norm + skew_operator * (1 - dr.cos(norm)) + skew_square * (norm - dr.sin(norm))
  Gv = G @ (t / norm) 
  return mi.Transform4f.translate(Gv) @ R


if __name__ == "__main__":
  w = mi.Vector3f((1.0, 2.0, 3.0))
  print(so3_exp(w))

  t = mi.Vector3f((3.0, 2.0, 1.0))
  print(se3_exp(w, t))

  import torch
  from pytorch3d.transforms import so3_exp_map, se3_exp_map
  a = torch.Tensor((1.0, 2.0, 3.0))[None, :]
  b = torch.Tensor((3.0, 2.0, 1.0, 1.0, 2.0, 3.0))[None, :]
  print(so3_exp_map(a))
  print(se3_exp_map(b))

  # opt = mi.ad.Adam(lr=5)
  # opt["p"] = mi.Vector3f((1.0, 2.0, 3.0))
  # dr.enable_grad(opt["p"])
  # for i in range(100):
  #   # x = so3_exp(opt["p"])
  #   # print(x)
  #   # loss = dr.sum(dr.sqr(x - mi.Float([0.0])))
  #   matrix = so3_exp(opt["p"])
  #   print(matrix)
  #   loss = dr.sum(dr.sqr(dr.ravel(matrix - mi.Matrix4f(1))))
  #   dr.backward(loss)
  #   opt.step()
  #   print(opt["p"])


