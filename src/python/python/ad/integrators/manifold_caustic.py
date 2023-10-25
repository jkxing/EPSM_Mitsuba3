from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .common import RBIntegrator, mis_weight, LagRBIntegrator
import torch


class ManifoldCausticIntegrator(LagRBIntegrator):
    def calc_grad(self, path_info, dlduv, dldp, Lt):
        def find_orthogonal_vector(normal):
            v = torch.cat([torch.zeros_like(normal[:,0:1]), -normal[:,2:], normal[:,1:2]],dim=-1)
            return v / torch.norm(v,dim=-1,keepdim=True)

        def create_local_frame(normal):
            normal_normalized = normal / torch.norm(normal,dim=-1,keepdim=True)
            tangent = find_orthogonal_vector(normal_normalized)
            bitangent = torch.cross(normal_normalized, tangent)

            local_frame = torch.column_stack((tangent[:,None,:], bitangent[:,None,:], normal_normalized[:,None,:]))
            return local_frame

        def get_point(pointinfo):
            return pointinfo['points'][0]*pointinfo['uv'][0][...,None]+pointinfo['points'][1]*pointinfo['uv'][1][...,None]+pointinfo['points'][2]*(1-pointinfo['uv'][0]-pointinfo['uv'][1])[...,None]

        def get_normal(pointinfo):
            return pointinfo['normals'][0]*pointinfo['uv'][0][...,None]+pointinfo['normals'][1]*pointinfo['uv'][1][...,None]+pointinfo['normals'][2]*(1-pointinfo['uv'][0]-pointinfo['uv'][1])[...,None]

        def add(v):
            v.requires_grad_(True)
            v.retain_grad()
            param_list.append(v)
            param_grad_list.append(torch.zeros((v.shape[0],len(path_info)*2,v.shape[1])).cuda())
            final_param_grad.append(torch.zeros((v.shape[0],v.shape[1])).cuda())

        constraint = torch.zeros((path_info[0]["cam"].shape[0],len(path_info)*2,len(path_info)*2)).cuda()
        diffuse_pos = torch.zeros((path_info[0]["cam"].shape[0])).cuda()
        hasdiffuse = torch.zeros((path_info[0]["cam"].shape[0])).cuda()
        param_grad_list = []
        final_param_grad = []
        diffuse_grad = []
        light_grad = []
        param_list = []

        for id in range(1,len(path_info)):
            isdiffuse = (mi.has_flag(path_info[id]["bsdf"],mi.BSDFFlags.Diffuse).torch()>0)
            path_info[id]["uv"][0].requires_grad_(True)
            path_info[id]["uv"][1].requires_grad_(True)
            path_info[id]["uv"][0].retain_grad()
            path_info[id]["uv"][1].retain_grad()
            add(path_info[id]["points"][0])
            add(path_info[id]["points"][1])
            add(path_info[id]["points"][2])
            if id==1:
                point_prev = path_info[id-1]["cam"]
                dldp[isdiffuse==0] = 0
                dlduv[isdiffuse==0] = 0
                diffuse_grad.append(dldp)
                valid = (path_info[id]["ismesh"]>0)
            else:
                point_prev = get_point(path_info[id-1])
                valid &= (path_info[id]["ismesh"]>0)
            
            hasdiffuse+=isdiffuse
            valid&=(hasdiffuse<2)
            diffuse_pos[isdiffuse] = id 
            nolight = (path_info[id]['active_em']==0)
            point_cur = get_point(path_info[id])
            ## light sampling path
            path_info[id]['light'].requires_grad_(True)
            path_info[id]['light'].retain_grad()
            point_next = path_info[id]['light']
            param_light_grad = torch.zeros((path_info[id]['light'].shape[0],len(path_info)*2,path_info[id]['light'].shape[1])).cuda()
            wi = point_prev-point_cur
            wo = point_next-point_cur
            wi = wi/torch.norm(wi,dim=-1,keepdim=True)
            wo = wo/torch.norm(wo,dim=-1,keepdim=True)
            n = get_normal(path_info[id])
            transmat = create_local_frame(n).detach()
            wi2 = torch.bmm(transmat,wi[...,None])[...,0]
            wo2 = torch.bmm(transmat,wo[...,None])[...,0]
            res = (wi2+wo2*path_info[id]["eta"][...,None])
            res = res/torch.norm(res,dim=-1,keepdim=True)
            #res[isdiffuse] = wo2[isdiffuse]-wo2[isdiffuse].detach()
            res2 = wo2-wo2.detach()
            for i in range(2):
                loss = torch.sum(res[:,i])
                loss.backward(retain_graph=True)
                if id>1:
                    constraint[:,2*id+i-2,2*id-2] = path_info[id-1]['uv'][0].grad
                    constraint[:,2*id+i-2,2*id-1] = path_info[id-1]['uv'][1].grad
                    path_info[id-1]['uv'][0].grad = None
                    path_info[id-1]['uv'][1].grad = None
                constraint[:,2*id+i-2,2*id+0] = path_info[id+0]['uv'][0].grad
                constraint[:,2*id+i-2,2*id+1] = path_info[id-0]['uv'][1].grad
                path_info[id+0]['uv'][0].grad = None
                path_info[id-0]['uv'][1].grad = None

                for (idx,para) in enumerate(param_list):
                    if para.grad!=None:
                        param_grad_list[idx][:,2*id-2+i,:] = para.grad
                        para.grad = None
                    else:
                        param_grad_list[idx][:,2*id-2+i,:] = 0
                param_light_grad[:,2*id-2+i,:] = point_next.grad
                point_next.grad = None

                loss = torch.sum(res2[:,i])
                loss.backward(retain_graph=True)
                for j in range(1,id+1):
                    constraint[diffuse_pos==j,2*j-2+i] = 0
                    constraint[diffuse_pos==j,2*j-2+i,2*id+0] = path_info[id+0]['uv'][0].grad[diffuse_pos==j]
                    constraint[diffuse_pos==j,2*j-2+i,2*id+1] = path_info[id+0]['uv'][1].grad[diffuse_pos==j]
                    for (idx,para) in enumerate(param_list):
                        if para.grad!=None:
                            param_grad_list[idx][diffuse_pos==j,2*j-2+i,:] = para.grad[diffuse_pos==j]
                        else:
                            param_grad_list[idx][diffuse_pos==j,2*j-2+i,:] = 0
                    if point_next.grad != None:
                        param_light_grad[diffuse_pos==j,2*j-2+i,:] = point_next.grad[diffuse_pos==j]
                        point_next.grad = None
                    else:
                        param_light_grad[diffuse_pos==j,2*j-2+i,:] = 0
                path_info[id+0]['uv'][0].grad = None
                path_info[id-0]['uv'][1].grad = None
                for (idx,para) in enumerate(param_list):
                    para.grad = None
                
            cur = constraint[:,:2*id,2:2*id+2].clone()
            cur[valid==0] = torch.eye(2*id).cuda()
            cur[path_info[id]['active']==0] = torch.eye(2*id).cuda()
            cur[nolight] = torch.eye(2*id).cuda()
            grad_uv_inv = torch.linalg.inv(cur)
            for (idx,param_grad) in enumerate(param_grad_list):
                duvdp = -torch.bmm(grad_uv_inv,param_grad[:,:2*id,:]) #(N,C,3)
                dldp = torch.bmm(dlduv[...,:2*id],duvdp)[:,0,:]
                dldp[valid==0] = 0
                dldp[path_info[id]['active']==0] = 0
                dldp[nolight] = 0
                dldp[hasdiffuse>0] = 0
                dldp = torch.nan_to_num(dldp)
                final_param_grad[idx]+=dldp

            duvdlp = -torch.bmm(grad_uv_inv,param_light_grad[:,:2*id,:]) #(N,C,3)
            dldlp = torch.bmm(dlduv[...,:2*id],duvdlp)[:,0,:] #(N,3)
            dldlp[valid==0] = 0
            dldlp[hasdiffuse>0] = 0
            dldlp[nolight] = 0
            dldlp[path_info[id]['active']==0] = 0
            dldlp = torch.nan_to_num(dldlp)
            light_grad.append(dldlp)

            if id<len(path_info)-1:
                path_info[id+1]['uv'][0].requires_grad_(True)
                path_info[id+1]['uv'][1].requires_grad_(True)
                path_info[id+1]['uv'][0].retain_grad()
                path_info[id+1]['uv'][1].retain_grad()
                point_next = get_point(path_info[id+1])
                n = get_normal(path_info[id])
                m = path_info[id]["hf"]
                add(n)
                add(m)
                point_next.retain_grad()
                wi = point_prev-point_cur
                wo = point_next-point_cur
                wi = wi/torch.norm(wi,dim=-1,keepdim=True)
                wo = wo/torch.norm(wo,dim=-1,keepdim=True)
                transmat = create_local_frame(n)
                wi2 = torch.bmm(transmat,wi[...,None])[...,0]
                wo2 = torch.bmm(transmat,wo[...,None])[...,0]
                res = (wi2+wo2*path_info[id]["eta"][...,None])
                res = res/torch.norm(res,dim=-1,keepdim=True)-m
                res2 = wo2-wo2.detach()
                param_diffuse_grad = torch.zeros((path_info[id+1]['points'][3].shape[0],len(path_info)*2,path_info[id+1]['points'][3].shape[1])).cuda()
                for i in range(2):
                    loss = torch.sum(res[:,i])
                    loss.backward(retain_graph=True)
                    if id>1: 
                        constraint[:,2*id+i-2,2*id-2] = path_info[id-1]['uv'][0].grad
                        constraint[:,2*id+i-2,2*id-1] = path_info[id-1]['uv'][1].grad
                        path_info[id-1]['uv'][0].grad = None
                        path_info[id-1]['uv'][1].grad = None
                    constraint[:,2*id+i-2,2*id+0] = path_info[id+0]['uv'][0].grad
                    constraint[:,2*id+i-2,2*id+1] = path_info[id-0]['uv'][1].grad
                    constraint[:,2*id+i-2,2*id+2] = path_info[id+1]['uv'][0].grad
                    constraint[:,2*id+i-2,2*id+3] = path_info[id+1]['uv'][1].grad
                    path_info[id+0]['uv'][0].grad = None
                    path_info[id-0]['uv'][1].grad = None
                    path_info[id+1]['uv'][0].grad = None
                    path_info[id+1]['uv'][1].grad = None
                    for (idx,para) in enumerate(param_list):
                        if para.grad!=None:
                            param_grad_list[idx][:,2*id-2+i,:] = para.grad
                            para.grad = None
                    
                    param_diffuse_grad[:,2*id-2+i,:] = point_next.grad
                    point_next.grad = None
                    loss = torch.sum(res2[:,i])
                    loss.backward(retain_graph=True)
                    for j in range(1,id+1):
                        constraint[diffuse_pos==j,2*j-2+i] = 0
                        constraint[diffuse_pos==j,2*j-2+i,2*id+0] = path_info[id+0]['uv'][0].grad[diffuse_pos==j]
                        constraint[diffuse_pos==j,2*j-2+i,2*id+1] = path_info[id+0]['uv'][1].grad[diffuse_pos==j]
                        constraint[diffuse_pos==j,2*j-2+i,2*id+2] = path_info[id+1]['uv'][0].grad[diffuse_pos==j]
                        constraint[diffuse_pos==j,2*j-2+i,2*id+3] = path_info[id+1]['uv'][1].grad[diffuse_pos==j]
                        for (idx,para) in enumerate(param_list):
                            if para.grad!=None:
                                param_grad_list[idx][diffuse_pos==j,2*j-2+i,:] = para.grad[diffuse_pos==j]
                            else:
                                param_grad_list[idx][diffuse_pos==j,2*j-2+i,:] = 0
                        if point_next.grad != None:
                            param_diffuse_grad[diffuse_pos==j,2*j-2+i,:] = point_next.grad[diffuse_pos==j]
                        else:
                            param_diffuse_grad[diffuse_pos==j,2*j-2+i,:] = 0
                    path_info[id+0]['uv'][0].grad = None
                    path_info[id-0]['uv'][1].grad = None
                    path_info[id+1]['uv'][0].grad = None
                    path_info[id+1]['uv'][1].grad = None
                    for (idx,para) in enumerate(param_list):
                        para.grad = None
                
                cur = constraint[:,:2*id,2:2*id+2].clone()
                cur[valid==0] = torch.eye(2*id).cuda()
                cur[path_info[id+1]['active']==0] = torch.eye(2*id).cuda()
                grad_uv_inv = torch.linalg.inv(cur)
                for (idx,param_grad) in enumerate(param_grad_list):
                    duvdp = -torch.bmm(grad_uv_inv,param_grad[:,:2*id,:]) #(N,C,3)
                    dldp = torch.bmm(dlduv[...,:2*id],duvdp)[:,0,:]
                    dldp[valid==0]=0
                    dldp[path_info[id+1]['active']==0]=0
                    dldp[mi.has_flag(path_info[id+1]['bsdf'],mi.BSDFFlags.Diffuse).torch()==0] = 0
                    dldp = torch.nan_to_num(dldp)
                    final_param_grad[idx]+=dldp

                duvddp = -torch.bmm(grad_uv_inv,param_diffuse_grad[:,:2*id,:])
                dlddp = torch.bmm(dlduv[...,:2*id],duvddp)[:,0,:] #(N,3)
                dlddp[valid==0]=0
                dlddp[path_info[id+1]['active']==0]=0
                dlddp[(mi.has_flag(path_info[id+1]['bsdf'],mi.BSDFFlags.Null).torch()==0)&(mi.has_flag(path_info[id+1]['bsdf'],mi.BSDFFlags.Diffuse).torch()==0)] = 0
                dlddp = torch.nan_to_num(dlddp)
                diffuse_grad.append(dlddp)

        #remove outlier
        for idx in range(len(final_param_grad)):
            final_param_grad[idx][final_param_grad[idx]>0.1] = 0
            final_param_grad[idx][final_param_grad[idx]<-0.1] = 0
            final_param_grad[idx] = mi.Point3f(final_param_grad[idx])
        for idx in range(len(light_grad)):
            light_grad[idx][light_grad[idx]>0.1] = 0
            light_grad[idx][light_grad[idx]<-0.1] = 0
            light_grad[idx] = mi.Point3f(light_grad[idx])
        for idx in range(len(diffuse_grad)):
            diffuse_grad[idx][diffuse_grad[idx]>0.1] = 0
            diffuse_grad[idx][diffuse_grad[idx]<-0.1] = 0
            diffuse_grad[idx] = mi.Point3f(diffuse_grad[idx])
    
        return final_param_grad,light_grad,diffuse_grad
    def sample2(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               log_path=False,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        path_grad = kwargs.get("final_grad",mi.Point3f(0.0))
        light_grad = kwargs.get("light_grad",mi.Point3f(0.0))
        diffuse_grad = kwargs.get("diffuse_grad",mi.Point3f(0.0))
        Lt = kwargs.get("Lt")
        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        iteration = 0
        logs=[{"cam":ray.o.torch()}]
        debug=False
        max_depth = min(self.max_depth, 6)
        while iteration<max_depth and dr.any(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=dr.eq(depth, 0))
                pi = scene.ray_intersect_preliminary(ray,coherent=True,active=active)
                si_follow = pi.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape)
                if not primal and iteration*5+4<len(path_grad):
                    dr.backward(si.p0*path_grad[iteration*5]+si.p1*path_grad[iteration*5+1]+si.p2*path_grad[iteration*5+2])
                if not primal and iteration<len(diffuse_grad):
                    dr.backward(si_follow.p*diffuse_grad[iteration])
            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Given the detached emitter sample, *recompute* its
                    # contribution with AD to enable light source optimization
                    ds.d = dr.normalize(ds.p - si.p)
                    em_val = scene.eval_emitter_direction(si, ds, active_em)
                    em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)
                    dr.disable_grad(ds.d)

                # Evaluate BSDF * cos(theta) differentiably
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mis_em * bsdf_value_em * em_weight

                if not primal:
                    ray_direct = si.spawn_ray(ds.d)
                    si_direct = scene.ray_intersect(ray_direct,
                                        ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                        coherent=dr.eq(depth, 0),active=active_em)
                    if iteration<len(light_grad):
                        dr.backward(si_direct.p*mi.Vector3f(light_grad[iteration].torch()*torch.sum(Lr_dir.torch(),dim=-1,keepdim=True)))
                    

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)
            # ------------------ Attached BSDF sampling -------------------
            with dr.resume_grad(when=not primal):
                bsdf = si.bsdf(ray)
                bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                    sampler.next_1d(),
                                                    sampler.next_2d(),
                                                    active_next)
                if not primal and iteration*5+4<len(path_grad):
                    dr.backward(bsdf_sample.hf*path_grad[iteration*5+4]+si_follow.sh_frame.n*path_grad[iteration*5+3])
            #dr.disable_grad(bsdf_sample)        
            #dr.disable_grad(bsdf_weight)   
            if log_path and iteration<5:
                logs.append({"it":iteration,"active":(active&si.is_valid()).torch(),"bsdf":bsdf.flags(),"valid":si.valid.torch(),
                            "light":ds.p.torch(),"active_em":active_em.torch(),
                            "points":[si.p0.torch(),si.p1.torch(),si.p2.torch(),si.p.torch()],"uv":[si.b0.torch(),si.b1.torch()],"normal":si.sh_frame.n.torch(),\
                            "normals":[si.n0.torch(),si.n1.torch(),si.n2.torch()],\
                            "eta":bsdf_sample.eta.torch(),"hf":bsdf_sample.hf.torch()
                            })
            
            # ---- Update loop variables based on current interaction -----

            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ------------------ Differential phase only ------------------

            if not primal:
                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(dr.neq(bsdf_val_det, 0),
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_det * bsdf_val)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + Lr_ind

                    if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            depth[si.is_valid()] += 1
            active = active_next
            iteration+=1
        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L,                    # State for the differential phase，
            logs
        )
    

mi.register_integrator("manifold_caustic", lambda props: ManifoldCausticIntegrator(props))
