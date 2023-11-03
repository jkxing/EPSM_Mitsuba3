from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

import gc
import numpy as np
import math
from .common import ADIntegrator, mis_weight
import torch

class EPSMIntegrator(ADIntegrator):
    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, state = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                reparam=None,
                active=mi.Bool(True)
            )

            # Prepare an ImageBlock as specified by the film
            block = sensor.film().create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            alpha = dr.select(valid, mi.Float(1), mi.Float(0))
            if mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special):
                aovs = sensor.film().prepare_sample(L * weight, ray.wavelengths,
                                                    block.channel_count(), alpha=alpha)
                block.put(pos, aovs)
                del aovs
            else:
                block.put(pos, ray.wavelengths, L * weight, alpha)

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid, alpha
            gc.collect()

            # Perform the weight division and return an image tensor
            sensor.film().put_block(block)
            self.primal_image = sensor.film().develop()
            import numpy as np
            image = np.array(self.primal_image)
            position = np.zeros((image.shape[0],image.shape[1],2))
            image = np.concatenate([image,position],axis=-1)
            self.primal_image = mi.TensorXf(image)
            return self.primal_image

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
        """
        Evaluates the reverse-mode derivative of the rendering step.

        Reverse-mode differentiation transforms image-space gradients into scene
        parameter gradients, enabling simultaneous optimization of scenes with
        millions of free parameters. The function is invoked with an input
        *gradient image* (``grad_in``) and transforms and accumulates these into
        the gradient arrays of scene parameters that previously had gradient
        tracking enabled.

        Before calling this function, you must first enable gradient tracking for
        one or more scene parameters, or the function will not do anything. This is
        typically done by invoking ``dr.enable_grad()`` on elements of the
        ``SceneParameters`` data structure that can be obtained obtained via a call
        to ``mi.traverse()``. Use ``dr.grad()`` to query the
        resulting gradients of these parameters once ``render_backward()`` returns.

        Parameter ``scene`` (``mi.Scene``):
            The scene to be rendered differentially.

        Parameter ``params``:
           An arbitrary container of scene parameters that should receive
           gradients. Typically this will be an instance of type
           ``mi.SceneParameters`` obtained via ``mi.traverse()``. However, it
           could also be a Python list/dict/object tree (DrJit will traverse it
           to find all parameters). Gradient tracking must be explicitly enabled
           for each of these parameters using ``dr.enable_grad(params['parameter_name'])``
           (i.e. ``render_backward()`` will not do this for you).

        Parameter ``grad_in`` (``mi.TensorXf``):
            Gradient image that should be back-propagated.

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor or a (sensor index) to render the scene from a
            different viewpoint. By default, the first sensor within the scene
            description (index 0) will take precedence.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator. It is crucial that you specify different seeds (e.g., an
            increasing sequence) if subsequent calls should produce statistically
            independent images (e.g. to de-correlate gradient-based optimization
            steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            differential rendering step. The value provided within the original
            scene specification takes precedence if ``spp=0``.
        """

        #print(scene.sensors())
        sensor = scene.sensors()[2]
        film = sensor.film()
        aovs = self.aovs()
        spp = 8
        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)
            # When the underlying integrator supports reparameterizations,
            # perform necessary initialization steps and wrap the result using
            # the _ReparamWrapper abstraction defined above
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam,
                    wavefront_size=sampler.wavefront_size(),
                    seed=seed
                )
            else:
                reparam = None

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)
            
            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, state_out, path_info = self.sample_path(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                reparam=None,
                active=mi.Bool(True),
                log_path=True
            )


            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            with dr.resume_grad():
                dr.enable_grad(L)

                # Accumulate into the image block.
                # After reparameterizing the camera ray, we need to evaluate
                #   Σ (fi Li det)
                #  ---------------
                #   Σ (fi det)
                if (dr.all(mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special))):
                    aovs = sensor.film().prepare_sample(L * weight * 1, ray.wavelengths,
                                                        block.channel_count(),
                                                        weight=1,
                                                        alpha=dr.select(valid, mi.Float(1), mi.Float(0)))
                    block.put(pos, aovs)
                    del aovs
                else:
                    block.put(
                        pos=pos,
                        wavelengths=ray.wavelengths,
                        value=L * weight * det,
                        weight=det,
                        alpha=dr.select(valid, mi.Float(1), mi.Float(0))
                    )

                sensor.film().put_block(block)

                # Probably a little overkill, but why not.. If there are any
                # DrJit arrays to be collected by Python's cyclic GC, then
                # freeing them may enable loop simplifications in dr.eval().
                del valid
                gc.collect()

                # This step launches a kernel
                dr.schedule(state_out, block.tensor())
                image = sensor.film().develop()

                # Differentiate sample splatting and weight division steps to
                # retrieve the adjoint radiance

                # decide whether using color derivatives or path derivatives based on input shape
                if grad_in.shape[-1]==3:
                    dr.set_grad(image, grad_in)
                    dr.enqueue(dr.ADMode.Backward, image)
                    dr.traverse(mi.Float, dr.ADMode.Backward)
                    δL = dr.grad(L)
                else:

                    # seperate path grad
                    tmp = ray.d.torch().reshape(-1,spp,3)
                    res = int(math.sqrt(tmp.shape[0]))
                    grad_in = grad_in[:res,:res,:]

                    # same as above
                    grad_color_in = grad_in[...,:3]
                    dr.set_grad(image, grad_color_in)
                    dr.enqueue(dr.ADMode.Backward, image)
                    dr.traverse(mi.Float, dr.ADMode.Backward)
                    δL = dr.grad(L)

                    # get gradient of ray_direction based on ray_differential
                    tmp = tmp.reshape(res,res,spp,3)
                    grad_d = tmp.clone()
                    tmp_x = ray.d_x.torch().reshape(res,res,-1,3)
                    tmp_y = ray.d_y.torch().reshape(res,res,-1,3)
                    grad_pos = grad_in[...,3:].torch()[...,None,:]
                    grad_d = (tmp_x-tmp)*grad_pos[...,0:1]+(tmp_y-tmp)*grad_pos[...,1:2]
                    dlduv = torch.zeros((path_info[1]["uv"][0].shape[0],1,len(path_info*2))).cuda()
                    grad_d = mi.Vector3f(grad_d.reshape(-1,3))

                    #camera grad enable
                    if dr.grad_enabled(ray.o):
                        dr.backward(ray.o*-grad_d, flags = dr.ADFlag.ClearInterior)

                    dr.enable_grad(ray.d)
                    dr.set_grad(ray.d, grad_d)
                    pi = scene.ray_intersect_preliminary(ray, coherent=True)
                    si = pi.compute_surface_interaction(ray, mi.RayFlags.All)
                    dr.forward_to(si.p, flags=dr.ADFlag.ClearEdges)
                    dlduv[:,0,0] = dr.grad(si.b0).torch()
                    dlduv[:,0,1] = dr.grad(si.b1).torch()
                    dldp1 = dr.grad(si.p).torch()
                    dr.disable_grad(ray.d)
                    dr.set_grad(ray.d, 0)
                    Lt = L.torch()
                    Lt = torch.sum(Lt,dim=-1)
                    path_grad, light_grad, diffuse_grad = self.calc_grad(path_info=path_info,dlduv=dlduv,dldp=dldp1,Lt=Lt)
                    # img = image.torch()
                    # img = torch.sum(img,dim=-1)
                    #final_grad, light_grad, diffuse_grad = calc_grad(path_info=path_info,dlduv=dlduv,dldp=dldp1,Lt=L.torch())
                    #final_grad, light_grad, diffuse_grad = calc_grad_caustic(path_info=path_info,dlduv=dlduv,dldp=dldp1,Lt=Lt)
                    #print(diffuse_grad[0].torch()[Lt==0])
                    #exit()
            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, state_out_2, _ = self.sample_path(
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=δL,
                state_in=state_out,
                reparam=reparam,
                active=mi.Bool(True),
                final_grad = path_grad,
                light_grad = light_grad,
                diffuse_grad = diffuse_grad,
                Lt=Lt
            )

            # We don't need any of the outputs here
            del L_2, valid_2, state_out, state_out_2, δL, \
                ray, weight, pos, block, sampler

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()
    
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
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

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path Replay Backpropagation (%s)" % mode.name,
                       state=lambda: (sampler, ray, depth, L, δL, β, η, active,
                                      prev_si, prev_bsdf_pdf, prev_bsdf_delta))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.

            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=dr.eq(depth, 0))

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

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)

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

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State for the differential phase
        )

    def sample_path(self,
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
                pi = scene.ray_intersect_preliminary(ray,coherent=True,active=active)
                si = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All)
                si_follow = pi.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape)
                if not primal and iteration*5+4<len(path_grad) and dr.grad_enabled(si.p):
                    dr.backward(si.p0*path_grad[iteration*5]+si.p1*path_grad[iteration*5+1]+si.p2*path_grad[iteration*5+2], flags = dr.ADFlag.ClearInterior)
                if not primal and iteration<len(diffuse_grad) and dr.grad_enabled(si_follow.p):
                    dr.backward(si_follow.p*diffuse_grad[iteration], flags = dr.ADFlag.ClearInterior)
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
                    # handle shadow path for direct lighting (not finished yet)
                    if iteration==0 and max_depth<=3:
                        ray_direct = si.spawn_ray(ds.d)
                        si_direct = scene.ray_intersect(ray_direct,
                                            ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                            coherent=dr.eq(depth, 0),active=(active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)))
                        dis = dr.detach(dr.norm(ds.p-si_direct.p)/dr.norm(ds.p-si.p))
                        dis[dis<0.01] = 0
                        if iteration<len(diffuse_grad) and dr.grad_enabled(si_direct.p):
                            try:
                                dr.backward(si_direct.p*diffuse_grad[iteration]*dis, flags = dr.ADFlag.ClearInterior)
                            except:
                                pass
                    
                    ray_direct = si.spawn_ray(ds.d)
                    si_direct = scene.ray_intersect(ray_direct,
                                        ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                        coherent=dr.eq(depth, 0),active=active_em)
                    if iteration<len(light_grad) and dr.grad_enabled(si_direct.p):
                        dr.backward(si_direct.p*mi.Vector3f(light_grad[iteration].torch()*torch.sum(Lr_dir.torch(),dim=-1,keepdim=True)), flags = dr.ADFlag.ClearInterior)
                        #dr.backward(si_direct.p*light_grad[iteration], flags = dr.ADFlag.ClearInterior)
                    

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
                if not primal and iteration*5+4<len(path_grad) and dr.grad_enabled(si_follow.sh_frame.n):
                   dr.backward(bsdf_sample.hf*path_grad[iteration*5+4]+si_follow.sh_frame.n*path_grad[iteration*5+3], flags = dr.ADFlag.ClearInterior)
            #dr.disable_grad(bsdf_sample)        
            #dr.disable_grad(bsdf_weight)   
            if log_path and iteration<5:
                logs.append({"it":iteration,"active":(active&si.is_valid()).torch(),"bsdf":bsdf.flags(),"ismesh":si.ismesh.torch(),
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

                    # if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                    #     raise Exception(
                    #         "The contribution computed by the differential "
                    #         "rendering phase is not attached to the AD graph! "
                    #         "Raising an exception since this is usually "
                    #         "indicative of a bug (for example, you may have "
                    #         "forgotten to call dr.enable_grad(..) on one of "
                    #         "the scene parameters, or you may be trying to "
                    #         "optimize a parameter that does not generate "
                    #         "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    # if mode == dr.ADMode.Backward:
                    #     dr.backward_from(δL * Lo)
                    # else:
                    #     δL += dr.forward_to(Lo)

            depth[si.is_valid()] += 1
            active = active_next
            iteration+=1
        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L,                    # State for the differential phase，
            logs
        )
    
class ManifoldIntegrator(EPSMIntegrator):
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
                diffuse_grad.append(dldp)
                valid = (path_info[id]["ismesh"]>0)
            else:
                point_prev = get_point(path_info[id-1])
                valid &= (path_info[id]["ismesh"]>0)
            
            #ignore path after second diffuse point
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
            m = path_info[id]["hf"]
            add(n)
            add(m)
            transmat = create_local_frame(n)
            wi2 = torch.bmm(transmat,wi[...,None])[...,0]
            wo2 = torch.bmm(transmat,wo[...,None])[...,0]
            res = (wi2+wo2*path_info[id]["eta"][...,None])
            res = res/torch.norm(res,dim=-1,keepdim=True)
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
            dldlp[path_info[id]['active']==0] = 0
            dldlp[nolight] = 0
            dldlp = torch.nan_to_num(dldlp)
            light_grad.append(dldlp)

            if id<len(path_info)-1:
                path_info[id+1]['uv'][0].requires_grad_(True)
                path_info[id+1]['uv'][1].requires_grad_(True)
                path_info[id+1]['uv'][0].retain_grad()
                path_info[id+1]['uv'][1].retain_grad()
                point_next = get_point(path_info[id+1])
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
                    dldp[hasdiffuse>0] = 0
                    final_param_grad[idx]+=dldp

                duvddp = -torch.bmm(grad_uv_inv,param_diffuse_grad[:,:2*id,:])
                dlddp = torch.bmm(dlduv[...,:2*id],duvddp)[:,0,:] #(N,3)
                dlddp[valid==0]=0
                dlddp[path_info[id+1]['active']==0]=0
                dlddp[mi.has_flag(path_info[id+1]['bsdf'],mi.BSDFFlags.Diffuse).torch()==0] = 0
                dlddp[hasdiffuse>0] = 0
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

mi.register_integrator("manifold", lambda props: ManifoldIntegrator(props))


class ManifoldCausticIntegrator(EPSMIntegrator):
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
            
            #ignore path after second diffuse point
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

mi.register_integrator("manifold_caustic", lambda props: ManifoldCausticIntegrator(props))
