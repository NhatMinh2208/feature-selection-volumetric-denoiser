#volume feature output
import mitsuba as mi
import drjit as dr
from typing import Any, Callable, Iterable, Iterator, Tuple, List, TypeVar, Union,  overload
mi.set_variant('cuda_ad_rgb')
def index_spectrum(spec, idx):
    m = spec[0]
    #if mi.is_rgb:
    m[dr.eq(idx, 1)] = spec[1]
    m[dr.eq(idx, 2)] = spec[2]
    return m

def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    b2 = dr.sqr(pdf_b)
    w = a2 / (a2 + b2)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))
class vfo(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', -1)
        self.rr_depth = props.get('rr_depth', 5)
        self.hide_emitters = props.get('hide_emitters', False)

        self.use_nee = False
        self.nee_handle_homogeneous = False
        self.handle_null_scattering = False
        self.is_prepared = False

    def prepare_scene(self, scene):
        if self.is_prepared:
            return

        for shape in scene.shapes():
            for medium in [shape.interior_medium(), shape.exterior_medium()]:
                if medium:
                    # Enable NEE if a medium specifically asks for it
                    self.use_nee = self.use_nee or medium.use_emitter_sampling()
                    self.nee_handle_homogeneous = self.nee_handle_homogeneous or medium.is_homogeneous()
                    self.handle_null_scattering = self.handle_null_scattering or (not medium.is_homogeneous())
        self.is_prepared = True
        # By default enable always NEE in case there are surfaces
        self.use_nee = True
    
    #return ["radiance", "albedo", "sigma_s", "transmittance"
        #            ,"position", "sigma_t", "tau"] 
    def L_ss(self, scene, sampler, ray, medium, active):
        self.prepare_scene(scene)
        max_depth = 2
        result = mi.Color3f(0.0)
        ray = mi.Ray3f(ray)
        cam_pos = mi.Point3f(ray.o)
        depth = mi.UInt32(0)                          # Depth of current vertex
        #L = mi.Spectrum(0 if is_primal else state_in) # Radiance accumulator
       # δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        throughput = mi.Color3f(1.0)                   # Path throughput weight
        η = mi.Float(1.0)                               # Index of refraction
        active = mi.Bool(active)

        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        last_scatter_event = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)

        # TODO: Support sensors inside media
        # medium = mi.MediumPtr(medium)
        medium = dr.zeros(mi.MediumPtr)

        channel = 0
        depth = mi.UInt32(0)
        valid_ray = mi.Bool(False)
        specular_chain = mi.Bool(True)
        '''
        if mi.is_rgb: # Sample a color channel to sample free-flight distances
            n_channels = dr.size_v(mi.Spectrum)
            channel = mi.UInt32(dr.minimum(n_channels * sampler.next_1d(active), n_channels - 1))
        '''
        n_channels = 3
        channel = mi.UInt32(dr.minimum(sampler.next_1d(active) * n_channels, n_channels - 1))

        #volumetric feature
        z_cam_vector = mi.Vector3f(0.0)
        albedo = mi.Color3f(0.0)
        position = mi.Point3f(0.0)
        sigma_s = mi.Color3f(0.0)
        sigma_t = mi.Color3f(0.0)
        z_cam = mi.Float(0.0)
        z_v = mi.Float(0.0)
        T = mi.Color3f(0.0)
        tau = mi.Color3f(0.0)
        shape_bound = mi.Point3f(float('inf'))
        z_bound_vector = mi.Vector3f(0.0)
        loop = mi.Loop(name= "Volume Feature Output",
                    state=lambda: (sampler, active, depth, ray, medium, si,
                                   throughput,result ,needs_intersection,
                                   last_scatter_event, specular_chain, η,
                                   last_scatter_direction_pdf, valid_ray, 
                                   albedo, position, sigma_s, sigma_t, z_cam, z_v, T, tau, cam_pos, z_cam_vector, shape_bound, z_bound_vector))
        while loop(active):
            active &= dr.any(dr.neq(throughput, 0.0))
            q = dr.minimum(dr.max(throughput) * dr.sqr(η), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr
            throughput[perform_rr] = throughput * dr.rcp(q)

            active_medium = active & dr.neq(medium, None) # TODO this is not necessary
            active_surface = active & ~active_medium

            #with dr.resume_grad(when=not is_primal):
            # Handle medium sampling and potential medium escape
            u = sampler.next_1d(active_medium)
            mei = medium.sample_interaction(ray, u, channel, active_medium)
            mei.t = dr.detach(mei.t)

            ray.maxt[active_medium & medium.is_homogeneous() & mei.is_valid()] = mei.t
            intersect = needs_intersection & active_medium
            si[intersect] = scene.ray_intersect(ray, intersect)

            needs_intersection &= ~active_medium
            mei.t[active_medium & (si.t < mei.t)] = dr.inf

            # Evaluate ratio of transmittance and free-flight PDF
            tr, free_flight_pdf = medium.eval_tr_and_pdf(mei, si, active_medium)
            tr_pdf = index_spectrum(free_flight_pdf, channel)
            weight = mi.Color3f(1.0)
            weight[active_medium] *= dr.select(tr_pdf > 0.0, tr / dr.detach(tr_pdf), 0.0)

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()

            # Handle null and real scatter events
            if self.handle_null_scattering:
                scatter_prob = index_spectrum(mei.sigma_t, channel) / index_spectrum(mei.combined_extinction, channel)
                act_null_scatter = (sampler.next_1d(active_medium) >= scatter_prob) & active_medium
                act_medium_scatter = ~act_null_scatter & active_medium
                weight[act_null_scatter] *= mei.sigma_n / dr.detach(1 - scatter_prob)
            else:
                scatter_prob = mi.Float(1.0)
                act_medium_scatter = active_medium

            depth[act_medium_scatter] += 1
            last_scatter_event[act_medium_scatter] = dr.detach(mei)
            #TOCHANGE
            albedo[act_medium_scatter] = dr.clamp(mei.sigma_s / mei.sigma_t, 0, 1)
            position[act_medium_scatter] = mei.p
            sigma_s[act_medium_scatter] = mei.sigma_s
            sigma_t[act_medium_scatter] = mei.sigma_t
            
            z_cam_vector[act_medium_scatter] = mi.Vector3f(mei.p - cam_pos)
            z_cam[act_medium_scatter] = dr.sqrt(z_cam_vector.x*z_cam_vector.x + z_cam_vector.y*z_cam_vector.y + z_cam_vector.z*z_cam_vector.z)
            z_bound_vector[act_medium_scatter] = mi.Vector3f(mei.p - shape_bound)
            #z_v[act_medium_scatter] = mei.t
            z_v[act_medium_scatter] = dr.sqrt(z_bound_vector.x*z_bound_vector.x + z_bound_vector.y*z_bound_vector.y + z_bound_vector.z*z_bound_vector.z)
            T[act_medium_scatter] = tr
            tau[act_medium_scatter] = (dr.minimum(mei.t, si.t) - mei.mint) *  mei.combined_extinction
            # Don't estimate lighting if we exceeded number of bounces
            active &= depth < max_depth
            act_medium_scatter &= active
            if self.handle_null_scattering:
                ray.o[act_null_scatter] = dr.detach(mei.p)
                si.t[act_null_scatter] = si.t - dr.detach(mei.t)

            weight[act_medium_scatter] *= mei.sigma_s / dr.detach(scatter_prob)
            throughput *= dr.detach(weight)

            mei = dr.detach(mei)

            '''
            if not is_primal and dr.grad_enabled(weight):
                Lo = dr.detach(dr.select(active_medium | escaped_medium, L / dr.maximum(1e-8, weight), 0.0))
                dr.backward(δL * weight * Lo)
            '''
            phase_ctx = mi.PhaseFunctionContext(sampler)
            phase = mei.medium.phase_function()
            phase[~act_medium_scatter] = dr.zeros(mi.PhaseFunctionPtr)

            valid_ray |= act_medium_scatter
            #with dr.suspend_grad():
            wo, phase_pdf = phase.sample(phase_ctx, mei, sampler.next_1d(act_medium_scatter), sampler.next_2d(act_medium_scatter), act_medium_scatter)
            act_medium_scatter &= phase_pdf > 0.0
            new_ray = mei.spawn_ray(wo)
            ray[act_medium_scatter] = new_ray
            needs_intersection |= act_medium_scatter
            last_scatter_direction_pdf[act_medium_scatter] = phase_pdf

            #--------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium
            intersect = active_surface & needs_intersection
            si[intersect] = scene.ray_intersect(ray, intersect)

            # ---------------- Intersection with emitters ----------------
            ray_from_camera = active_surface & dr.eq(depth, 0)
            count_direct = ray_from_camera | specular_chain
            emitter = si.emitter(scene)
            active_e = active_surface & dr.neq(emitter, None) & ~(dr.eq(depth, 0) & self.hide_emitters)

            # Get the PDF of sampling this emitter using next event estimation
            ds = mi.DirectionSample3f(scene, si, last_scatter_event)
            if self.use_nee:
                emitter_pdf = scene.pdf_emitter_direction(last_scatter_event, ds, active_e)
            else:
                emitter_pdf = 0.0
            emitted = emitter.eval(si, active_e)
            contrib = dr.select(count_direct, throughput * emitted,
                                throughput * mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted)
            #L[active_e] += dr.detach(contrib if is_primal else -contrib)
            #if not is_primal and dr.grad_enabled(contrib):
            #    dr.backward(δL * contrib)
            result[active_e] += contrib
            active_surface &= si.is_valid()
            ctx = mi.BSDFContext()
            bsdf = si.bsdf(ray)

            # --------------------- Emitter sampling ---------------------
            if self.use_nee:
                active_e_surface = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & (depth + 1 < max_depth)
                sample_emitters = mei.medium.use_emitter_sampling()
                specular_chain &= ~act_medium_scatter
                specular_chain |= act_medium_scatter & ~sample_emitters

                active_e_medium = act_medium_scatter & sample_emitters
                active_e = active_e_surface | active_e_medium

                nee_sampler = sampler #if is_primal else sampler.clone()
                emitted, ds = self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                    scene, nee_sampler, medium, channel, active_e)

                # Query the BSDF for that emitter-sampled direction
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, si.to_local(ds.d), active_e_surface)
                phase_val = phase.eval(phase_ctx, mei, ds.d, active_e_medium)
                nee_weight = dr.select(active_e_surface, bsdf_val, phase_val)
                nee_directional_pdf = dr.select(ds.delta, 0.0, dr.select(active_e_surface, bsdf_pdf, phase_val))

                contrib = throughput * nee_weight * mis_weight(ds.pdf, nee_directional_pdf) * emitted
                result[active_e] += dr.detach(contrib)
                '''
                if not is_primal:
                    self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                        scene, nee_sampler, medium, channel, active_e, adj_emitted=contrib,
                        δL=δL, mode=mode)

                    if dr.grad_enabled(nee_weight) or dr.grad_enabled(emitted):
                        dr.backward(δL * contrib)
                '''
            # ----------------------- BSDF sampling ----------------------
            #with dr.suspend_grad():
            bs, bsdf_weight = bsdf.sample(ctx, si, sampler.next_1d(active_surface),
                                        sampler.next_2d(active_surface), active_surface)
            active_surface &= bs.pdf > 0

            bsdf_eval = bsdf.eval(ctx, si, bs.wo, active_surface)
            '''
            if not is_primal and dr.grad_enabled(bsdf_eval):
                Lo = bsdf_eval * dr.detach(dr.select(active, L / dr.maximum(1e-8, bsdf_eval), 0.0))
                if mode == dr.ADMode.Backward:
                    dr.backward_from(δL * Lo)
                else:
                    δL += dr.forward_to(Lo)
            '''
            throughput[active_surface] *= bsdf_weight
            η[active_surface] *= bs.eta
            bsdf_ray = si.spawn_ray(si.to_world(bs.wo))
            ray[active_surface] = bsdf_ray
            #TOCHANGE
            z_cam[active_surface & active] = si.t
            #z_v[active_surface & active] = si.t
            shape_bound[active_surface & active & dr.eq(depth, 0)] = si.t
            needs_intersection |= active_surface
            non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
            depth[non_null_bsdf] += 1

            # update the last scatter PDF event if we encountered a non-null scatter event
            last_scatter_event[non_null_bsdf] = si
            last_scatter_direction_pdf[non_null_bsdf] = bs.pdf

            valid_ray |= non_null_bsdf
            specular_chain |= non_null_bsdf & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            specular_chain &= ~(active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Smooth))
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)
            active &= (active_surface | active_medium)
        return result, albedo, position, sigma_s, sigma_t, T, tau, z_cam, z_v
    
    def L_ms(self, scene, sampler, ray, medium, active, max_depth):
        self.prepare_scene(scene)
        max_depth = max_depth
        result = mi.Color3f(0.0)
        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        #L = mi.Spectrum(0 if is_primal else state_in) # Radiance accumulator
       # δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        throughput = mi.Color3f(1.0)                   # Path throughput weight
        η = mi.Float(1.0)                               # Index of refraction
        active = mi.Bool(active)

        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        last_scatter_event = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)

        # TODO: Support sensors inside media
        # medium = mi.MediumPtr(medium)
        medium = dr.zeros(mi.MediumPtr)

        channel = 0
        depth = mi.UInt32(0)
        valid_ray = mi.Bool(False)
        specular_chain = mi.Bool(True)
        '''
        if mi.is_rgb: # Sample a color channel to sample free-flight distances
            n_channels = dr.size_v(mi.Spectrum)
            channel = mi.UInt32(dr.minimum(n_channels * sampler.next_1d(active), n_channels - 1))
        '''
        n_channels = 3
        channel = mi.UInt32(dr.minimum(sampler.next_1d(active) * n_channels, n_channels - 1))

        #volumetric feature
        albedo = mi.Color3f(0.0)
        position = mi.Point3f(0.0)
        sigma_s = mi.Color3f(0.0)
        sigma_t = mi.Color3f(0.0)
        shape_bound = mi.Point3f(float('inf'))
        z_bound_vector = mi.Vector3f(0.0)
        z_v = mi.Float(0.0)
        T = mi.Color3f(0.0)
        tau = mi.Color3f(0.0)
        
        #TOCHANGE
        #mei = dr.zeros(mi.MediumInteraction3f)
        #tr = mi.Color3f(0.0)

        loop = mi.Loop(name= "Volume Feature Output",
                    state=lambda: (sampler, active, depth, ray, medium, si,
                                   throughput,result ,needs_intersection,
                                   last_scatter_event, specular_chain, η,
                                   last_scatter_direction_pdf, valid_ray, 
                                   albedo, position, sigma_s, sigma_t,  z_v, T, tau, shape_bound, z_bound_vector))
        while loop(active):
            active &= dr.any(dr.neq(throughput, 0.0))
            q = dr.minimum(dr.max(throughput) * dr.sqr(η), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr
            throughput[perform_rr] = throughput * dr.rcp(q)

            active_medium = active & dr.neq(medium, None) # TODO this is not necessary
            active_surface = active & ~active_medium

            #with dr.resume_grad(when=not is_primal):
            # Handle medium sampling and potential medium escape
            u = sampler.next_1d(active_medium)
            mei = medium.sample_interaction(ray, u, channel, active_medium)
            mei.t = dr.detach(mei.t)

            ray.maxt[active_medium & medium.is_homogeneous() & mei.is_valid()] = mei.t
            intersect = needs_intersection & active_medium
            si[intersect] = scene.ray_intersect(ray, intersect)

            needs_intersection &= ~active_medium
            mei.t[active_medium & (si.t < mei.t)] = dr.inf

            # Evaluate ratio of transmittance and free-flight PDF
            tr, free_flight_pdf = medium.eval_tr_and_pdf(mei, si, active_medium)
            tr_pdf = index_spectrum(free_flight_pdf, channel)
            weight = mi.Color3f(1.0)
            weight[active_medium] *= dr.select(tr_pdf > 0.0, tr / dr.detach(tr_pdf), 0.0)

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()

            # Handle null and real scatter events
            if self.handle_null_scattering:
                scatter_prob = index_spectrum(mei.sigma_t, channel) / index_spectrum(mei.combined_extinction, channel)
                act_null_scatter = (sampler.next_1d(active_medium) >= scatter_prob) & active_medium
                act_medium_scatter = ~act_null_scatter & active_medium
                weight[act_null_scatter] *= mei.sigma_n / dr.detach(1 - scatter_prob)
            else:
                scatter_prob = mi.Float(1.0)
                act_medium_scatter = active_medium

            depth[act_medium_scatter] += 1
            last_scatter_event[act_medium_scatter] = dr.detach(mei)
            #TOCHANGE
            #get the volume feature
            albedo[act_medium_scatter & dr.eq(depth, 3)] = dr.clamp(mei.sigma_s / mei.sigma_t, 0, 1)
            position[act_medium_scatter & dr.eq(depth, 3)] = mei.p
            sigma_s[act_medium_scatter & dr.eq(depth, 3)] = mei.sigma_s
            sigma_t[act_medium_scatter & dr.eq(depth, 3)] = mei.sigma_t
            T[act_medium_scatter & dr.eq(depth, 3)] = tr
            tau[act_medium_scatter & dr.eq(depth, 3)] = (dr.minimum(mei.t, si.t) - mei.mint) *  mei.combined_extinction

            z_bound_vector[act_medium_scatter & dr.eq(depth, 3)] = mi.Vector3f(mei.p - shape_bound)
            z_v[act_medium_scatter & dr.eq(depth, 3)] = dr.sqrt(z_bound_vector.x*z_bound_vector.x + z_bound_vector.y*z_bound_vector.y + z_bound_vector.z*z_bound_vector.z)

            # Don't estimate lighting if we exceeded number of bounces
            active &= depth < max_depth
            act_medium_scatter &= active
            if self.handle_null_scattering:
                ray.o[act_null_scatter] = dr.detach(mei.p)
                si.t[act_null_scatter] = si.t - dr.detach(mei.t)

            weight[act_medium_scatter] *= mei.sigma_s / dr.detach(scatter_prob)
            throughput *= dr.detach(weight)

            mei = dr.detach(mei)

            '''
            if not is_primal and dr.grad_enabled(weight):
                Lo = dr.detach(dr.select(active_medium | escaped_medium, L / dr.maximum(1e-8, weight), 0.0))
                dr.backward(δL * weight * Lo)
            '''
            phase_ctx = mi.PhaseFunctionContext(sampler)
            phase = mei.medium.phase_function()
            phase[~act_medium_scatter] = dr.zeros(mi.PhaseFunctionPtr)

            valid_ray |= act_medium_scatter
            #with dr.suspend_grad():
            wo, phase_pdf = phase.sample(phase_ctx, mei, sampler.next_1d(act_medium_scatter), sampler.next_2d(act_medium_scatter), act_medium_scatter)
            act_medium_scatter &= phase_pdf > 0.0
            new_ray = mei.spawn_ray(wo)
            ray[act_medium_scatter] = new_ray
            needs_intersection |= act_medium_scatter
            last_scatter_direction_pdf[act_medium_scatter] = phase_pdf

            #--------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium
            intersect = active_surface & needs_intersection
            si[intersect] = scene.ray_intersect(ray, intersect)

            # ---------------- Intersection with emitters ----------------
            ray_from_camera = active_surface & dr.eq(depth, 0)
            count_direct = ray_from_camera | specular_chain
            emitter = si.emitter(scene)
            active_e = active_surface & dr.neq(emitter, None) & ~(dr.eq(depth, 0) & self.hide_emitters)

            # Get the PDF of sampling this emitter using next event estimation
            ds = mi.DirectionSample3f(scene, si, last_scatter_event)
            if self.use_nee:
                emitter_pdf = scene.pdf_emitter_direction(last_scatter_event, ds, active_e)
            else:
                emitter_pdf = 0.0
            emitted = emitter.eval(si, active_e)
            contrib = dr.select(count_direct, throughput * emitted,
                                throughput * mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted)
            #L[active_e] += dr.detach(contrib if is_primal else -contrib)
            #if not is_primal and dr.grad_enabled(contrib):
            #    dr.backward(δL * contrib)
            result[active_e] += contrib
            active_surface &= si.is_valid()
            ctx = mi.BSDFContext()
            bsdf = si.bsdf(ray)

            # --------------------- Emitter sampling ---------------------
            if self.use_nee:
                active_e_surface = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & (depth + 1 < max_depth)
                sample_emitters = mei.medium.use_emitter_sampling()
                specular_chain &= ~act_medium_scatter
                specular_chain |= act_medium_scatter & ~sample_emitters

                active_e_medium = act_medium_scatter & sample_emitters
                active_e = active_e_surface | active_e_medium

                nee_sampler = sampler #if is_primal else sampler.clone()
                emitted, ds = self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                    scene, nee_sampler, medium, channel, active_e)

                # Query the BSDF for that emitter-sampled direction
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, si.to_local(ds.d), active_e_surface)
                phase_val = phase.eval(phase_ctx, mei, ds.d, active_e_medium)
                nee_weight = dr.select(active_e_surface, bsdf_val, phase_val)
                nee_directional_pdf = dr.select(ds.delta, 0.0, dr.select(active_e_surface, bsdf_pdf, phase_val))

                contrib = throughput * nee_weight * mis_weight(ds.pdf, nee_directional_pdf) * emitted
                result[active_e] += dr.detach(contrib)
                '''
                if not is_primal:
                    self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                        scene, nee_sampler, medium, channel, active_e, adj_emitted=contrib,
                        δL=δL, mode=mode)

                    if dr.grad_enabled(nee_weight) or dr.grad_enabled(emitted):
                        dr.backward(δL * contrib)
                '''
            # ----------------------- BSDF sampling ----------------------
            #with dr.suspend_grad():
            bs, bsdf_weight = bsdf.sample(ctx, si, sampler.next_1d(active_surface),
                                        sampler.next_2d(active_surface), active_surface)
            active_surface &= bs.pdf > 0

            bsdf_eval = bsdf.eval(ctx, si, bs.wo, active_surface)
            '''
            if not is_primal and dr.grad_enabled(bsdf_eval):
                Lo = bsdf_eval * dr.detach(dr.select(active, L / dr.maximum(1e-8, bsdf_eval), 0.0))
                if mode == dr.ADMode.Backward:
                    dr.backward_from(δL * Lo)
                else:
                    δL += dr.forward_to(Lo)
            '''
            throughput[active_surface] *= bsdf_weight
            η[active_surface] *= bs.eta
            bsdf_ray = si.spawn_ray(si.to_world(bs.wo))
            ray[active_surface] = bsdf_ray
            #TOCHANGE
            shape_bound[active_surface & active & dr.eq(depth, 0)] = si.t
            #z_v[active_surface & active] = si.t

            needs_intersection |= active_surface
            non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
            depth[non_null_bsdf] += 1

            # update the last scatter PDF event if we encountered a non-null scatter event
            last_scatter_event[non_null_bsdf] = si
            last_scatter_direction_pdf[non_null_bsdf] = bs.pdf

            valid_ray |= non_null_bsdf
            specular_chain |= non_null_bsdf & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            specular_chain &= ~(active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Smooth))
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)
            active &= (active_surface | active_medium)
        return result, albedo, position, sigma_s, sigma_t, T, tau,  z_v
    def sample(self, scene, sampler, ray, medium, active):
        L_ss, albedo_ss, position_ss, sigma_s_ss, sigma_t_ss, T_ss, tau_ss, z_cam, z_v = self.L_ss(scene,sampler,ray,medium,active) 
        color, albedo_ms, position_ms, sigma_s_ms, sigma_t_ms, T_ms, tau_ms,  z_v2 = self.L_ms(scene,sampler,ray,medium,active, 4)
        L_ms = color - L_ss

        self.prepare_scene(scene)
        result = mi.Color3f(0.0)
        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        #L = mi.Spectrum(0 if is_primal else state_in) # Radiance accumulator
       # δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        throughput = mi.Color3f(1.0)                   # Path throughput weight
        η = mi.Float(1.0)                               # Index of refraction
        active = mi.Bool(active)

        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        last_scatter_event = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)

        # TODO: Support sensors inside media
        # medium = mi.MediumPtr(medium)
        medium = dr.zeros(mi.MediumPtr)

        channel = 0
        depth = mi.UInt32(0)
        valid_ray = mi.Bool(False)
        specular_chain = mi.Bool(True)
        '''
        if mi.is_rgb: # Sample a color channel to sample free-flight distances
            n_channels = dr.size_v(mi.Spectrum)
            channel = mi.UInt32(dr.minimum(n_channels * sampler.next_1d(active), n_channels - 1))
        '''
        n_channels = 3
        channel = mi.UInt32(dr.minimum(sampler.next_1d(active) * n_channels, n_channels - 1))
        loop = mi.Loop(name= "Volume Feature Output",
                    state=lambda: (sampler, active, depth, ray, medium, si,
                                   throughput,result ,needs_intersection,
                                   last_scatter_event, specular_chain, η,
                                   last_scatter_direction_pdf, valid_ray))
        while loop(active):
            active &= dr.any(dr.neq(throughput, 0.0))
            q = dr.minimum(dr.max(throughput) * dr.sqr(η), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr
            throughput[perform_rr] = throughput * dr.rcp(q)

            active_medium = active & dr.neq(medium, None) # TODO this is not necessary
            active_surface = active & ~active_medium

            #with dr.resume_grad(when=not is_primal):
            # Handle medium sampling and potential medium escape
            u = sampler.next_1d(active_medium)
            mei = medium.sample_interaction(ray, u, channel, active_medium)
            mei.t = dr.detach(mei.t)

            ray.maxt[active_medium & medium.is_homogeneous() & mei.is_valid()] = mei.t
            intersect = needs_intersection & active_medium
            si[intersect] = scene.ray_intersect(ray, intersect)

            needs_intersection &= ~active_medium
            mei.t[active_medium & (si.t < mei.t)] = dr.inf

            # Evaluate ratio of transmittance and free-flight PDF
            tr, free_flight_pdf = medium.eval_tr_and_pdf(mei, si, active_medium)
            tr_pdf = index_spectrum(free_flight_pdf, channel)
            weight = mi.Color3f(1.0)
            weight[active_medium] *= dr.select(tr_pdf > 0.0, tr / dr.detach(tr_pdf), 0.0)

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()

            # Handle null and real scatter events
            if self.handle_null_scattering:
                scatter_prob = index_spectrum(mei.sigma_t, channel) / index_spectrum(mei.combined_extinction, channel)
                act_null_scatter = (sampler.next_1d(active_medium) >= scatter_prob) & active_medium
                act_medium_scatter = ~act_null_scatter & active_medium
                weight[act_null_scatter] *= mei.sigma_n / dr.detach(1 - scatter_prob)
            else:
                scatter_prob = mi.Float(1.0)
                act_medium_scatter = active_medium

            depth[act_medium_scatter] += 1
            last_scatter_event[act_medium_scatter] = dr.detach(mei)

            # Don't estimate lighting if we exceeded number of bounces
            active &= depth < self.max_depth
            act_medium_scatter &= active
            if self.handle_null_scattering:
                ray.o[act_null_scatter] = dr.detach(mei.p)
                si.t[act_null_scatter] = si.t - dr.detach(mei.t)

            weight[act_medium_scatter] *= mei.sigma_s / dr.detach(scatter_prob)
            throughput *= dr.detach(weight)

            mei = dr.detach(mei)

            '''
            if not is_primal and dr.grad_enabled(weight):
                Lo = dr.detach(dr.select(active_medium | escaped_medium, L / dr.maximum(1e-8, weight), 0.0))
                dr.backward(δL * weight * Lo)
            '''
            phase_ctx = mi.PhaseFunctionContext(sampler)
            phase = mei.medium.phase_function()
            phase[~act_medium_scatter] = dr.zeros(mi.PhaseFunctionPtr)

            valid_ray |= act_medium_scatter
            #with dr.suspend_grad():
            wo, phase_pdf = phase.sample(phase_ctx, mei, sampler.next_1d(act_medium_scatter), sampler.next_2d(act_medium_scatter), act_medium_scatter)
            act_medium_scatter &= phase_pdf > 0.0
            new_ray = mei.spawn_ray(wo)
            ray[act_medium_scatter] = new_ray
            needs_intersection |= act_medium_scatter
            last_scatter_direction_pdf[act_medium_scatter] = phase_pdf

            #--------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium
            intersect = active_surface & needs_intersection
            si[intersect] = scene.ray_intersect(ray, intersect)

            # ---------------- Intersection with emitters ----------------
            ray_from_camera = active_surface & dr.eq(depth, 0)
            count_direct = ray_from_camera | specular_chain
            emitter = si.emitter(scene)
            active_e = active_surface & dr.neq(emitter, None) & ~(dr.eq(depth, 0) & self.hide_emitters)

            # Get the PDF of sampling this emitter using next event estimation
            ds = mi.DirectionSample3f(scene, si, last_scatter_event)
            if self.use_nee:
                emitter_pdf = scene.pdf_emitter_direction(last_scatter_event, ds, active_e)
            else:
                emitter_pdf = 0.0
            emitted = emitter.eval(si, active_e)
            contrib = dr.select(count_direct, throughput * emitted,
                                throughput * mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted)
            #L[active_e] += dr.detach(contrib if is_primal else -contrib)
            #if not is_primal and dr.grad_enabled(contrib):
            #    dr.backward(δL * contrib)
            result[active_e] += contrib
            active_surface &= si.is_valid()
            ctx = mi.BSDFContext()
            bsdf = si.bsdf(ray)

            # --------------------- Emitter sampling ---------------------
            if self.use_nee:
                active_e_surface = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & (depth + 1 < self.max_depth)
                sample_emitters = mei.medium.use_emitter_sampling()
                specular_chain &= ~act_medium_scatter
                specular_chain |= act_medium_scatter & ~sample_emitters

                active_e_medium = act_medium_scatter & sample_emitters
                active_e = active_e_surface | active_e_medium

                nee_sampler = sampler #if is_primal else sampler.clone()
                emitted, ds = self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                    scene, nee_sampler, medium, channel, active_e)

                # Query the BSDF for that emitter-sampled direction
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, si.to_local(ds.d), active_e_surface)
                phase_val = phase.eval(phase_ctx, mei, ds.d, active_e_medium)
                nee_weight = dr.select(active_e_surface, bsdf_val, phase_val)
                nee_directional_pdf = dr.select(ds.delta, 0.0, dr.select(active_e_surface, bsdf_pdf, phase_val))

                contrib = throughput * nee_weight * mis_weight(ds.pdf, nee_directional_pdf) * emitted
                result[active_e] += dr.detach(contrib)
                '''
                if not is_primal:
                    self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                        scene, nee_sampler, medium, channel, active_e, adj_emitted=contrib,
                        δL=δL, mode=mode)

                    if dr.grad_enabled(nee_weight) or dr.grad_enabled(emitted):
                        dr.backward(δL * contrib)
                '''
            # ----------------------- BSDF sampling ----------------------
            #with dr.suspend_grad():
            bs, bsdf_weight = bsdf.sample(ctx, si, sampler.next_1d(active_surface),
                                        sampler.next_2d(active_surface), active_surface)
            active_surface &= bs.pdf > 0

            bsdf_eval = bsdf.eval(ctx, si, bs.wo, active_surface)
            '''
            if not is_primal and dr.grad_enabled(bsdf_eval):
                Lo = bsdf_eval * dr.detach(dr.select(active, L / dr.maximum(1e-8, bsdf_eval), 0.0))
                if mode == dr.ADMode.Backward:
                    dr.backward_from(δL * Lo)
                else:
                    δL += dr.forward_to(Lo)
            '''
            throughput[active_surface] *= bsdf_weight
            η[active_surface] *= bs.eta
            bsdf_ray = si.spawn_ray(si.to_world(bs.wo))
            ray[active_surface] = bsdf_ray

            needs_intersection |= active_surface
            non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
            depth[non_null_bsdf] += 1

            # update the last scatter PDF event if we encountered a non-null scatter event
            last_scatter_event[non_null_bsdf] = si
            last_scatter_direction_pdf[non_null_bsdf] = bs.pdf

            valid_ray |= non_null_bsdf
            specular_chain |= non_null_bsdf & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            specular_chain &= ~(active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Smooth))
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)
            active &= (active_surface | active_medium)

        '''
        return  (result, valid_ray, [L_ss[0], L_ss[1], L_ss[2], result[0], result[1],result[2], 
                                     albedo_ss[0], albedo_ss[1], albedo_ss[2], sigma_s_ss[0], sigma_s_ss[1], sigma_s_ss[2],  
                                       sigma_t_ss[0], sigma_t_ss[1], sigma_t_ss[2], position_ss[0], position_ss[1], position_ss[2], 
                                         T_ss[0], T_ss[1], T_ss[2], tau_ss[0], tau_ss[1], tau_ss[2], z_cam , z_v ])
        '''
        return  (result, valid_ray, [color[0], color[1], color[2], L_ss[0], L_ss[1], L_ss[2], L_ms[0], L_ms[1],L_ms[2], 
                                     albedo_ss[0], albedo_ss[1], albedo_ss[2],  albedo_ms[0], albedo_ms[1], albedo_ms[2], 
                                     sigma_s_ss[0], sigma_s_ss[1], sigma_s_ss[2], sigma_s_ms[0], sigma_s_ms[1], sigma_s_ms[2],   
                                       sigma_t_ss[0], sigma_t_ss[1], sigma_t_ss[2],  sigma_t_ms[0], sigma_t_ms[1], sigma_t_ms[2],
                                         position_ss[0], position_ss[1], position_ss[2], position_ms[0], position_ms[1], position_ms[2], 
                                         T_ss[0], T_ss[1], T_ss[2],  T_ms[0], T_ms[1], T_ms[2],
                                           tau_ss[0], tau_ss[1], tau_ss[2],  tau_ms[0], tau_ms[1], tau_ms[2], z_cam , z_v, z_v2 ])
    def sample_emitter(self, mei, si, active_medium, active_surface, scene, sampler, medium, channel, active):

        #is_primal = mode == dr.ADMode.Primal

        active = mi.Bool(active)

        ref_interaction = dr.zeros(mi.Interaction3f)
        ref_interaction[active_medium] = mei
        ref_interaction[active_surface] = si

        ds, emitter_val = scene.sample_emitter_direction(ref_interaction, sampler.next_2d(active), False, active)
        ds = dr.detach(ds)
        invalid = dr.eq(ds.pdf, 0.0)
        emitter_val[invalid] = 0.0
        active &= ~invalid

        medium = dr.select(active, medium, dr.zeros(mi.MediumPtr))
        medium[(active_surface & si.is_medium_transition())] = si.target_medium(ds.d)

        ray = ref_interaction.spawn_ray(ds.d)
        total_dist = mi.Float(0.0)
        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        transmittance = mi.Color3f(1.0)
        loop = mi.Loop(name="VFO Next Event Estimation",
                       state=lambda: (sampler, active, medium, ray, total_dist,
                                      needs_intersection, si, transmittance))
        while loop(active):
            remaining_dist = ds.dist * (1.0 - mi.math.ShadowEpsilon) - total_dist
            ray.maxt = dr.detach(remaining_dist)
            active &= remaining_dist > 0.0

            # This ray will not intersect if it reached the end of the segment
            needs_intersection &= active
            si[needs_intersection] = scene.ray_intersect(ray, needs_intersection)
            needs_intersection &= False

            active_medium = active & dr.neq(medium, None)
            active_surface = active & ~active_medium

            # Handle medium interactions / transmittance
            mei = medium.sample_interaction(ray, sampler.next_1d(active_medium), channel, active_medium)
            mei.t[active_medium & (si.t < mei.t)] = dr.inf
            mei.t = dr.detach(mei.t)

            tr_multiplier = mi.Color3f(1.0)

            # Special case for homogeneous media: directly advance to the next surface / end of the segment
            if self.nee_handle_homogeneous:
                active_homogeneous = active_medium & medium.is_homogeneous()
                mei.t[active_homogeneous] = dr.minimum(remaining_dist, si.t)
                tr_multiplier[active_homogeneous] = medium.eval_tr_and_pdf(mei, si, active_homogeneous)[0]
                mei.t[active_homogeneous] = dr.inf

            escaped_medium = active_medium & ~mei.is_valid()

            # Ratio tracking transmittance computation
            active_medium &= mei.is_valid()
            ray.o[active_medium] = dr.detach(mei.p)
            si.t[active_medium] = dr.detach(si.t - mei.t)
            tr_multiplier[active_medium] *= mei.sigma_n / mei.combined_extinction


            # Handle interactions with surfaces
            active_surface |= escaped_medium
            active_surface &= si.is_valid() & ~active_medium
            bsdf = si.bsdf(ray)
            bsdf_val = bsdf.eval_null_transmission(si, active_surface)
            tr_multiplier[active_surface] *= bsdf_val

            '''
            if not is_primal and dr.grad_enabled(tr_multiplier):
                active_adj = (active_surface | active_medium) & (tr_multiplier > 0.0)
                dr.backward(tr_multiplier * dr.detach(dr.select(active_adj, δL * adj_emitted / tr_multiplier, 0.0)))
            '''
            transmittance *= dr.detach(tr_multiplier)

            # Update the ray with new origin & t parameter
            new_ray = si.spawn_ray(mi.Vector3f(ray.d))
            ray[active_surface] = dr.detach(new_ray)
            ray.maxt = dr.detach(remaining_dist)
            needs_intersection |= active_surface

            # Continue tracing through scene if non-zero weights exist
            active &= (active_medium | active_surface) & dr.any(dr.neq(transmittance, 0.0))
            total_dist[active] += dr.select(active_medium, mei.t, si.t)

            # If a medium transition is taking place: Update the medium pointer
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)

        return emitter_val * transmittance, ds
    
    def aov_names(self: mi.Integrator):
        #return ["ss", "ms", "albedo", "sigma_s", "transmittance"
        #            ,"position", "sigma_t", "tau"]
        '''
         return ["L_ss.R", "L_ss.G", "L_ss.B", "L_ms.R", "L_ms.G", "L_ms.B",
                 "albedo_ss.R", "albedo_ss.G", "albedo_ss.B", 
                "sigma_s_ss.R", "sigma_s_ss.G", "sigma_s_ss.B",
                 "sigma_t_ss.R", "sigma_t_ss.G", "sigma_t_ss.B", 
                 "position_ss.X", "position_ss.Y", "position_ss.Z", 
                 "T_ss.R","T_ss.G", "T_ss.B", "tau.R", "tau.G", "tau.B",
                 "z_cam", "z_v"]
        '''
        return ["color.R", "color.G", "color.B",
                "L_ss.R", "L_ss.G", "L_ss.B", "L_ms.R", "L_ms.G", "L_ms.B",
                 "albedo_ss.R", "albedo_ss.G", "albedo_ss.B", "albedo_ms.R", "albedo_ms.G", "albedo_ms.B",
                "sigma_s_ss.R", "sigma_s_ss.G", "sigma_s_ss.B", "sigma_s_ms.R", "sigma_s_ms.G", "sigma_s_ms.B",
                 "sigma_t_ss.R", "sigma_t_ss.G", "sigma_t_ss.B", "sigma_t_ms.R", "sigma_t_ms.G", "sigma_t_ms.B", 
                 "position_ss.X", "position_ss.Y", "position_ss.Z", "position_ms.X", "position_ms.Y", "position_ms.Z", 
                 "T_ss.R","T_ss.G", "T_ss.B", "T_ms.R","T_ms.G", "T_ms.B",
                 "tau_ss.R", "tau_ss.G", "tau_ss.B",  "tau_ms.R", "tau_ms.G", "tau_ms.B",
                 "z_cam", "z_v_ss", "z_v_ms"]
    def to_string(self):
        return f'VFO[max_depth = {self.max_depth}]'


mi.register_integrator("vfo", lambda props: vfo(props))


'''
class vfp(mi.SamplingIntegrator):
    def __init__(self , arg0: mi.Properties) -> None:
        super().__init__(arg0)
        self.max_depth = arg0.get('max_depth', 12)
        self.rr_depth = arg0.get('rr_depth', 5)
        self.hide_emitters = mi.Mask(arg0.get('hide_emitters', False))
        
    def sample(self, scene: mi.Scene, sampler : mi.Sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True) -> Tuple[mi.Color3f, bool, List[float]]:
        #If there is an environment emitter and emitters are visible: all rays will be valid
        #Otherwise, it will depend on whether a valid interaction is sampled
        valid_ray =  ~self.hide_emitters & dr.neq(scene.environment(), None) 
        #For now, don't use ray differentials
        ray = mi.Ray3f(ray) 
        #Tracks radiance scaling due to index of refraction changes
        eta = mi.Float(1.0)
        throughput = mi.Color3f(1.0)
        result = mi.Color3f(0.0)
        medium = medium
        depth = mi.UInt32(0)
        channel = mi.UInt32(0)
        n_channels = mi.UInt32(3)
        channel = mi.UInt32(dr.minimum(sampler.next_1d(active) * n_channels, n_channels - 1))
        specular_chain = active & ~self.hide_emitters
        mei = dr.zeros(mi.MediumInteraction3f)
        #mei = medium.sample_interaction(ray, sampler.next_1d, channel, active)
        si = dr.zeros(mi.SurfaceInteraction3f)
        last_scatter_event = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)
        needs_intersection = mi.Bool(True)
        loop = mi.Loop("Volpath integrator",
                            lambda: (active, depth, ray, throughput,
                            result, si, mei, medium, eta, last_scatter_event,
                            last_scatter_direction_pdf, needs_intersection,
                            specular_chain, valid_ray, sampler))
        #loop = dr.cuda.Loop("Volpath integrator", active, depth, ray, throughput, result, si, mei, medium, eta, last_scatter_event, last_scatter_direction_pdf, needs_intersection,specular_chain, valid_ray, sampler)
        while(loop(active)):
            # ----------------- Handle termination of paths ------------------
            # Russian roulette: try to keep path weights equal to one, while accounting for the
            # solid angle compression at refractive index boundaries. Stop with at least some
            # probability to avoid  getting stuck (e.g. due to total internal reflection)
            active &= dr.any(dr.neq(throughput, 0.0))
            
            q = dr.minimum(dr.max(throughput) * dr.sqr(eta), 0.95)
            perform_rr = (depth >  mi.UInt32(self.rr_depth))
            active &= sampler.next_1d(active) < q | ~perform_rr
            #if(perform_rr):
            #   throughput *= dr.rcp(dr.detach(q))
            #throughput = dr.select(perform_rr, throughput * dr.rcp(dr.detach(q)), throughput)
            throughput[perform_rr] *= dr.rcp(dr.detach(q))
            active &= depth <  mi.UInt32(self.max_depth)
            #if (dr.none(active) or ~active):
            #    break
          
            #sampling RTE
            active_medium  = active & dr.neq(medium, None)
            active_surface = active & ~active_medium
            act_null_scatter = mi.Bool(False)
            act_medium_scatter = mi.Bool(False)
            escaped_medium = mi.Bool(False)
            
            # If the medium does not have a spectrally varying extinction,
            # we can perform a few optimizations to speed up rendering
            is_spectral = active_medium
            not_spectral = mi.Bool(False)
            #if (dr.any(active_medium) or active_medium):
            #is_spectral &= medium.has_spectral_extinction()
            is_spectral = dr.select(active_medium ,is_spectral & medium.has_spectral_extinction(), is_spectral)
            not_spectral = ~is_spectral & active_medium
            
            #if(dr.any(active_medium or active_medium)):
                 #sample t 
            mei : mi.MediumInteraction3f = medium.sample_interaction(ray, sampler.next_1d(active_medium), channel, active_medium)
            #if(active_medium & medium.is_homogeneous() & mei.is_valid()):
            ray.maxt[active_medium & medium.is_homogeneous() & mei.is_valid()] = mei.t
            intersect = needs_intersection & active_medium
            #if (dr.any(intersect) or intersect):
            si[intersect] = scene.ray_intersect(ray, intersect)
            needs_intersection &= ~active_medium
            #if(active_medium & (si.t < mei.t)):
            mei.t[active_medium & (si.t < mei.t)] = float('inf')
            #if (dr.any(is_spectral) or is_spectral):
            tr, free_flight_pdf = medium.eval_tr_and_pdf(mei, si, is_spectral)
            tr_pdf = self.index_spectrum(free_flight_pdf, channel)
            #if(is_spectral):
            throughput[is_spectral] *= dr.select(tr_pdf > 0.0, tr / tr_pdf, 0.0)
                
            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()

            #Handle null and real scatter events
            null_scatter = sampler.next_1d(active_medium) >= self.index_spectrum(mei.sigma_t, channel) / self.index_spectrum(mei.combined_extinction, channel)

            act_null_scatter |= null_scatter & active_medium
            act_medium_scatter |= ~act_null_scatter & active_medium

            #if ((dr.any(is_spectral) or is_spectral) & act_null_scatter):
            throughput[is_spectral & act_null_scatter] *= mei.sigma_n * self.index_spectrum(mei.combined_extinction, channel) / self.index_spectrum(mei.sigma_n, channel)
                
            #if(act_medium_scatter):
            depth[act_medium_scatter] += 1
            last_scatter_event[act_medium_scatter] = mei
            #Dont estimate lighting if we exceeded number of bounces
            active &= depth <  self.max_depth
            act_medium_scatter &= active
            
            #if (dr.any(act_null_scatter) or act_null_scatter): 
            ray.o[act_null_scatter] = mei.p
            si.t[act_null_scatter] = si.t - mei.t
            
            #if (dr.any(act_medium_scatter) or act_medium_scatter):
            #if (dr.any(is_spectral) or is_spectral):
            throughput[is_spectral & act_medium_scatter] *= mei.sigma_s * self.index_spectrum(mei.combined_extinction, channel) / self.index_spectrum(mei.sigma_t, channel)
            #if (dr.any(not_spectral) or not_spectral):
            throughput[not_spectral & act_medium_scatter] *= mei.sigma_s / mei.sigma_t
            phase_ctx = mi.PhaseFunctionContext(sampler)
            phase = mei.medium.phase_function()
            # Emitter sampling
            sample_emitters = mei.medium.use_emitter_sampling()
            valid_ray |= act_medium_scatter
            specular_chain &= ~act_medium_scatter
            specular_chain |= act_medium_scatter & ~sample_emitters

            active_e = act_medium_scatter & sample_emitters
            #if(dr.any(active_e) or active_e):
            emitted, ds = self.sample_emitter(mei, scene, sampler, medium, channel, active_e)
            phase_val = phase.eval(phase_ctx, mei, ds.d, active_e)
            result[active_e] += throughput * phase_val * emitted * self.mis_weight(ds.pdf, dr.select(ds.delta, 0.0, phase_val))

            #------------------ Phase function sampling -----------------
            #if(~act_medium_scatter):
            phase[~act_medium_scatter] = mi.PhaseFunctionPtr(None)
            wo, phase_pdf = phase.sample(phase_ctx, mei, sampler.next_1d(act_medium_scatter), sampler.next_2d(act_medium_scatter),act_medium_scatter)
            act_medium_scatter &= phase_pdf > 0.0    
            new_ray  = mei.spawn_ray(wo)
            #if(act_medium_scatter):
            ray[act_medium_scatter] = new_ray
            last_scatter_direction_pdf[act_medium_scatter] = phase_pdf
            needs_intersection |= act_medium_scatter

            #--------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium
            intersect = active_surface & needs_intersection
            #if (dr.any(intersect) or intersect):
            si[intersect] = scene.ray_intersect(ray, intersect)

            #if (dr.any(active_surface) or active_surface):
                #---------------- Intersection with emitters ----------------
            ray_from_camera = active_surface & dr.eq(depth, mi.UInt32(0))
            count_direct = ray_from_camera | specular_chain
            emitter = si.emitter(scene)
            active_e = active_surface & dr.neq(emitter, None) & ~(dr.eq(depth, mi.UInt32(0)) & self.hide_emitters)
            #if (dr.any(active_e) or active_e):
            emitter_pdf = 1.0
            #if (dr.any(active_e & ~count_direct) or (active_e & ~count_direct)):
            #Get the PDF of sampling this emitter using next event estimation
            ds = mi.DirectionSample3f(scene, si, last_scatter_event)
            emitter_pdf = scene.pdf_emitter_direction(last_scatter_event, ds, active_e)
            emitted = emitter.eval(si, active_e)
            contrib = dr.select(count_direct, throughput * emitted, throughput * self.mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted)
            #if (active_e):
            result[active_e] += contrib
            
            active_surface &= si.is_valid()
            #if (dr.any(active_surface) or active_surface):
                # --------------------- Emitter sampling ---------------------
            ctx = mi.BSDFContext()
            bsdf  = si.bsdf(ray)
            active_e = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & (depth + 1 < mi.UInt32(self.max_depth))

            #if (dr.any(active_e) or active_e): 
            emitted, ds = self.sample_emitter(si, scene, sampler, medium, channel, active_e)

            #Query the BSDF for that emitter-sampled direction
            wo       = si.to_local(ds.d)
            bsdf_val = bsdf.eval(ctx, si, wo, active_e)
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            #Determine probability of having sampled that same
            #direction using BSDF sampling.
            bsdf_pdf = bsdf.pdf(ctx, si, wo, active_e)
            result[active_e] += throughput * bsdf_val * self.mis_weight(ds.pdf, dr.select(ds.delta, 0.0, bsdf_pdf)) * emitted
                

            #----------------------- BSDF sampling ----------------------
            bs, bsdf_val = bsdf.sample(ctx, si, sampler.next_1d(active_surface),sampler.next_2d(active_surface), active_surface)
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi)

            #if(active_surface):
            throughput[active_surface] *= bsdf_val
            eta[active_surface] *= bs.eta
               
            bsdf_ray  = si.spawn_ray(si.to_world(bs.wo))
            #if(active_surface):
            ray[active_surface] = bsdf_ray
            needs_intersection |= active_surface

            non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
            #if(non_null_bsdf):
            depth[non_null_bsdf] += 1

            #update the last scatter PDF event if we encountered a non-null scatter event
            #if(non_null_bsdf):
            last_scatter_event[non_null_bsdf] = si
            last_scatter_direction_pdf[non_null_bsdf] = bs.pdf
                

            valid_ray |= non_null_bsdf
            specular_chain |= non_null_bsdf & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            specular_chain &= ~(active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Smooth))
            act_null_scatter |= active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
            has_medium_trans = active_surface & si.is_medium_transition()
            #if(has_medium_trans):
            medium[has_medium_trans] = si.target_medium(ray.d)
                
            active &= (active_surface | active_medium)

        return result, valid_ray, [dr.select(mei.is_valid(), mei.sigma_s, float(0.0))]
    
    def aov_names(self: mi.Integrator) -> List[str]:
        #return ["ss", "ms", "albedo", "sigma_s", "transmittance"
        #            ,"position", "sigma_t", "tau"]
        return ["L_ss.R", "L_ss.G", "L_ss.B", "L_ms.R", "L_ms.G", "L_ms.B",
                 "albedo_ss.R", "albedo_ss.G", "albedo_ss.B", "albedo_ms.R", "albedo_ms.G", "albedo_ms.B",
                "sigma_s_ss.R", "sigma_s_ss.G", "sigma_s_ss.B", "sigma_s_ms.R", "sigma_s_ms.G", "sigma_s_ms.B",
                 "sigma_t_ss.A", "sigma_t_ss.G", "sigma_t_ss.B", "sigma_t_ms.A", "sigma_t_ms.G", "sigma_t_ms.B", 
                 "position_ss.X", "position_ss.Y", "position_ss.Z", "position_ms.X", "position_ms.Y", "position_ms.Z",
                 "T", "tau"]

    def to_string():
        return "VFP[]"

    def mis_weight(self, pdf_a, pdf_b):
        pdf_a *= pdf_a
        pdf_b *= pdf_b
        w = pdf_a / (pdf_a + pdf_b)
        return dr.select(dr.isfinite(w), w, 0.0)
    
    def vol_albedo(sigma_s, sigma_t):
        return sigma_s / sigma_t
    
    #Optical thickness
    def tau(sigma_t, t):
        return 0
    
    #Scattering ratio: ratio of scattering interactions over the total number of paths in the pixel.
    def r_sct():
        return 0
    
    def transmittance():
        return 0
    
    def index_spectrum(self, spec : mi.Color3f, idx : mi.UInt32):
        m = spec[0]
        #if dr.eq(idx, mi.UInt32(1)):
        m[dr.eq(idx, mi.UInt32(1))] = spec[1]
        #if dr.eq(idx, mi.UInt32(2)):
        m[dr.eq(idx, mi.UInt32(2))] = spec[2]
        return m
    
    def sample_emitter(self, ref_interaction : mi.Interaction3f , scene : mi.Scene,
                   sampler : mi.Sampler , medium: mi.Medium , channel: mi.UInt32,
                   active : bool):
        transmittance = mi.Color3f(1.0)

        ds, emitter_val = scene.sample_emitter_direction(ref_interaction, sampler.next_2d(active), False, active)
        #if(dr.eq(ds.pdf, 0.0)):
        #emiiter_val = dr.select(dr.eq(ds.pdf, 0.0), 0.0, emitter_val)
        emitter_val[dr.eq(ds.pdf, 0.0)] = 0.0
        active &= dr.neq(ds.pdf, 0.0)

        #if (dr.none(active) or ~active):
        #    return emitter_val, ds 

        ray = ref_interaction.spawn_ray(ds.d)

        #// Potentially escaping the medium if this is the current medium's boundary
        #if constexpr (std::is_convertible_v<Interaction, SurfaceInteraction3f>)
        #medium[ref_interaction.is_medium_transition()] = ref_interaction.target_medium(ray.d)
        total_dist = mi.Float(0.0)
        si : mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)

        loop = mi.Loop("Volpath integrator emitter sampling", lambda: (active, ray, total_dist, needs_intersection, medium, si,transmittance))
       
        loop.put(active, ray, total_dist, needs_intersection, medium, si,
                 transmittance)
        sampler->loop_put(loop);
        loop.init();
     
        sampler.loop_put(loop)
        loop.init()
        while (loop(dr.detach(active))):
            remaining_dist = ds.dist * (mi.Float(1.0) - mi.math.ShadowEpsilon) - total_dist
            ray.maxt = remaining_dist
            active &= remaining_dist > 0.0
            #if (dr.none(active) or ~active):
            #    break

            escaped_medium = mi.Bool(False)
            active_medium  = active & dr.neq(medium, None)
            active_surface = active & ~active_medium

            #if (dr.any(active_medium) or active_medium):
            mei = medium.sample_interaction(ray, sampler.next_1d(active_medium), channel, active_medium)
            #if(active_medium & medium.is_homogeneous() & mei.is_valid()):
            ray.maxt[ray.maxt, active_medium & medium.is_homogeneous() & mei.is_valid()] = dr.minimum(mei.t, remaining_dist)
            intersect = needs_intersection & active_medium
            #if (dr.any(intersect) or intersect):
            si[intersect] = scene.ray_intersect(ray, intersect)
                
            #if(active_medium & (si.t < mei.t)):
            mei.t[active_medium & (si.t < mei.t)] = float('inf')
                
            needs_intersection &= ~active_medium

            is_spectral = medium.has_spectral_extinction() &active_medium
            not_spectral = ~is_spectral & active_medium
            #if (dr.any(is_spectral) or is_spectral):
            t = dr.minimum(remaining_dist, dr.minimum(mei.t, si.t)) - mei.mint
            tr  = mi.Color3f(dr.exp(-t * mei.combined_extinction))
            free_flight_pdf = mi.Color3f(dr.select(si.t < mei.t | mei.t > remaining_dist, tr, tr * mei.combined_extinction))
            tr_pdf = self.index_spectrum(free_flight_pdf, channel)
            #if(is_spectral):
            transmittance[is_spectral] *= dr.select(tr_pdf > 0.0, tr / tr_pdf, 0.0)
                

            #Handle exceeding the maximum distance by medium sampling
            #if(active_medium & (mei.t > remaining_dist) & mei.is_valid()):
            total_dist[active_medium & (mei.t > remaining_dist) & mei.is_valid()] = ds.dist
            #if(active_medium & (mei.t > remaining_dist)):
            mei.t[active_medium & (mei.t > remaining_dist)] = float('inf')
                

            escaped_medium = active_medium & ~mei.is_valid()
            active_medium &= mei.is_valid()
            is_spectral &= active_medium
            not_spectral &= active_medium

            #if(active_medium):
            total_dist[active_medium] += mei.t
                
            #if (dr.any(active_medium) or active_medium):
            #if(active_medium):
            ray.o[active_medium]   = mei.p
            si.t[active_medium] = si.t - mei.t

            #if (dr.any(is_spectral) or is_spectral):
            transmittance[is_spectral] *= mei.sigma_n
            #if (dr.any(not_spectral) or not_spectral):
            transmittance[not_spectral] *= mei.sigma_n / mei.combined_extinction
                
            

            #Handle interactions with surfaces
            intersect = active_surface & needs_intersection
            #if (dr.any(intersect) or intersect):
            si[intersect] = scene.ray_intersect(ray, intersect)
            needs_intersection &= ~intersect
            active_surface |= escaped_medium
            #if(active_surface):
            total_dist[active_surface] += si.t

            active_surface &= si.is_valid() & active & ~active_medium
            #if (dr.any(active_surface) or active_surface):
            bsdf     = si.bsdf(ray)
            bsdf_val = bsdf.eval_null_transmission(si, active_surface)
            bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi)
            #if(active_surface):
            transmittance[active_surface] *= bsdf_val

            #Update the ray with new origin & t parameter
            #if(active_surface):
            ray[active_surface] = si.spawn_ray(ray.d)
            ray.maxt = remaining_dist
            needs_intersection |= active_surface

            #Continue tracing through scene if non-zero weights exist
            active &= (active_medium | active_surface) & dr.any(dr.neq(mi.Color3f(transmittance), 0.0))

            #If a medium transition is taking place: Update the medium pointer
            has_medium_trans = active_surface & si.is_medium_transition()
            #if (dr.any(has_medium_trans) or has_medium_trans): 
            medium[has_medium_trans] = si.target_medium(ray.d)
        return transmittance * emitter_val, ds
    
# Register new integrator
mi.register_integrator("vfo", lambda props: vfo(props))

scene = mi.cornell_box()
scene['integrator']['type'] = 'vfo'
scene['integrator']['max_depth'] = 16
scene['integrator']['rr_depth'] = 2
scene['sensor']['sampler']['sample_count'] = 32
scene['sensor']['film']['width'] = 1024
scene['sensor']['film']['height'] = 1024
scene = mi.load_dict(scene)

img = mi.render(scene)

import matplotlib.pyplot as plt
plt.imshow(img ** (1. / 2.2))
plt.axis("off")
plt.show()


ptr = mi.SurfaceInteraction3f()
ptr.is_valid()
'''