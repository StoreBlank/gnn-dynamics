import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import optimize
from torch.autograd import Variable

from gnn.utils import rotation_matrix_from_quaternion

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        s = x.size()
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [-1])

class Propagator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Propagator, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        s_x = x.size()

        x = self.linear(x.view(-1, s_x[-1]))

        if res is not None:
            s_res = res.size()
            x += res.view(-1, s_res[-1])

        x = self.relu(x).view(list(s_x[:-1]) + [-1])
        return x

class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x).view(list(s_x[:-1]) + [-1])

class DynamicsPredictor(nn.Module):
    def __init__(self, args, verbose=False, **kwargs):
        super(DynamicsPredictor, self).__init__()
        
        self.args = args
        self.verbose = verbose
        self.device = args.device
        
        self.nf_particle = args.nf_particle
        self.nf_relation = args.nf_relation
        self.nf_effect = args.nf_effect
        
        self.quat_offset = torch.tensor([1, 0, 0, 0], device=self.device)
        self.state_normalize = False
        
        # used for state normalization (not sued)
        self.mean_p = torch.FloatTensor(args.mean_p).to(args.device)
        self.std_p = torch.FloatTensor(args.std_p).to(args.device)
        self.mean_d = torch.FloatTensor(args.mean_d).to(args.device)
        self.std_d = torch.FloatTensor(args.std_d).to(args.device)
        self.motion_clamp = 100
        
        # Physics Encoder (TODO)
        self.physics_encoder = nn.Identity()
        self.phys_dim = args.phys_dim
        
        # ParticleEncoder
        particle_input_dim = (
            args.attr_dim # 2
            + args.density_dim # not used
            + args.n_his * args.state_dim # 4 * 0
            + args.n_his * args.offset_dim # 4 * 0
            + args.action_dim # 3
            + self.phys_dim # TODO: hyperparameter (2)
        )
        self.particle_encoder = Encoder(particle_input_dim, self.nf_particle, self.nf_effect)
        
        # RelationEncoder
        relation_input_dim = (
            args.rel_particle_dim * 2 # 0 * 2
            + args.rel_attr_dim * 2 # 2 * 2
            + args.rel_can_attr_dim # 0
            + args.rel_group_dim
            + args.n_his * args.rel_distance_dim
            + args.rel_density_dim
            + args.rel_canonical_distance_dim
            + args.rel_canonical_attr_dim
        )
        self.relation_encoder = Encoder(relation_input_dim, self.nf_relation, self.nf_effect)
        
        # ParticlePropagator
        self.particle_propagator = Propagator(self.nf_effect * 2, self.nf_effect)
        
        # RelationPropagator
        self.relation_propagator = Propagator(self.nf_effect * 3, self.nf_effect)
        
        # ParticlePredictor
        self.non_rigid_out_dim = kwargs["non_rigid_out_dim"]
        self.rigid_out_dim = kwargs["rigid_out_dim"]
        if kwargs["predict_non_rigid"]:
            self.non_rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, self.non_rigid_out_dim )
        else:
            self.non_rigid_predictor = None
        if kwargs["predict_rigid"]:
            self.rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, self.rigid_out_dim)
        else:
            self.rigid_predictor = None
        
        if verbose:
            print("DynamicsPredictor initialized")
            print("particle input dim: {}, relation input dim: {}".format(particle_input_dim, relation_input_dim))
            print("non-rigid output dim: {}, rigid output dim: {}".format(kwargs["non_rigid_out_dim"], kwargs["rigid_out_dim"]))
            
    def forward(self, state, attrs, Rr, Rs, p_instance, p_rigid,
                action=None, physics_param=None, particle_density=None,
                obj_mask=None, canonical_state=None, rel_can_attrs=None,
                verbose=False, phys_encoded_noise=0.1, **kwargs):
        
        args = self.args
        verbose = self.verbose
        
        # attrs: B x N x attr_dim
        # state: B x n_his x N x state_dim
        # Rr, Rs: B x n_rel x N
        # p_rigid: B x n_instance
        # p_instance: B x n_particle x n_instance
        # physics_param: B x n_particle
        # obj_mask: B x n_particle
        B, N = attrs.size(0), attrs.size(1) # B: batch size, N: total number of particles
        n_instance = p_instance.size(2) # n_instance: the number of instances
        n_p = p_instance.size(1) # n_p: the number of particles of the objects for prediction
        n_s = attrs.size(1) - n_p # n_s: the number of particles of the shapes (tools)
        
        n_rel = Rr.size(1) # n_rel: the number of relations
        state_dim = state.size(-1) # state dimension
        
        # Rr_t, Rs_t: B x N x n_rel
        Rr_t = Rr.transpose(1, 2).contiguous()
        Rs_t = Rs.transpose(1, 2).contiguous()
        p_instance_t = p_instance.transpose(1, 2)
        
        # particle belongings and rigidness (this is trying to keep both instance label and rigidity label)
        # p_rigid_per_particle: B x n_p x 1
        p_rigid_per_particle = torch.sum(p_instance * p_rigid[:, None, :], 2, keepdim=True)
        
        # state_residual: B x (n_his-1) x N x state_dim
        # state_current: B x 1 x N x state_dim
        state_res = state[:, 1:] - state[:, :-1]
        state_cur = state[:, -1:]
        
        # state normalization
        if self.state_normalize:
            # *_p: absolute scale to model scale, *_d: residual scale to model scale
            mean_p, std_p, mean_d, std_d = self.mean_p, self.std_p, self.mean_d, self.std_d
            state_res_norm = (state_res - mean_d) / std_d
            state_res_norm[:, :, :n_p, :] = 0  # n_p = 301?
            state_cur_norm = (state_cur - mean_p) / std_p
        else:
            state_res_norm = state_res
            state_cur_norm = state_cur
        
        # state_norm: B x n_his x N x state_dim
        # [0, n_his - 1): state_residual
        # [n_his - 1, n_his): the current position
        state_norm = torch.cat([state_res_norm, state_cur_norm], 1)
        state_norm_t = state_norm.transpose(1, 2).contiguous().view(B, N, args.n_his * state_dim)
        
        # instance_center: B x n_instance x (n_his * state_dim)
        instance_center = p_instance_t.bmm(state_norm_t[:, :n_p])
        instance_center /= torch.sum(p_instance, 1).unsqueeze(-1) + 1e-6
        
        # ================================================
        # ========== Preparing particle inputs ===========
        # ================================================
        # p_inputs: B x N x attr_dim
        p_inputs = attrs
        
        # state
        if args.state_dim > 0:
            # add state to attr
            # p_inputs: B x N x (attr_dim + n_his * state_dim)
            p_inputs = torch.cat([attrs, state_norm_t], 2)
        
        # offset
        if args.offset_dim > 0:
            # add offset to center-of-mass for rigids to attr
            # offset: B x N x (n_his * state_dim)
            offset = torch.zeros(B, N, args.n_his * state_dim, device=self.device)
            
            # c_per_particle: B x n_p x (n_his * state_dim)
            # particle offset: B x n_p x (n_his * state_dim)
            c_per_particle = p_instance.bmm(instance_center)
            c = (1 - p_rigid_per_particle) * state_norm_t[:, :n_p] + p_rigid_per_particle * c_per_particle
            offset[:, :n_p] = state_norm_t[:, :n_p] - c

            # p_inputs: B x N x (attr_dim + 2 * n_his * state_dim)
            p_inputs = torch.cat([p_inputs, offset], 2)

        # physics parameters
        if args.phys_dim > 0:
            assert physics_param is not None
            # physics_param: B x N x phys_dim
            phys_dim = physics_param.shape[2]
            physics_param_s = torch.zeros(B, n_s, phys_dim, device=self.device) # shape states do not have physical parameters
            physics_param = torch.cat([physics_param, physics_param_s], 1)
            
            physics_param_encoded = self.physics_encoder(physics_param)
            
            # p_inputs: B x N x (... + phys_dim)
            p_inputs = torch.cat([p_inputs, physics_param_encoded], 2)
        
        # actions
        if args.action_dim > 0:
            assert action is not None
            # p_inputs: B x N x (... + action_dim)
            p_inputs = torch.cat([p_inputs, action], 2)
        
        # density
        if args.density_dim > 0:
            assert particle_den is not None
            # particle_den = particle_den / 5000.

            # particle_den: B x N x 1
            particle_den = particle_den[:, None, None].repeat(1, n_p, 1)
            particle_den_s = torch.zeros(B, n_s, 1).to(args.device)
            particle_den = torch.cat([particle_den, particle_den_s], 1)

            # p_inputs: B x N x (... + density_dim)
            p_inputs = torch.cat([p_inputs, particle_den], 2)
        # ================================================
        # ========= Finish preparing p_inputs ============
        # ================================================
        
        # ================================================
        # ========== Preparing relation inputs ===========
        # ================================================
        rel_inputs = torch.empty((B, n_rel, 0), dtype=torch.float32, device=self.device)
        
        # particle relation
        if args.rel_particle_dim > 0:
            assert args.rel_particle_dim == p_inputs.size(2)
            # p_inputs_r: B x n_rel x -1
            # p_inputs_s: B x n_rel x -1
            p_inputs_r = Rr.bmm(p_inputs)
            p_inputs_s = Rs.bmm(p_inputs)

            # rel_inputs: B x n_rel x (2 x rel_particle_dim)
            rel_inputs = torch.cat([rel_inputs, p_inputs_r, p_inputs_s], 2)
        
        # relation attributes
        if args.rel_attr_dim > 0:
            assert args.rel_attr_dim == attrs.size(2)
            # attr_r: B x n_rel x attr_dim
            # attr_s: B x n_rel x attr_dim
            attrs_r = Rr.bmm(attrs)
            attrs_s = Rs.bmm(attrs)

            # rel_inputs: B x n_rel x (... + 2 x rel_attr_dim)
            rel_inputs = torch.cat([rel_inputs, attrs_r, attrs_s], 2)
        
        # relation group
        if args.rel_group_dim > 0:
            assert args.rel_group_dim == 1
            # receiver_group, sender_group
            # group_r: B x n_rel x -1
            # group_s: B x n_rel x -1
            g = torch.cat([p_instance, torch.zeros(B, n_s, n_instance).to(args.device)], 1)
            group_r = Rr.bmm(g)
            group_s = Rs.bmm(g)
            group_diff = torch.sum(torch.abs(group_r - group_s), 2, keepdim=True)

            # rel_inputs: B x n_rel x (... + 1)
            rel_inputs = torch.cat([rel_inputs, group_diff], 2)
        
        # relation distance
        if args.rel_distance_dim > 0:
            assert args.rel_distance_dim == 3
            # receiver_pos, sender_pos
            # pos_r: B x n_rel x -1
            # pos_s: B x n_rel x -1
            pos_r = Rr.bmm(state_norm_t)
            pos_s = Rs.bmm(state_norm_t)
            pos_diff = pos_r - pos_s

            # rel_inputs: B x n_rel x (... + 3)
            rel_inputs = torch.cat([rel_inputs, pos_diff], 2)
        
        # ================================================
        # ========= Finish preparing rel_inputs ==========
        # ================================================
        
        ## particle encode
        particle_encode = self.particle_encoder(p_inputs)
        particle_effect = particle_encode
        if verbose:
            print("particle encode:", particle_encode.size())
        
        ## relation encode
        relation_encode = self.relation_encoder(rel_inputs)
        if verbose:
            print("relation encode:", relation_encode.size())
        
        ## propagration steps (TODO: hyperparameter)
        for i in range(args.pstep):
            if verbose:
                print("propagation step:", i)
            
            # effect_r, effect_s: B x n_rel x nf
            effect_r = Rr.bmm(particle_effect)
            effect_s = Rs.bmm(particle_effect)
            
            # relation effect
            # effect_rel: B x n_rel x nf
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2)
            )
            if verbose:
                print("relation effect:", effect_rel.size())
            
            # particle effect by aggregating relation effect
            effect_rel_agg = Rr_t.bmm(effect_rel)
            
            # particle effect
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2), res=particle_effect
            )
            if verbose:
                print("particle effect:", particle_effect.size())
        
        # non_rigid_motion
        if self.non_rigid_predictor is not None:
            # non_rigid_motion: B x n_p x state_dim
            non_rigid_motion = self.non_rigid_predictor(particle_effect[:, :n_p].contiguous())
        
        # rigid motion TODO
        if self.rigid_predictor is not None and self.rigid_out_dim == 7:
            # instance effect: B x n_instance x nf_effect
            n_instance = p_instance.size(2)
            instance_effect = p_instance_t.bmm(particle_effect[:, :n_p])
            
            # instance_rigid_params: (B * n_instance) x 7
            instance_rigid_params = self.rigid_predictor(instance_effect.view(B*n_instance, 7))
            
            # decode rotation
            # R: (B * n_instance) x 3 x 3
            R = rotation_matrix_from_quaternion(instance_rigid_params[:, :4] + self.quat_offset)
            if verbose:
                print("Rotation matrix", R.size(), "should be (B x n_instance, 3, 3)")
            
            # decode translation
            b = instance_rigid_params[:, 4:]
            if self.state_normalize:  # denormalize
                b = b * std_d + mean_d
            b = b.view(B * n_instance, 1, state_dim)
            if verbose:
                print("b", b.size(), "should be (B x n_instance, 1, state_dim)")
            
            # current particle state
            # p_0: B x 1 x n_p x state_dim -> (B * n_instance) x n_p x state_dim
            p_0 = state[:, -1:, :n_p]
            p_0 = p_0.repeat(1, n_instance, 1, 1).view(B * n_instance, n_p, state_dim)
            if verbose:
                print("p_0", p_0.size(), "should be (B x n_instance, n_p, state_dim)")
                
            # current per-instance center state
            c = instance_center[:, :, -3:]
            if self.state_normalize:  # denormalize
                c = c * std_p + mean_p
            c = c.view(B * n_instance, 1, state_dim)
            if verbose:
                print("c", c.size(), "should be (B x n_instance, 1, state_dim)")

            # updated state after rigid motion
            p_1 = torch.bmm(p_0 - c, R) + b + c
            if verbose:
                print("p_1", p_1.size(), "should be (B x n_instance, n_p, state_dim)")
            
            # compute difference for per-particle rigid motion
            # rigid_motion: B x n_instance x n_p x state_dim
            rigid_motion = (p_1 - p_0).view(B, n_instance, n_p, state_dim)
            if self.state_normalize:  # normalize
                rigid_motion = (rigid_motion - mean_d) / std_d
            
        if self.rigid_predictor is not None and self.rigid_out_dim == 3:
            # assert args.state_dim == 3
            # rigid_motion: B x n_p x state_dim
            rigid_motion = self.rigid_predictor(particle_effect[:, :n_p].contiguous())
        
        # aggregate motions
        rigid_part = p_rigid_per_particle[..., 0].bool()
        non_rigid_part = ~rigid_part
        
        rigid_part = obj_mask & rigid_part
        non_rigid_part = obj_mask & non_rigid_part

        pred_motion = torch.zeros(B, n_p, state_dim).to(args.device)
        if self.non_rigid_predictor is not None:
            pred_motion[non_rigid_part] = non_rigid_motion[non_rigid_part]
        if self.rigid_predictor is not None and self.rigid_out_dim == 7:
            pred_motion[rigid_part] = torch.sum(p_instance.transpose(1, 2)[..., None] * rigid_motion, 1)[rigid_part]
        if self.rigid_predictor is not None and self.rigid_out_dim == 3:
            pred_motion[rigid_part] = rigid_motion[rigid_part]
        if self.state_normalize:  # denormalize
            pred_motion = pred_motion * std_d + mean_d
        pred_pos = state[:, -1, :n_p] + torch.clamp(pred_motion, max=self.motion_clamp, min=-self.motion_clamp)
        if verbose:
            print('pred_pos', pred_pos.size())
        
        return pred_pos, pred_motion

class Model(nn.Module):
    def __init__(self, args, **kwargs):

        super(Model, self).__init__()

        self.args = args
        self.device = args.device

        # PropNet to predict forward dynamics
        self.dynamics_predictor = DynamicsPredictor(args, **kwargs)

    def init_memory(self, B, N):
        """
        memory  (B, mem_layer, N, nf_memory)
        """
        mem = torch.zeros(B, self.args.mem_nlayer, N, self.args.nf_effect, device=self.device)
        return mem

    def predict_dynamics(self, **inputs):
        """
        return:
        ret - predicted position of all particles, shape (n_particles, 3)
        """
        ret = self.dynamics_predictor(**inputs, verbose=self.args.verbose_model)
        return ret