import torch
import numpy as np

from dynamical_systems.dynamical_system import DynamicalSystem

class MPPIController:
    def __init__(self, K, N, R, control_input_dim, disturbance_param, lambda_S, dt, system_obj: DynamicalSystem):
        self.K = K
        self.N = N
        self.R = R
        self.control_input_dim = control_input_dim
        self.disturbance_param = disturbance_param
        self.lambda_S = lambda_S
        self.dt = dt
        self.system_obj = system_obj
        
        self.init_u_seq = torch.zeros(self.N, self.control_input_dim) # Initial control input sequence

    def get_action(self, state, control_input_dim):
        state_dim = state.shape[0]

        u_init = torch.zeros(control_input_dim) # Value to initialize new controls to

        # Generate random control variations
        epsilon = torch.randn(size=(self.K, self.N, control_input_dim))
        deltau = (self.disturbance_param)*(epsilon / np.sqrt(self.dt)) # [K x N x control_input_dim]

        # Random x
        x_rollouts = torch.zeros(size=(self.K, self.N, state_dim))
        x_rollouts[:, 0, :] = state

        # Add noise
        # init_u_seq.unsqueeze(0) = [1 x N x control_input_dim]
        u_rollouts = self.init_u_seq.unsqueeze(0) + deltau # K x N x control_input_dim (K rollouts start with the same initial u sequence N, then added noise)

        S_rollouts = torch.zeros(self.K)
        for k in range(self.K):
            # Get one rollout state and one rollout control input
            x_seq = x_rollouts[k] # N x state_dim
            u_seq = u_rollouts[k] # N x control_input_dim

            # Cost 1 rollout
            cost = 0

            for i in range(self.N-1):
                x_seq[i + 1] = self.system_obj.state_transition(x_seq[i], u_seq[i], self.dt).clone()
                cost += self.system_obj.cost_at_state(x_seq[i + 1]) + u_seq[i].item()**2 * self.R # Cost at one state

            S_rollouts[k] = cost

        S_rollouts_score = torch.softmax((-1/self.lambda_S)*S_rollouts, dim=0) # K x 1

        # Update control input u
        # (deltau * S_rollouts_score) = [KxNxcontrol_input_dim] * [Kx1x1] = [KxNxcontrol_input_dim]
        # sum(dim=0) = [Nxcontrol_input_dim]
        self.init_u_seq = self.init_u_seq + (deltau * S_rollouts_score.unsqueeze(dim=-1).unsqueeze(-1)).sum(dim=0)
        u_0 = self.init_u_seq[0]

        self.init_u_seq = torch.roll(self.init_u_seq, shifts=-1, dims=0)
        self.init_u_seq[-1] = u_init

        return u_0