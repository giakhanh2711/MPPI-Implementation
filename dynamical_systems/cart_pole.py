import torch

from dynamical_systems.dynamical_system import DynamicalSystem

class CartPole(DynamicalSystem):
    def __init__(self, m_c=1.0, m_p=0.1, l=1, g=9.81):
        super().__init__()
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.g = g


    def state_transition(self, x, u, dt):
      x_pos, x_dot, theta, theta_dot = x
      u = u.squeeze()  # đảm bảo scalar nếu u có shape (1,)

      # Precompute
      sin_theta = torch.sin(theta)
      cos_theta = torch.cos(theta)
      total_mass = self.m_c + self.m_p

      # Cart acceleration (velocity servo model)
      x_ddot = 10.0 * (u - x_dot)

      # Pole acceleration (derived from Lagrangian)
      theta_ddot = (
          self.g * sin_theta
          - x_ddot * cos_theta
      ) / (
          self.l * (4.0/3.0 - (self.m_p * cos_theta**2) / total_mass)
      )

      # Euler integration
      x_pos_next = x_pos + x_dot * dt
      x_dot_next = x_dot + x_ddot * dt
      theta_next = theta + theta_dot * dt
      theta_dot_next = theta_dot + theta_ddot * dt


      noise = torch.randn(4) * 0.01

      return torch.stack([x_pos_next, x_dot_next, theta_next, theta_dot_next]) + noise


    def cost_at_state(self, x):
        x, x_dot, theta, theta_dot = x
        cost = x**2 + 500*(1 + torch.cos(theta))**2 + theta_dot**2 + x_dot**2
        return cost