import numpy as np
from scipy import sparse
import osqp
from multiagent.core import Agent

class Agent():

    def control(self):
        cbf_alpha = 0.1
        # nominal_ux, nominal_uy = 1, 1
        ux, uy = cbf_control()
        return [0, ux, 0, uy, 0]


    def cbf_control(self):
        calculate_h_and_coeffs_dhdx()
        P, q, A, l, u = define_QP()

        if self.prob is None:
            # Create an OSQP object and setup workspace
            self.prob = osqp.OSQP()
            self.prob.setup(P, q, A, l, u, verbose=False, time_limit=0)
        else:
            self.prob.update(q=q, l=l, u=u, Ax=A.data)
        return


    def calculate_h_and_coeffs_dhdx(self, collision_objects: list):
        h = []  # barrier values (here, remaining distance to each obstacle)
        coeffs_dhdx = []  # dhdt = dhdx * dxdt = dhdx * u
        for obj in collision_objects:
            model_state = [self.x, self.y]
            self.model.update_params({"xr": obj.x, "yr": obj.y, "size": self.size})
            h.append(self.model.h(model_state))
            # Note: append additional elements for the slack variable δ
            coeffs_dhdx.append(self.model.h_dot(model_state) + [1])

            # NOTE: To speedup computation, we can calculate dhdu offline and hardcoded it as below
            # h.append((self.x - obj.x) ** 2 + (self.y - obj.y) ** 2 - (self.size * 2.3) ** 2)
            # coeffs_dhdx.append([2 * self.x - 2 * obj.x, 2 * self.y - 2 * obj.y, penalty_slack])
        return h, coeffs_dhdx


    def define_QP(
        self,
        ux: float,
        uy: float,
        cbf_alpha: float,
        penalty_slack: float,
        h: np.ndarray,
        coeffs_dhdx: np.ndarray,
        force_direction_unchanged: bool,
    ):
        # P: shape (nx, nx)
        # q: shape (nx,)
        # A: shape (nh+nx, nx)
        # l: shape (nh+nx,)
        # u: shape (nh+nx,)
        # (nx: number of state; nh: number of control barrier functions)
        P = sparse.csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        q = np.array([-ux, -uy, penalty_slack])
        A = sparse.csc_matrix([c for c in coeffs_dhdx] + [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if force_direction_unchanged:
            l = np.array(
                [-cbf_alpha * h_ for h_ in h] + [np.minimum(ux, 0), np.minimum(uy, 0), 0]
            )
            u = np.array(
                [np.inf for _ in h] + [np.maximum(ux, 0), np.maximum(uy, 0), np.inf]
            )
        else:
            l = np.array([-cbf_alpha * h_ for h_ in h] + [-self.vel, -self.vel, 0])
            u = np.array([np.inf for _ in h] + [self.vel, self.vel, np.inf])
        return P, q, A, l, u
