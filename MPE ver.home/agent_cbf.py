import numpy as np
from scipy.linalg import solve_continuous_are
from scipy import sparse
import osqp


class AgentCBF:
    def __init__(self, agent, landmark):
        self.x = agent.state.p_pos[0]  # agent x position
        self.y = agent.state.p_pos[1]  # agent y position
        self.vx = agent.state.p_vel[0]  # agent x velocity
        self.vy = agent.state.p_vel[1]  # agent y velocity
        self.finalx = landmark.state.p_pos[0]
        self.finaly = landmark.state.p_pos[1]
        self.ux = 0  # actual control input
        self.uy = 0  # actual control input
        self.u = np.array([self.ux, self.uy])
        self.u_max = np.array([10, 10])
        self.u_min = np.array([-10, -10])
        self.nominal_ux = 0  # user control input
        self.nominal_uy = 0  # user control input
        self.size = agent.size
        self.collision_objects = []
        # double integrator model parameter
        self.A = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0]])
        self.B = np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0]])

    def update_state(self, agent):
        self.x = agent.state.p_pos[0]  # agent x position
        self.y = agent.state.p_pos[1]  # agent y position
        self.vx = agent.state.p_vel[0]  # agent x velocity
        self.vy = agent.state.p_vel[1]  # agent y velocity

    def update_collision_objects(self, world, self_index):
        self.collision_objects = []
        for i in range(len(world.landmarks)):
            self.collision_objects.append(world.landmarks[i].state)

    def control(
        self,
        use_cbf: bool = False,
        cbf_alpha: float = 5e-1,
        penalty_slack: float = 5,
    ):
        """
        Processes control inputs, applies CBF for safety if enabled, and updates the robot's position.
        """
        self.nominal_ux, self.nominal_uy = 0, 0

        K = self._calculate_LQR_K()
        # get user command
        self._update_nominal_control(K)

        # get cbf filtered command
        if use_cbf:
            self._apply_cbf_safe_control(
                self.nominal_ux,
                self.nominal_uy,
                cbf_alpha,
                penalty_slack,
            )
        else:
            self.ux = self.nominal_ux
            self.uy = self.nominal_uy
        self.u = np.array([self.ux, self.uy])

    def _calculate_LQR_K(self):
        """
        take LQR controller as nominal controller
         X=[vx, vy, x - finalx, y - finaly]T U=[ux, uy]T
         X_dot=AX+BU
        """
        Q = 10**2*np.eye(4)
        R = np.eye(2)

        # Solve Riccati equation
        P = solve_continuous_are(self.A, self.B, Q, R)

        # calculate LQR K
        K = np.linalg.inv(R) @ self.B.T @ P

        return K

    def _update_nominal_control(self, K):
        error = np.array([self.vx,
                          self.vy,
                          self.x - self.finalx,
                          self.y - self.finaly])
        u = -K @ error
        self.nominal_ux = u[0]
        self.nominal_uy = u[1]

    def _apply_cbf_safe_control(
            self,
            ux: float,
            uy: float,
            cbf_alpha: float,
            penalty_slack: float,
    ):
        """
            Calculate the safe command by solveing the following optimization problem

                        minimize  || u - u_nom ||^2 + k * δ
                          u, δ
                        s.t.
                                h'(x) ≥ -𝛼 * h(x) - δ
                                u_min ≤ u ≤ u_max
                                    0 ≤ δ ≤ inf
            where
                u = [ux, uy] is the control input in x and y axis respectively.
                δ is the slack variable
                h(x) is the control barrier function and h'(x) its derivative

            The problem above can be formulated as QP (ref: https://osqp.org/docs/solver/index.html)

                        minimize 1/2 * x^T * Px + q^T x
                            x
                        s.t.
                                    l ≤ Ax ≤ u
            where
                x = [ux, uy, δ]

        """
        # Calculate barrier values and coeffs in h_dot
        h, coeffs_dhdx, Lf_h = self._calculate_h_and_coeffs_dhdx()

        # Define problem data
        P, q, A, l, u = self._define_QP_problem_data(
            ux, uy, cbf_alpha, penalty_slack, h, coeffs_dhdx, Lf_h
        )

        # Create an OSQP object and setup workspace
        self.prob = osqp.OSQP()
        self.prob.setup(P, q, A, l, u, verbose=False, time_limit=0)

        # Solve QP problem
        res = self.prob.solve()
        ux, uy, _ = res.x

        # Handle infeasible sol.
        ux = self.nominal_ux if ux is None else ux
        uy = self.nominal_uy if uy is None else uy

        self.ux, self.uy = ux, uy

    def _calculate_h_and_coeffs_dhdx(self):
        h = []  # barrier values (here, remaining distance to each obstacle)
        coeffs_dhdx = []  # dhdt = dhdx * dxdt = dhdx * u
        Lf_h = []
        for obj in self.collision_objects:
            # calculate h values
            h_value = 2 * ((obj.p_vel[0] - self.vx) * (obj.p_pos[0] - self.x) +
                           (obj.p_vel[1] - self.vy) * (obj.p_pos[1] - self.y)) + \
                      (self.x - obj.p_pos[0]) ** 2 + (self.y - obj.p_pos[1]) ** 2 - (self.size * 2.3) ** 2
            h.append(h_value)
            # Note: append additional elements for the slack variable δ
            coeffs_dhdx.append([2 * self.x - 2 * obj.p_pos[0], 2 * self.y - 2 * obj.p_pos[1], 1])
            Lf_h_value = 2 * self.vx * (self.vx + self.x - obj.p_vel[0] - obj.p_pos[0]) + \
                         2 * self.vy * (self.vy + self.y - obj.p_vel[1] - obj.p_pos[1])
            Lf_h.append(Lf_h_value)
        return np.array(h), np.array(coeffs_dhdx), np.array(Lf_h)

    def _define_QP_problem_data(
            self,
            ux: float,
            uy: float,
            cbf_alpha: float,
            penalty_slack: float,
            h: np.ndarray,
            coeffs_dhdx: np.ndarray,
            Lf_h: np.ndarray
    ):
        P = sparse.csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        q = np.array([-ux, -uy, penalty_slack])
        A = sparse.csc_matrix([c for c in coeffs_dhdx] + [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        l = np.array([-cbf_alpha * h[i] - Lf_h[i] for i in range(len(h))] + [self.u_min[0], self.u_min[1], 0])
        u = np.array([np.inf for _ in h] + [self.u_max[0], self.u_max[1], np.inf])

        return P, q, A, l, u
