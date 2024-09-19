import osqp
import numpy as np
from scipy import sparse
from scipy.linalg import solve_continuous_are


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


# action of the agent
class Action(object):
    def __init__(self):
        # 动作控制，u是大小为2的动作空间数组
        # u[0]代表左右动作速度，u[1]代表上下动作速度
        self.u = None
        # communication action
        self.c = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        self.collide = True
        super(Landmark, self).__init__()


# 沿特定路径运动的障碍物
class Movmark(Landmark):
    def __init__(self):
        super(Movmark, self).__init__()
        # physical motor noise amount
        self.u_noise = None
        # agents are movable by default
        self.movable = True
        # state(包含位置p_pos[x,y]和速度p_vel[vx,vy])
        self.state = AgentState()
        self.action = Action()
        self.direct = 1

    def get_pos(self):
        x = self.state.p_pos[0]
        y = self.state.p_pos[1]
        return np.array([x,y])

    def control(self, direction, velocity=0.2):
        # 向量归一化
        magnitude = np.linalg.norm(direction)
        if magnitude == 0:
            magnitude = 1
        normalized_direction = direction / magnitude

        self.action.u[0] = normalized_direction[0] * velocity
        self.action.u[1] = normalized_direction[1] * velocity


# Agent
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # 控制范围
        self.u_range = 1.0
        # state(包含位置p_pos[x,y]和速度p_vel[vx,vy])
        self.state = AgentState()
        self.action = Action()
        # script behavior to execute
        self.action_callback = None


# CBF
class AgentCBF:
    # agent为套用CBF的对象
    def __init__(self, agent, terminal):
        self.agent = agent
        self.x = agent.state.p_pos[0]  # agent x position
        self.y = agent.state.p_pos[1]  # agent y position
        self.vx = agent.state.p_vel[0]  # agent x velocity
        self.vy = agent.state.p_vel[1]  # agent y velocity
        self.finalx = terminal.state.p_pos[0]
        self.finaly = terminal.state.p_pos[1]
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
        for i in range(len(world.movmarks)):
            self.collision_objects.append(world.movmarks[i].state)
        for i in range(len(world.agents)):
            if world.agents[i] is not self.agent:
                self.collision_objects.append(world.agents[i].state)

    def control(
            self,
            use_cbf: bool = False,
            cbf_alpha: float = 8e1,
            penalty_slack: float = 20,
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
        Q = 10 ** 2 * np.eye(4)
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


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.terminals = []
        self.landmarks = []
        self.movmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.05
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.terminals + self.movmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # 更新世界的状态
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # 收集 agent 的动作力（自发决策产生）
    def apply_action_force(self, p_force):
        # set applied forces
        for i, entity in enumerate(self.agents):
            if entity.movable:
                noise = (
                    np.random.randn(*entity.action.u.shape) * entity.u_noise
                    if entity.u_noise
                    else 0.0
                )
                p_force[i] = entity.action.u + noise
        return p_force

    # 收集作用在 entity 上的物理力（由环境作用被动产生）
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # 整合实体物理状态，包括速度（是否超过上限）、位置
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if isinstance(entity, Movmark):
                entity.state.p_vel = entity.action.u
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                            entity.state.p_vel
                            / np.sqrt(
                        np.square(entity.state.p_vel[0])
                        + np.square(entity.state.p_vel[1])
                    )
                            * entity.max_speed
                    )
            entity.state.p_pos += entity.state.p_vel * self.dt

    # 更新代理的状态
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # 计算两个实体之间的碰撞力
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
