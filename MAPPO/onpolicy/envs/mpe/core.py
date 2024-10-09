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

# properties of wall entities
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
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

    def control(self, direction, velocity=0.4):
        # 向量归一化
        magnitude = np.linalg.norm(direction)
        if magnitude == 0:
            magnitude = 1
        normalized_direction = direction / magnitude

        self.action.u[0] = normalized_direction[0] * velocity
        self.action.u[1] = normalized_direction[1] * velocity

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
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
        # 上一时刻与目的地距离
        self.prev_dist_to_goal = None
        # control range
        self.u_range = 1.0
        # state(包含位置p_pos[x,y]和速度p_vel[vx,vy])
        self.state = AgentState()
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # 目的地(一个类似位置p_pos[x,y]的坐标)
        self.goal = None

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
            cbf_alpha: float = 5e1,
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
        self.walls = []
        self.num_agents = 0
        self.num_movmarks = 0
        self.num_landmarks = 0
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.05
        # physical damping（阻尼）
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        # zoe 20200420
        self.world_length = 25
        self.world_step = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.terminals + self.movmarks
    
    # 返回所有障碍物
    @property
    def marks(self):
        return self.landmarks + self.movmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities (size相加)
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        # sns.color_palette("OrRd_d", n_adversaries)
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        # sns.color_palette("GnBu_d", n_good_agents)
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

    # update state of the world
    def step(self):
        # zoe 20200420
        self.world_step += 1
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
        # calculate and store distances between all entities
        if self.cache_dists:
            self.calculate_distances()

    # 收集 agent 的动作力（自发决策产生）
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(
                    *agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                # force = mass * a * action + n
                p_force[i] = (
                    agent.mass * agent.accel if agent.accel is not None else agent.mass) * agent.action.u + noise
        return p_force

    # 收集作用在 entity 上的物理力（由环境作用被动产生）
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force

    # 整合实体物理状态，包括速度（是否超过上限）、位置
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            # movmark没有p_force，所以单独控制
            if isinstance(entity, Movmark):
                entity.state.p_vel = entity.action.u
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    # 更新代理的状态
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * \
                agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]  # neither entity moves
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
                ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
              ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force
