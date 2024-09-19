import numpy as np
from multiagent.core import World, Agent, Landmark, Entity, Movmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        # 二维，2个agent，2个landmark，2个movmark
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        num_movmarks = 2
        world.collaborative = True

        # 创建2个agent，size默认0.05
        # Agent()创建的agent归类为policy-agent
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True  # 支持碰撞检测（该agent是否与其他agent碰撞）
            agent.silent = True
        # 为每个agent创建一个terminal
        world.terminals = [Entity() for i in range(num_agents)]
        for i, terminal in enumerate(world.terminals):
            terminal.name = 'terminal %d' % i
            terminal.collide = False
            terminal.movable = False
        # 创建2个landmark，size默认0.05
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
        # 创建2个movmark，size默认0.05
        world.movmarks = [Movmark() for i in range(num_movmarks)]
        for i, movmark in enumerate(world.movmarks):
            movmark.name = 'movmark %d' % i
            movmark.collide = True
            movmark.movable = True
            if i == 0:
                movmark.direction = 1   # 左右移动
            else:
                movmark.direction = -2   # 上下移动

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # agents统一颜色
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # terminals统一颜色
        for i, terminal in enumerate(world.terminals):
            terminal.color = np.array([1.0, 0.75, 0.8])
        # marks统一颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        for i, movmark in enumerate(world.movmarks):
            movmark.color = np.array([0.75, 0.75, 0.75])

        # agents、terminal和landmarks的位置随机
        positions = []  # agent和landmark的位置数组，防止随机生成过于密集
        min_dis = 0.25
        for agent in world.agents:
            while True:
                new_pos = np.random.uniform(-1, -0.7, world.dim_p)
                if all(np.linalg.norm(new_pos - pos) >= min_dis for pos in positions):
                    positions.append(new_pos)
                    agent.state.p_pos = new_pos
                    agent.state.p_vel = np.zeros(world.dim_p)
                    agent.state.c = np.zeros(world.dim_c)
                    break
        for i, terminal in enumerate(world.terminals):
            if i == 0:
                terminal.state.p_pos = np.array([0.9, 0.83])
                terminal.state.p_vel = np.zeros(world.dim_p)
            if i == 1:
                terminal.state.p_pos = np.array([0.7, 0.97])
                terminal.state.p_vel = np.zeros(world.dim_p)
            # while True:
            #     new_pos = np.random.uniform(0.7, +1, world.dim_p)
            #     if all(np.linalg.norm(new_pos - pos) >= min_dis for pos in positions):
            #         positions.append(new_pos)
            #         terminal.state.p_pos = new_pos
            #         terminal.state.p_vel = np.zeros(world.dim_p)
            #         break
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.state.p_pos = np.array([-0.5, -0.5])
                landmark.state.p_vel = np.zeros(world.dim_p)
            if i == 1:
                landmark.state.p_pos = np.array([0.2, 0.5])
                landmark.state.p_vel = np.zeros(world.dim_p)
        for i, movmark in enumerate(world.movmarks):
            if i == 0:
                movmark.state.p_pos = np.array([0.2, 0.3])
                movmark.state.p_vel = np.zeros(world.dim_p)
            if i == 1:
                movmark.state.p_pos = np.array([-0.2, 0.4])
                movmark.state.p_vel = np.zeros(world.dim_p)
            # if i == 2:
            #     landmark.state.p_pos = np.array([0.454,0.055])
            #     landmark.state.p_vel = np.zeros(world.dim_p)
            # if i == 3:
            #     landmark.state.p_pos = np.array([-0.022,-0.245])
            #     landmark.state.p_vel = np.zeros(world.dim_p)
            # if i == 4:
            #     landmark.state.p_pos = np.array([0.206,0.511])
            #     landmark.state.p_vel = np.zeros(world.dim_p)
            # if i == 5:
            #     landmark.state.p_pos = np.array([-0.459,-0.349])
            #     landmark.state.p_vel = np.zeros(world.dim_p)
            # while True:
            #     new_pos = np.random.uniform(-0.6, +0.6, world.dim_p)
            #     if all(np.linalg.norm(new_pos - pos) >= min_dis for pos in positions):
            #         positions.append(new_pos)
            #         landmark.state.p_pos = new_pos
            #         landmark.state.p_vel = np.zeros(world.dim_p)
            #         break

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0   # 所有agent到每个landmark的最小距离之和
        for l in world.landmarks:
            # 计算每个landmark到所有agent的欧氏距离（数组）
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        # 对支持碰撞检测的agent进行检测
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return rew, collisions, min_dists, occupied_landmarks

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # 位置距离信息
        entity_pos = []
        for entity in world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # 颜色信息
        entity_color = []
        for entity in world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
