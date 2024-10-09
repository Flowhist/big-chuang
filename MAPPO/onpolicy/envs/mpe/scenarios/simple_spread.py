import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark, Movmark, Entity
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # 二维，2个agent，2个landmark，2个movmark
        world.dim_c = 2
        world.num_agents = 2
        world.num_movmarks = 0
        world.num_landmarks = 4
        world.collaborative = True

        # 创建2个agent，size默认0.05
        # Agent()创建的agent归类为policy-agent
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True  # 支持碰撞检测（该agent是否与其他agent碰撞）
            agent.silent = True
        # 为每个agent创建一个terminal
        world.terminals = [Entity() for i in range(world.num_agents)]
        for i, terminal in enumerate(world.terminals):
            terminal.name = 'terminal %d' % i
            terminal.collide = False
            terminal.movable = False
        # 创建可移动障碍物movmark
        world.movmarks = [Movmark() for i in range(world.num_movmarks)]
        for i, movmark in enumerate(world.movmarks):
            movmark.name = 'movmark %d' % i
            movmark.collide = True
            movmark.movable = True
        # 创建2个landmark
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # world.assign_agent_colors()
        # world.assign_landmark_colors()
        # 初始化可动障碍物的动作空间
        for movmark in world.movmarks:
            movmark.action.u = np.zeros(world.dim_p)
        # agents统一颜色
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # terminals统一颜色
        for i, terminal in enumerate(world.terminals):
            terminal.color = np.array([1.0, 0.75, 0.8])
        # movmarks统一颜色
        for i, movmark in enumerate(world.movmarks):
            movmark.color = np.array([0.75, 0.75, 0.75])
        # landmarks统一颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])

        # 以下是位置生成(纯随机)
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.state.p_pos = np.array([-0.75,-0.75])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            if i == 1:
                agent.state.p_pos = np.array([0.75,0.75])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.state.p_pos = np.array([-0.45,-0.45])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.state.c = np.zeros(world.dim_c)
            if i == 1:
                landmark.state.p_pos = np.array([-0.45,0.35])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.state.c = np.zeros(world.dim_c)
            if i == 2:
                landmark.state.p_pos = np.array([0.35,-0.35])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.state.c = np.zeros(world.dim_c)
            if i == 3:
                landmark.state.p_pos = np.array([0.45,0.55])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.state.c = np.zeros(world.dim_c)
        for i, terminal in enumerate(world.terminals):
            for j, agent in enumerate(world.agents):
                if i == 0:
                    terminal.state.p_pos = np.array([0.9,0.9])
                    terminal.state.p_vel = np.zeros(world.dim_p)
                    if j == i:
                        agent.goal = terminal.state.p_pos
                if i == 1:
                    terminal.state.p_pos = np.array([-0.9,-0.9])
                    terminal.state.p_vel = np.zeros(world.dim_p)
                    if j == i:
                        agent.goal = terminal.state.p_pos

        # # 以下是位置生成(提前指定)
        # for i, agent in enumerate(world.agents):
        #     if i == 0:
        #         agent.state.p_pos = np.array([-0.75, -0.8])
        #         agent.state.p_vel = np.zeros(world.dim_p)
        #         agent.state.c = np.zeros(world.dim_c)
        #     if i == 1:
        #         agent.state.p_pos = np.array([-1.0, -1.0])
        #         agent.state.p_vel = np.zeros(world.dim_p)
        #         agent.state.c = np.zeros(world.dim_c)
        # # 生成terminal位置时顺便将agent和terminal用goal一一对应
        # for i, terminal in enumerate(world.terminals):
        #     for j, agent in enumerate(world.agents):
        #         if i == 0:
        #             terminal.state.p_pos = np.array([0.9, 0.8])
        #             terminal.state.p_vel = np.zeros(world.dim_p)
        #             if j == i:
        #                 agent.goal = terminal.state.p_pos
        #         if i == 1:
        #             terminal.state.p_pos = np.array([0.7, 0.95])
        #             terminal.state.p_vel = np.zeros(world.dim_p)
        #             if j == i:
        #                 agent.goal = terminal.state.p_pos
        # for i, movmark in enumerate(world.movmarks):
        #     if i == 0:
        #         movmark.state.p_pos = np.array([-0.2, 0.0])
        #         movmark.state.p_vel = np.zeros(world.dim_p)
        #     if i == 1:
        #         movmark.state.p_pos = np.array([0.0, 0.4])
        #         movmark.state.p_vel = np.zeros(world.dim_p)
        # for i, landmark in enumerate(world.landmarks):
        #     if i == 0:
        #         landmark.state.p_pos = np.array([-0.57, -0.47])
        #         landmark.state.p_vel = np.zeros(world.dim_p)
        #     if i == 1:
        #         landmark.state.p_pos = np.array([0.50, 0.53])
        #         landmark.state.p_vel = np.zeros(world.dim_p)

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

    def is_collision(self, agent, world):
        for entity in world.entities:
            if entity.collide and entity is not agent:
                delta_pos = agent.state.p_pos - entity.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + entity.size
                return True if dist < 1.75 * dist_min else False

    def reward(self, agent, world):
        rew = 0
        max_dist = 2.5
        pre_dist = 0.15
        goal_reached_threshold = 0.02

        # 计算 agent 到所有障碍物的最小欧几里得距离，距离越远，奖励越多
        # dist = min(np.sqrt(np.sum(np.square(agent.state.p_pos - m.state.p_pos))) for m in world.marks)
        # if dist < pre_dist:
        #     rew += 0.01 * dist

        # 碰撞惩罚
        if self.is_collision(agent, world):
            rew -= 10

        # 一定范围内逐步奖励
        dist_to_goal = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal)))
        # if dist_to_goal <= max_dist:
        #     rew += np.exp(-0.5 * dist_to_goal)
        
        # 如果当前距离比之前小，则给予奖励；反之，给予惩罚
        if agent.prev_dist_to_goal is not None and dist_to_goal < agent.prev_dist_to_goal:
            rew += 1
        else:
            rew -= 1
        agent.prev_dist_to_goal = dist_to_goal
        
        # 额外的目标达成奖励        
        if dist_to_goal < goal_reached_threshold:
            rew += 10

        return rew

    def observation(self, agent, world):
        # 位置距离信息
        entity_pos = []
        for entity in world.entities:
            if entity is not agent:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        # comm = []
        # other_pos = []
        # for other in world.agents:
        #     if other is agent: continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
