import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        # add agents（会动）
        # Agent()创建的agent归类为policy-agent
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = False
            agent.silent = True

        # add landmarks（不会动）
        world.landmarks = [Landmark() for i in range(2)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])

        # 一个Agent，在(-1,-1)，最大1
        for agent in world.agents:
            agent.state.p_pos = np.array([-0.5, -0.5])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # 两个障碍物，一个在(0,0)，一个在(0.5,0.5)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([0.6 * i, 0.4 * i])
            landmark.state.p_vel = np.zeros(world.dim_p)

    # reward(self, agent, world)定义了奖励函数，用于计算代理和第一个地标之间距离的平方的负值作为奖励。

    # observation(self, agent, world)定义了观察函数，用于返回代理参考框架下所有实体的位置和代理的速度

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
