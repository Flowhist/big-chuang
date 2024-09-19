import imageio
import numpy as np

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from multiagent.core import AgentCBF

# 创建场景、世界、环境
scenario = scenarios.load("simple_spread.py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

# 启用CBF避障
use_cbf = True
agent_cbf = [AgentCBF(agent, world.terminals[i]) for i, agent in enumerate(world.agents)]
action_list = []

# 初始化 imageio 编写器
writer = imageio.get_writer('simulation.mp4', fps=60)  # 保存为 mp4 格式

# 初始化可动障碍物的动作空间
for movmark in world.movmarks:
    movmark.action.u = np.zeros(world.dim_p)


while True:
    distance = 0

    # agent的控制
    for i, agent in enumerate(agent_cbf):
        agent.update_state(world.agents[i])
        agent.update_collision_objects(world, 0)
        agent.control(use_cbf=use_cbf)
        action_list.append([agent.ux, agent.uy])
    env.step(action_list)
    action_list = []

    # 渲染并获取图像数据
    frame = env.render(mode='rgb_array')[0]
    writer.append_data(frame)

    for i, agent in enumerate(world.agents):
        for j, terminal in enumerate(world.terminals):
            if i == j:
                distance += (agent.state.p_pos[0] - terminal.state.p_pos[0]) ** 2 +\
                            (agent.state.p_pos[1] - terminal.state.p_pos[1]) ** 2
    if distance < 0.00001:
        break

# 关闭编写器
writer.close()
