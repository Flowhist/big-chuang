try:
    while True:
        action_list = []

        # 生成每个agent的动作指令
        for agent in env.agents:
            action_list.append(agent.control(collision_objects))

        # 执行动作
        obs_next, r, done, info = env.step(action_list)
        print(action_list)

        # 渲染环境
        env.render()

        # 更新观察
        obs = obs_next

        # 判断是否完成
        if all(done):
            break

except KeyboardInterrupt:
    print("终止运行")

finally:
    env.close()