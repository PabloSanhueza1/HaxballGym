import time
from ursinaxball import Game
import haxballgym
from stable_baselines3 import A2C

# Para manejar multiples instancias del mismo entorno de forma paralela:
from stable_baselines3.common.env_util import make_vec_env

game = Game(
    folder_rec="./recordings/",
    enable_renderer=True,
    enable_vsync=True,
)

# 4 entornos paralelos
env = make_vec_env(lambda: haxballgym.make(game=game), n_envs=4)

# modelo con politica MlpPolicy (red neuronal MLP)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_haxball")

# base para cargar modelo
model = A2C.load("a2c_haxball")

ep_reward = 0

# Utilizar el modelo para interactuar con el entorno
obs = env.reset()

while True:
    save_rec = False
    if abs(ep_reward) > 1:
        save_rec = True
    obs = env.reset(save_recording=save_rec)
    obs_1 = obs[0]
    obs_2 = obs[1]
    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    while not done:
        actions_1 = env.action_space.sample()
        actions_2 = env.action_space.sample()
        actions = [actions_1, actions_2]
        new_obs, reward, done, state = env.step(actions)
        ep_reward += reward[0]
        obs_1 = new_obs[0]
        obs_2 = new_obs[1]
        steps += 1

    length = time.time() - t0
    print(
        "Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(
            length / steps, length, ep_reward
        )
    )