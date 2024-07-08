import time
from ursinaxball import Game
import haxballgym

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space_channels_first
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.env_checker import check_env

# Inicializa el juego de Ursinaxball
game = Game(
    folder_rec="./recordings/",
    enable_renderer=True,
    enable_vsync=True,
)

# Crea el entorno de haxballgym
env = haxballgym.make(game=game)

# Verifica el entorno con stable-baselines3
check_env(env, warn=True, skip_render_check=True)
