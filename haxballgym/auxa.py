from dataclasses import dataclass
from ursina import Keys, held_keys
from ursinaxball import Game
from ursinaxball.common_values import BaseMap, TeamID
from ursinaxball.modules import GameScore, PlayerHandler, ChaseBot

import numpy as np
import random
from collections import defaultdict

# Configuración del juego
game = Game(
    folder_rec="./recordings/",
    enable_vsync=True,
    stadium_file=BaseMap.CLASSIC,
)
game.score = GameScore(time_limit=3, score_limit=3)
tick_skip = 2

# Ajuste de la velocidad del bot
bot_blue = ChaseBot(tick_skip)
bot_blue.speed = 10  # Ajusta la velocidad según sea necesario

player_red = PlayerHandler("P1", TeamID.RED)
player_blue = PlayerHandler("P2", TeamID.BLUE, bot=bot_blue)
game.add_players([player_red, player_blue])


@dataclass
class InputPlayer:
    left: list[str]
    right: list[str]
    up: list[str]
    down: list[str]
    shoot: list[str]


input_player = InputPlayer(
    left=[Keys.left_arrow],
    right=[Keys.right_arrow],
    up=[Keys.up_arrow],
    down=[Keys.down_arrow],
    shoot=["x"],
)


def action_handle(actions_player_output: list[int], inputs_player: InputPlayer):
    actions_player_output = [0, 0, 0]
    for key, value in held_keys.items():
        if value != 0:
            if key in inputs_player.left:
                actions_player_output[0] -= 1
            if key in inputs_player.right:
                actions_player_output[0] += 1
            if key in inputs_player.up:
                actions_player_output[1] += 1
            if key in inputs_player.down:
                actions_player_output[1] -= 1
            if key in inputs_player.shoot:
                actions_player_output[2] += 1
    return actions_player_output


class QLearningAgent:
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

    def select_action(self, state):
        state_tuple = tuple(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state_tuple])

    def update(self, state, action, reward, next_state):
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)
        best_next_action = np.argmax(self.q_table[next_state_tuple])
        td_target = reward + self.discount_factor * self.q_table[next_state_tuple][best_next_action]
        td_error = td_target - self.q_table[state_tuple][action]
        self.q_table[state_tuple][action] += self.learning_rate * td_error


def convert_action(action):
    if action == 0: return [0, 0, 0]
    if action == 1: return [-1, 0, 0]
    if action == 2: return [1, 0, 0]
    if action == 3: return [0, 1, 0]
    if action == 4: return [0, -1, 0]
    if action == 5: return [-1, 1, 0]
    if action == 6: return [1, 1, 0]
    if action == 7: return [-1, -1, 0]
    if action == 8: return [1, -1, 0]
    if action == 9: return [0, 0, 1]


def get_state(game):
    ball_pos = game.stadium_game.discs[0].position
    ball_vel = game.stadium_game.discs[0].velocity
    red_player_pos = game.get_player_by_id(0).disc.position
    red_player_vel = game.get_player_by_id(0).disc.velocity
    blue_player_pos = game.get_player_by_id(1).disc.position
    blue_player_vel = game.get_player_by_id(1).disc.velocity

    state = np.array([
        ball_pos[0], ball_pos[1],
        ball_vel[0], ball_vel[1],
        red_player_pos[0], red_player_pos[1],
        red_player_vel[0], red_player_vel[1],
        blue_player_pos[0], blue_player_pos[1],
        blue_player_vel[0], blue_player_vel[1]
    ])

    return state


def get_reward(game, previous_score):
    reward = 0

    blue_player = game.get_player_by_team(TeamID.BLUE)
    if blue_player is not None:
        if blue_player.touched_ball:
            reward += 1
        
        current_score = game.score.blue - game.score.red

        if current_score > previous_score:
            reward += 10  # Gol a favor
        elif current_score < previous_score:
            reward -= 5  # Gol en contra

        ball_pos = game.stadium_game.discs[0].position
        blue_pos = blue_player.disc.position
        distance_to_ball = np.linalg.norm(np.array(blue_pos) - np.array(ball_pos))

        reward += max(0, 10 - distance_to_ball)

    return reward


agent = QLearningAgent(action_space_size=10)  # Actualizado para más espacio de acción
num_episodes = 1000

for episode in range(num_episodes):
    save_rec = False
    game.reset(save_recording=save_rec)
    state = get_state(game)
    done = False
    total_reward = 0
    previous_score = game.score.blue - game.score.red

    while not done:
        actions = [[0, 0, 0], [0, 0, 0]]
        
        actions[0] = action_handle(actions[0], input_player)
        action_blue = agent.select_action(state)
        actions[1] = convert_action(action_blue)
        
        done = game.step(actions)
        next_state = get_state(game)
        reward = get_reward(game, previous_score)
        
        agent.update(state, action_blue, reward, next_state)
        state = next_state
        total_reward += reward

        previous_score = game.score.blue - game.score.red

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
