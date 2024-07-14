import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import deque
from ursinaxball import Game
from ursinaxball.common_values import BaseMap, TeamID
from ursinaxball.modules import GameScore, PlayerHandler, ChaseBot, RandomBot
from haxballgym.utils.gamestates import GameState
from haxballgym.utils.terminal_conditions import TerminalCondition

# Inicializar el gráfico
plt.ion()  # Activa el modo interactivo
fig, (ax1, ax2) = plt.subplots(2, 1)  # Crea dos subgráficos: uno para la recompensa promedio, otro para la acumulativa
line1, = ax1.plot([], [], 'r-')  # Línea para la recompensa promedio (color rojo)
line2, = ax2.plot([], [], 'b-')  # Línea para la recompensa acumulativa (color azul)

def update_graph(episode_rewards, cumulative_rewards):
    """
    Actualiza los gráficos de recompensa promedio y recompensa acumulativa.

    Args:
        episode_rewards (deque): Una cola que contiene las recompensas de los últimos episodios.
        cumulative_rewards (list): Una lista que contiene las recompensas acumuladas a lo largo de todos los episodios.

    La función actualiza:
    - El gráfico de recompensa promedio con los datos de los episodios recientes.
    - El gráfico de recompensa acumulativa con la suma de las recompensas acumuladas.
    """
    # Actualiza el gráfico de recompensa promedio
    line1.set_xdata(np.append(line1.get_xdata(), len(episode_rewards)))
    line1.set_ydata(np.append(line1.get_ydata(), np.mean(episode_rewards)))
    ax1.relim()
    ax1.autoscale_view()

    # Actualiza el gráfico de recompensa acumulativa
    line2.set_xdata(np.append(line2.get_xdata(), len(cumulative_rewards)))
    line2.set_ydata(np.append(line2.get_ydata(), np.sum(cumulative_rewards)))
    ax2.relim()
    ax2.autoscale_view()

    # Redibuja los gráficos
    fig.canvas.draw()
    fig.canvas.flush_events()

# Inicialización del juego
game = Game(
    folder_rec="./recordings/",  # Carpeta para guardar las grabaciones del juego
    enable_vsync=True,  # Habilitar sincronización vertical (vsync)
    stadium_file=BaseMap.CLASSIC,  # Archivo del estadio a usar (mapa clásico)
)

# Configuración del puntaje del juego
game.score = GameScore(time_limit=3, score_limit=3)  # Límite de tiempo de 3 y límite de puntaje de 3

# Salto de ticks (para controlar la frecuencia de actualización del juego)
tick_skip = 2  # Saltar 2 ticks por cada actualización

class DQN(nn.Module):
    """
    Una red neuronal profunda (Deep Q-Network, DQN) para aproximar la función de acción-valor Q.

    Args:
        input_dim (int): La dimensión del espacio de entrada.
        output_dim (int): La dimensión del espacio de salida.

    Atributos:
        layers (nn.Sequential): Un contenedor secuencial de capas de la red neuronal, que incluye:
            - Una capa lineal con tamaño de entrada `input_dim` y tamaño de salida 128.
            - Una capa ReLU.
            - Una capa lineal con tamaño de entrada 128 y tamaño de salida 128.
            - Una capa ReLU.
            - Una capa lineal con tamaño de entrada 128 y tamaño de salida `output_dim`.
    """
    def __init__(self, input_dim, output_dim):
        """
        Inicializa la red neuronal con las capas especificadas.

        Args:
            input_dim (int): La dimensión del espacio de entrada.
            output_dim (int): La dimensión del espacio de salida.
        """
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),  # Aumentar el tamaño de las capas
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        Realiza una pasada hacia adelante a través de la red neuronal.

        Args:
            x: El tensor de entrada.

        Returns:
            Tensor: La salida de la red neuronal después de pasar a través de las capas.
        """
        return self.layers(x)

class DQNAgent:
    """
    Agente DQN para aprender una política de acción-valor.

    Args:
        state_dim (int): La dimensión del espacio de estados.
        action_dim (int): La dimensión del espacio de acciones.
        learning_rate (float): La tasa de aprendizaje para el optimizador. Por defecto es 0.001.
        gamma (float): El factor de descuento para las recompensas futuras. Por defecto es 0.99.
        epsilon (float): La probabilidad inicial de seleccionar una acción aleatoria. Por defecto es 0.7.

    Atributos:
        device (torch.device): El dispositivo (CPU o GPU) en el que se ejecuta la red.
        q_network (DQN): La red neuronal principal para aproximar Q.
        target_network (DQN): La red neuronal objetivo para estabilizar el entrenamiento.
        optimizer (torch.optim.Adam): El optimizador para entrenar la red neuronal.
        gamma (float): El factor de descuento para las recompensas futuras.
        epsilon (float): La probabilidad de seleccionar una acción aleatoria.
        epsilon_decay (float): El factor de decaimiento de epsilon.
        epsilon_min (float): El valor mínimo de epsilon.
        action_dim (int): La dimensión del espacio de acciones.
        memory (deque): Una memoria de repetición para almacenar experiencias pasadas.
        batch_size (int): El tamaño del lote para el entrenamiento de la red neuronal.
        learn_step (int): El contador de pasos de aprendizaje.
        learn_frequency (int): La frecuencia con la que se actualiza la red objetivo.
        previous_actions (list): Lista de acciones previas tomadas por el agente.
    """
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, memory_size, batch_size, epsilon_decay, epsilon_min):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.learn_step = 0
        self.learn_frequency = 4
        self.previous_actions = []

    def remember(self, state, action, reward, next_state, done):
        """
        Almacena una experiencia en la memoria.

        Args:
            state: El estado inicial.
            action: La acción tomada.
            reward: La recompensa recibida.
            next_state: El siguiente estado.
            done: Si el episodio ha terminado.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Entrena la red neuronal utilizando una muestra de experiencias almacenadas.

        Returns:
            float: El valor de pérdida del entrenamiento.
        """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.learn_frequency == 0:
            self.update_target_network()
        
        return loss.item()

    def select_action(self, player: PlayerHandler, state):
        """
        Selecciona una acción basada en el estado actual utilizando una política epsilon-greedy.

        Args:
            player (PlayerHandler): El controlador del jugador.
            state: El estado actual.

        Returns:
            int: La acción seleccionada.
        """
        if random.random() < self.epsilon:
            print("Chase bot logic")
            return self.chase_bot_logic(game, player)

        with torch.no_grad():
            print("DQN logic")
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        
    def chase_bot_logic(self, game: Game, player: PlayerHandler) -> int:
        """
        Lógica del bot para perseguir la pelota.

        Args:
            game (Game): El estado actual del juego.
            player (PlayerHandler): El controlador del jugador.

        Returns:
            int: La acción seleccionada.
        """
        action = 0  # Acción por defecto es no hacer nada
        ball = game.stadium_game.discs[0]
        threshold = 2

        # Verificar posición horizontal
        if player.disc.position[0] - ball.position[0] > threshold:
            action = 1  # La pelota está a la izquierda, moverse a la izquierda
        elif player.disc.position[0] - ball.position[0] < -threshold:
            action = 2  # La pelota está a la derecha, moverse a la derecha

        # Verificar posición vertical
        if player.disc.position[1] - ball.position[1] > threshold:
            action = 4  # La pelota está abajo, moverse hacia abajo
        elif player.disc.position[1] - ball.position[1] < -threshold:
            action = 3  # La pelota está arriba, moverse hacia arriba

        # Verificar si debe disparar
        dist = np.linalg.norm(np.array(ball.position) - np.array(player.disc.position))
        if (dist - player.disc.radius - ball.radius) < 15:
            # Asegurarse de que hay acciones previas y la última acción fue un disparo
            if not player._kick_cancel and self.previous_actions and self.previous_actions[-1] == 5:
                action = 0  # No hacer nada si el disparo es cancelado
            else:
                action = 5  # De lo contrario, disparar

        # Actualizar acciones previas con la acción actual
        self.previous_actions.append(action)
        if len(self.previous_actions) > 10:  # Limitar el tamaño a las últimas 10 acciones
            self.previous_actions.pop(0)

        return action

    def update_target_network(self):
        """
        Actualiza la red neuronal objetivo copiando los pesos de la red neuronal principal.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """
        Decae el valor de epsilon después de cada episodio.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def convert_action(action):
    """
    Convierte una acción numérica en una lista de comandos de acción.

    Args:
        action (int): El número que representa la acción a convertir.

    Returns:
        list: Una lista de tres elementos que representa la acción convertida.

    Las acciones se convierten de la siguiente manera:
    - 0: [0, 0, 0] - No hacer nada
    - 1: [-1, 0, 0] - Moverse a la izquierda
    - 2: [1, 0, 0] - Moverse a la derecha
    - 3: [0, 1, 0] - Moverse hacia arriba
    - 4: [0, -1, 0] - Moverse hacia abajo
    - 5: [0, 0, 1] - Disparar
    """
    if action == 0: return [0, 0, 0]
    if action == 1: return [-1, 0, 0]
    if action == 2: return [1, 0, 0]
    if action == 3: return [0, 1, 0]
    if action == 4: return [0, -1, 0]
    if action == 5: return [0, 0, 1]

def get_state(game):
    """
    Obtiene el estado actual del juego.

    Args:
        game: El estado actual del juego.

    Returns:
        np.array: Un array que contiene la posición y velocidad de la pelota, 
                  así como la posición y velocidad de los jugadores rojo y azul.
    """
    # Obtener la posición y velocidad de la pelota
    ball_pos = game.stadium_game.discs[0].position
    ball_vel = game.stadium_game.discs[0].velocity
    
    # Obtener la posición y velocidad del jugador rojo
    red_player_pos = game.get_player_by_id(0).disc.position
    red_player_vel = game.get_player_by_id(0).disc.velocity
    
    # Obtener la posición y velocidad del jugador azul
    blue_player_pos = game.get_player_by_id(1).disc.position
    blue_player_vel = game.get_player_by_id(1).disc.velocity

    # Crear un array con el estado del juego
    state = np.array([
        ball_pos[0], ball_pos[1],  # Posición de la pelota
        ball_vel[0], ball_vel[1],  # Velocidad de la pelota
        red_player_pos[0], red_player_pos[1],  # Posición del jugador rojo
        red_player_vel[0], red_player_vel[1],  # Velocidad del jugador rojo
        blue_player_pos[0], blue_player_pos[1],  # Posición del jugador azul
        blue_player_vel[0], blue_player_vel[1]  # Velocidad del jugador azul
    ])

    return state

class BlueScoreCondition(TerminalCondition):
    '''
    Condición de terminación basada en el puntaje del equipo azul.

    Args:
        max_blue_score (int): El puntaje máximo del equipo azul antes de que se cumpla la condición de terminación.

    Atributos:
        max_blue_score (int): El puntaje máximo del equipo azul.
        initial_blue_score (int, opcional): El puntaje inicial del equipo azul al comienzo de la condición.

    '''
    def __init__(self, max_blue_score: int):
        """
        Inicializa la condición de terminación con el puntaje máximo del equipo azul.

        Args:
            max_blue_score (int): El puntaje máximo del equipo azul antes de que se cumpla la condición de terminación.
        """
        super().__init__()
        self.max_blue_score = max_blue_score
        self.initial_blue_score = None

    def reset(self, initial_state: GameState):
        """
        Reinicia la condición de terminación con el estado inicial del juego.

        Args:
            initial_state (GameState): El estado inicial del juego desde el cual se reinicia la condición.
        """
        self.initial_blue_score = initial_state.score.blue

    def is_terminal(self, current_state: GameState) -> bool:
        """
        Verifica si se ha cumplido la condición de terminación basada en el puntaje del equipo azul.

        Args:
            current_state (GameState): El estado actual del juego.

        Returns:
            bool: True si la condición de terminación se cumple, False en caso contrario.
        """
        if self.initial_blue_score is None:
            return False

        if current_state.score.blue - self.initial_blue_score >= self.max_blue_score:
            return True
        return False

def get_reward(game, player):
    '''
    Calcula la recompensa para el jugador dado el estado actual del juego.

    Args:
        game: El estado actual del juego.
        player: El jugador para el cual se calcula la recompensa.

    Returns:
        reward (int): La recompensa calculada basada en varias condiciones del juego.

    Se asignan recompensas y penalizaciones basadas en:
    - Penalización por tener la pelota en su mitad del campo.
    - Penalización por la distancia del jugador a la pelota.
    - Recompensa por avanzar hacia la portería contraria.
    - Recompensas y penalizaciones ajustadas por goles anotados por el equipo azul o rojo.

    '''
    reward = 0
    ball_position = np.array(game.stadium_game.discs[0].position)
    player_position = np.array(player.disc.position)

    # Penalización por tener la pelota en su mitad del campo
    if ball_position[0] < 0:  # Asumiendo que la mitad está en x = 0
        reward -= 1

    # Penalización por distancia a la pelota
    if np.linalg.norm(ball_position - player_position) > 30:
        reward -= 2

    # Recompensa por avanzar hacia la portería contraria
    if ball_position[0] > player_position[0]:
        reward += 1

    # Ajusta las recompensas por gol
    if game.score.blue > game.score.red:
        reward += 10
    if game.score.blue < game.score.red:
        reward -= 10

    return reward

def train(agent, num_episodes, update_target_every=1, start_episode=0):
    '''
    Entrena al agente DQN durante un número determinado de
    episodios. Actualiza el gráfico de recompensas después de cada
    episodio y guarda el modelo después de cada episodio.

    Args:
        agent (DQNAgent): El agente DQN a entrenar
        num_episodes (int): Número de episodios de entrenamiento
        update_target_every (int): Frecuencia de actualización de la red objetivo
        start_episode (int): Número de episodio de inicio. Por defecto es 0.

    Returns:
        None
    '''

    episode_rewards = deque(maxlen=100)
    cumulative_rewards = []
    episode_steps = deque(maxlen=100)
    ball_touches = deque(maxlen=100)
    goals_scored = deque(maxlen=100)

    for episode in range(start_episode, start_episode + num_episodes):
        save_rec = False
        game.reset(save_recording=save_rec)
        state = get_state(game)
        done = False
        total_reward = 0
        steps = 0
        touches = 0
        initial_blue_score = game.score.blue
        blue_score_condition = BlueScoreCondition(max_blue_score=3)
        blue_score_condition.reset(game)

        max_steps = 5000  # Ajustar el número máximo de pasos

        while not done and steps < max_steps:
            print(agent.epsilon)
            actions = [[0, 0, 0], [0, 0, 0]]
            
            action_red = player_red.step(game)
            actions[0] = action_red
            
            action_blue = agent.select_action(player_blue, state)
            actions[1] = convert_action(action_blue)

            previous_blue_touched = game.get_player_by_team(TeamID.BLUE).touched_ball
            
            done = game.step(actions)
            next_state = get_state(game)
    
            # Actualización importante: pasar el jugador azul a get_reward
            reward = get_reward(game, player_blue)  # Asumiendo que player_blue es el jugador azul
            
            if game.get_player_by_team(TeamID.BLUE).touched_ball and not previous_blue_touched:
                touches += 1
            
            agent.remember(state, action_blue, reward, next_state, done)
            
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()

            state = next_state
            total_reward += reward
            cumulative_rewards.append(total_reward)
            steps += 1

        agent.decay_epsilon()

        goals = game.score.blue - initial_blue_score
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        ball_touches.append(touches)
        goals_scored.append(goals)

        if (episode + 1) % 10 == 0:
            update_graph(episode_rewards, cumulative_rewards)

            avg_reward = np.mean(episode_rewards)
            avg_steps = np.mean(episode_steps)
            avg_touches = np.mean(ball_touches)
            avg_goals = np.mean(goals_scored)
            
            print(f"Episode {episode + 1}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Steps (last 100): {avg_steps:.2f}")
            print(f"  Avg Ball Touches (last 100): {avg_touches:.2f}")
            print(f"  Avg Goals Scored (last 100): {avg_goals:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Memory Size: {len(agent.memory)}")
            if 'loss' in locals():
                print(f"  Last Loss: {loss:.4f}")
            print("-----------------------------")
        
        if (episode + 1) % 100 == 0 and total_reward == 0:
            print(f"Episode {episode + 1}: No progress, resetting epsilon but keeping learned weights")
            agent.epsilon = 1.0
        else:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}, Steps: {steps}")

        if (episode + 1) % update_target_every == 0:
            agent.update_target_network()
            print(f"Episode {episode + 1}: Updated target network")

        if (episode + 1) % 1000 == 0:
            save_checkpoint(agent, episode + 1)
            print(f"Episode {episode + 1}: Saved model checkpoint")

        # Guardar el modelo después de cada episodio
        save_checkpoint(agent, episode + 1)
        print(f"Episode {episode + 1}: Saved model checkpoint")

def save_checkpoint(agent, episode):
    checkpoint = {
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'memory': agent.memory,
        'episode': episode
    }
    torch.save(checkpoint, f'dqn_checkpoint_episode_{episode}.pth')
    torch.save(checkpoint, 'dqn_checkpoint_latest.pth')  # Guardar el último checkpoint como 'latest'

def load_checkpoint(filename):
    """
    Carga un punto de control (checkpoint) desde un archivo.

    Args:
        filename (str): El nombre del archivo que contiene el punto de control.

    Returns:
        dict: Un diccionario con el contenido del punto de control.
    """
    # Cargar el punto de control usando torch.load
    checkpoint = torch.load(filename)
    return checkpoint

if __name__ == "__main__":
    # Configuraciones
    config = {
        'Bot': 2, # 0 para controlar el jugador rojo, 1 activar ChaseBot, 2 para RandomBot
        'state_dim': 12,
        'action_dim': 6,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon': 0.7,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'memory_size': 20000,
        'batch_size': 128,
        'num_episodes': 10,
        'checkpoint_path': 'dqn_checkpoint_latest.pth',
        'tick_skip': 2
    }

    # Inicializar el agente DQN con las configuraciones
    agent = DQNAgent(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon=config['epsilon'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        epsilon_decay=config['epsilon_decay'],
        epsilon_min=config['epsilon_min']
    )

    player_blue = PlayerHandler("P2", TeamID.BLUE)
    
    # Inicializar el bot rojo y el controlador del jugador rojo
    if config['Bot'] == 0:
        player_red = PlayerHandler("P1", TeamID.RED)
    elif config['Bot'] == 1:
        bot_red = ChaseBot(config['tick_skip'])
        player_red = PlayerHandler("P1", TeamID.RED, bot=bot_red)
    elif config['Bot'] == 2:
        bot_red = RandomBot(config['tick_skip'])
        player_red = PlayerHandler("P1", TeamID.RED, bot=bot_red)

    # Añadir los jugadores al juego
    game.add_players([player_red, player_blue])
    
    # Cargar el checkpoint más reciente si existe
    try:
        checkpoint = load_checkpoint(config['checkpoint_path'])
        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        agent.target_network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        agent.memory = checkpoint['memory']
        start_episode = checkpoint['episode']
        print(f"Checkpoint loaded from episode {start_episode}")
    except FileNotFoundError:
        start_episode = 0
        print("No checkpoint found, starting from scratch.")
    
    # Entrenar al agente
    train(agent, config['num_episodes'], start_episode=start_episode)
    
    # Guardar el modelo final después del entrenamiento
    save_checkpoint(agent, start_episode + config['num_episodes'])
    print("Training completed. Final model saved.")
