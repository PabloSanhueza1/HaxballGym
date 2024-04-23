# Entrenamiento de un agente para el juego Haxball mediante aprendizaje por reforzamiento

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Qué es HaxBall

Haxball es un videojuego multijugador basado en navegador web, donde equipos de diferente cantidad de jugadores compiten en una cancha 2D para anotar goles golpeando una pelota virtual con sus pequeños avatares.

Puedes jugar HaxBall haciendo [click aquí](https://www.haxball.com).

## Descripción del problema

Se tendrán dos equipos con un jugador cada uno, es decir, habrán dos agentes. Se tiene el objetivo de entrenar estos agentes para que jueguen eficazmente el juego Haxball.

## Solución propuesta

Exploraremos el uso del Aprendizaje por Refuerzo como una estrategia para entrenar agentes que juegen Haxball. El Aprendizaje por Refuerzo destaca como la mejor opción para mejorar el rendimiento del agente en Haxball por su adaptabilidad en el entorno, su capacidad para explorar nuevas estrategias, la autonomía que ofrece al agente para aprender sin intervención humana constante, y su habilidad para optimizar las recompensas a lo largo del juego.

Este proyecto utilizará una versión adaptada llamada [HaxballGym](https://github.com/HaxballGym/HaxballGym), obtenida de GitHub. Esta versión está diseñada para permitir el entrenamiento de un agente de Aprendizaje por Refuerzo en el juego HaxBall. Se realizarán ajustes en el código para facilitar el entrenamiento del agente y su interacción con el entorno del juego. Además, HaxBallGym es un paquete de Python que se puede utilizar para tratar el juego HaxBall como si fuera un entorno de estilo [OpenAI Gym](https://gym.openai.com) para proyectos de Aprendizaje por Refuerzo. En esencia, haxBallGym actúa como un puente entre el juego Haxball y los algoritmos de Aprendizaje por Refuerzo, para facilitar la experimentación y el desarrollo de agentes de IA capaces de jugar este juego de forma autónoma.

## Información importante
### Descripción del Ambiente

- **Tipo:** Dinámico.
- **Discreto/Continuo:** Continuo.
- **Determinista/Estocástico:** Determinista.
- **Representación concreta del estado del juego, representada por una estructura de datos:** Se tienen las `observaciones`, que incluyen información sobre la posición de los jugadores, la pelota, la velocidad de los jugadores y la pelota, goles marcados, el tiempo restante, etc.
- **Acciones:** Las acciones pueden ser discretas y pueden incluir mover al jugador en diferentes direcciones (arriba, abajo, izquierda, derecha), así como acciones relacionadas con patear la pelota, defender, etc.
- **Recompensas:** Las recompensas pueden estar relacionadas con el éxito en marcar un gol, evitar que el oponente marque un gol, movimientos exitosos, etc.

## Representación de las Acciones

Podría ser una estructura de datos que represente la acción que el agente elige tomar en un momento dado, como un número que represente una dirección de movimiento o una acción específica del juego.

## Requisitos

- Python >= 3.10

## Installation

Instale la biblioteca mediante pip:

```bash
pip install haxballgym
```

Listo!. Se puede ejecutar `example.py` para ver si la instalación se ha realizado correctamente. El script asume que tienes una carpeta de grabaciones desde donde ejecutas el script.

## Grabaciones

Para ver las grabaciones, se puede ir a [HaxBall clone](https://wazarr94.github.io/) y cargar el archivo de grabacion.