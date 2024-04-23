# Entrenamiento de un agente para el juego Haxball mediante aprendizaje por reforzamiento

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Logo](https://www.haxball.com/f2XqsDz4/s/haxball-big-min.png)

## Integrantes:
- Pablo Sanhueza
- Tomás Cárdenas
- Vicente Ríos

## Qué es HaxBall

Haxball es un videojuego multijugador basado en navegador web, donde equipos de diferente cantidad de jugadores compiten en una cancha 2D para anotar goles golpeando una pelota virtual con sus pequeños avatares.

Puedes jugar HaxBall haciendo [click aquí](https://www.haxball.com).

## Descripción del problema

Se tendrán dos equipos con un jugador cada uno, es decir, habrán dos agentes. Se tiene el objetivo de entrenar estos agentes para que jueguen eficazmente el juego Haxball.

## Solución propuesta

Exploraremos el uso del Aprendizaje por Refuerzo como una estrategia para entrenar agentes que juegen Haxball. El Aprendizaje por Refuerzo destaca como la mejor opción para mejorar el rendimiento del agente en Haxball por su adaptabilidad en el entorno, su capacidad para explorar nuevas estrategias, la autonomía que ofrece al agente para aprender sin intervención humana constante, y su habilidad para optimizar las recompensas a lo largo del juego.

Este proyecto utilizará una versión adaptada llamada [HaxballGym](https://github.com/HaxballGym/HaxballGym), obtenida de GitHub. Esta versión está diseñada para permitir el entrenamiento de un agente de Aprendizaje por Refuerzo en el juego HaxBall. Se realizarán ajustes en el código para facilitar el entrenamiento del agente y su interacción con el entorno del juego. Además, HaxBallGym es un paquete de Python que se puede utilizar para tratar el juego HaxBall como si fuera un entorno de estilo [OpenAI Gym](https://gym.openai.com) para proyectos de Aprendizaje por Refuerzo. En esencia, haxBallGym actúa como un puente entre el juego Haxball y los algoritmos de Aprendizaje por Refuerzo, para facilitar la experimentación y el desarrollo de agentes de IA capaces de jugar este juego de forma autónoma.

## Descripción del Ambiente

- **Tipo:** Dinámico.
- **Discreto/Continuo:** Continuo.
- **Determinista/Estocástico:** Determinista.

## Representación concreta del estado del juego, representada por una estructura de datos:

Se tienen las `observaciones`, que incluyen información sobre la posición de los jugadores, la pelota, la velocidad de los jugadores y la pelota, goles marcados, el tiempo restante, etc.

Cabe destacar, que, en este proceso temprano del proyecto, todavía debemos seguir investigando sobre HaxBallGym.

## Recompensas
Se espera que las recomensas estén relacionadas con desempeño del agente. Por ejemplo, en primera instancia, recompensa positiva al golpear el balon, luego, en anotar un gol, o bien, recompensa negativa por permitir que el oponente anote un gol.

## Representación de las Acciones

En el juego Haxball, un agente entrenado con aprendizaje por refuerzo tiene a su disposición: 

### Acciones Continuas:

Control preciso de la dirección y velocidad de movimiento del jugador a través de vectores de velocidad.

### Acciones Discretas:

Patear la pelota eligiendo de un conjunto predefinido de ángulos y fuerzas de pateo.
Acciones defensivas como posicionar y orientar al jugador para interceptar/bloquear al oponente.
Moverse a posiciones estratégicas en el campo para maximizar oportunidades ofensivas/defensivas.

El agente debe aprender a coordinar óptimamente estas acciones continuas para control de movimiento y acciones discretas tácticas como patear, defender y posicionarse, a fin de desarrollar un juego sólido y efectivo en el entorno continuo de Haxball.

## Requisitos para la ejecución

- Python >= 3.10

## Instalación

Instale la biblioteca mediante pip:

```bash
pip install haxballgym
```

Listo!. Se puede ejecutar `example.py` para ver si la instalación se ha realizado correctamente. El script asume que tienes una carpeta de grabaciones desde donde ejecutas el script.

## Grabaciones

Para ver las grabaciones, se puede ir a [HaxBall clone](https://wazarr94.github.io/) y cargar el archivo de grabacion.