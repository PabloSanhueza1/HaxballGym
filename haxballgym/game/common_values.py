COLLISION_FLAG_NONE = 0
COLLISION_FLAG_BALL = 1
COLLISION_FLAG_RED = 2
COLLISION_FLAG_BLUE = 4
COLLISION_FLAG_REDKO = 8
COLLISION_FLAG_BLUEKO = 16
COLLISION_FLAG_WALL = 32
COLLISION_FLAG_ALL = 63
COLLISION_FLAG_KICK = 64
COLLISION_FLAG_SCORE = 128
COLLISION_FLAG_C0 = 268435456
COLLISION_FLAG_C1 = 536870912
COLLISION_FLAG_C2 = 1073741824
COLLISION_FLAG_C3 = -2147483648

DICT_COLLISION = {
    '': COLLISION_FLAG_NONE,
    'ball': COLLISION_FLAG_BALL,
    'red': COLLISION_FLAG_RED,
    'blue': COLLISION_FLAG_BLUE,
    'redKO': COLLISION_FLAG_REDKO,
    'blueKO': COLLISION_FLAG_BLUEKO,
    'wall': COLLISION_FLAG_WALL,
    'all': COLLISION_FLAG_ALL,
    'kick': COLLISION_FLAG_KICK,
    'score': COLLISION_FLAG_SCORE,
    'c0': COLLISION_FLAG_C0,
    'c1': COLLISION_FLAG_C1,
    'c2': COLLISION_FLAG_C2,
    'c3': COLLISION_FLAG_C3
}

DICT_KEYS = {
    'bCoef': 'bouncing_coefficient',
    'cGroup': 'collision_group',
    'cMask': 'collision_mask',
    'radius': 'radius',
    'invMass': 'inverse_mass',
    'damping': 'damping',
    'curve': 'curve',
    'curveF': '_curveF',
    'bias': 'bias'
}

TEAM_SPECTATOR_ID = 0
TEAM_RED_ID = 1
TEAM_BLUE_ID = 2

GAME_STATE_KICKOFF = 0
GAME_STATE_PLAYING = 1
GAME_STATE_GOAL = 2
GAME_STATE_END = 3

ACTION_BIN_UP = 0
ACTION_BIN_RIGHT = 1
ACTION_BIN_KICK = 2

INPUT_UP = 4
INPUT_LEFT = 2
INPUT_DOWN = 1
INPUT_RIGHT = 8
INPUT_SHOOT = 16
