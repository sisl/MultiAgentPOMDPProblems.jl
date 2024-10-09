const MAP_OPTIONS = Dict(
    1 => """gGGg
            bBBb
            1oo2
    """,
    2 => """goGGog
            oooooo
            obBBbo
            o1oo2o
    """
)

const AGENT_ORIENTATIONS = Dict(
    :north => 1,
    :east => 2,
    :south => 3,
    :west => 4
)

const BOX_PUSH_ACTIONS = Dict(
    :turn_left => 1,
    :turn_right => 2,
    :move_forward => 3,
    :stay => 4
)

const BOX_PUSH_ACTION_NAMES = Dict(
    1 => "Turn Left",
    2 => "Turn Right",
    3 => "Move Forward",
    4 => "Stay"
)

const BOX_PUSH_OBSERVATIONS = Dict(
    :empty => 1,
    :wall => 2,
    :agent => 3,
    :small_box => 4,
    :large_box => 5
)

const BOX_PUSH_OBSERVATION_NAMES = Dict(
    1 => :empty,
    2 => :wall,
    3 => :agent,
    4 => :small_box,
    5 => :large_box
)

const BOX_PUSH_MOD_OBSERVATIONS = Dict(
    :empty => 1,
    :wall => 2,
    :agent => 3,
    :small_box => 4,
    :large_box => 5,
    :no_obs => 6
)

const BOX_PUSH_MOD_OBSERVATION_NAMES = Dict(
    1 => :empty,
    2 => :wall,
    3 => :agent,
    4 => :small_box,
    5 => :large_box,
    6 => :no_obs
)
