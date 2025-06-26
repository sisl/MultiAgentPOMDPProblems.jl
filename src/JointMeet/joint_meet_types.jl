struct JointMeetState
    r_positions::Tuple
end

import Base: ==
function ==(s1::JointMeetState, s2::JointMeetState)
    return s1.r_positions == s2.r_positions
end

import Base: hash
function hash(s::JointMeetState, h::UInt)
    return hash(s.r_positions, h)
end

"""
    JointMeetPOMDP <: POMDP{JointMeetState, Int, Int}

POMDP type for the Joint Meet POMDP.

# Fields
- `mg::MetaDiGraph`: metagraph representing the map
- `dist_matrix::Matrix{Float64}`: distance matrix for the metagraph
- `meet_reward::Float64`: reward for the agents meeting
- `step_penalty::Float64`: reward for each movement action (negative = penalty)
- `wall_penalty::Float64`: penalty for hitting a wall
- `discount_factor::Float64`: discount factor
- `observation_option::Int`: observation option
- `transition_prob::Float64`: probability of transitioning to the next state
- `transition_alternatives::Symbol`: alternate transitions to consider when `transition_prob` is not 1.0
"""
struct JointMeetPOMDP <: POMDP{JointMeetState, Vector{Int}, Vector{Int}}
    mg::MetaDiGraph
    dist_matrix::Matrix{Float64}
    num_agents::Int
    meet_reward::Float64
    meet_reward_locations::Vector{Int}
    step_penalty::Float64
    wall_penalty::Float64
    discount_factor::Float64
    observation_option::Symbol
    observation_sigma::Float64
    observation_agent::Int
    transition_prob::Float64
    transition_alternatives::Symbol
    init_state::Union{JointMeetState, Nothing}
end

"""
    JointMeetPOMDP(; kwargs...)

# Keywords
- `map_str::String`: String representing the map, 'x' for walls, 'o' for open space.
    Default is the standard map from the original paper.\n
    Default: \"\"\"
    xxxxxoooxx\n
    xxxxxoooxx\n
    xxxxxoooxx\n
    oooooooooo\n
    oooooooooo\"\"\"
- `num_agents::Int`: Number of agents in the environment, default = 2
- `meet_reward::Float64`: Reward for being in the same spot as all other bots, default = +1.0
- `meet_reward_locations::Vector{Int}`: Locations where the meet reward is given, default = [] (all nodes)
- `step_penalty::Float64`: Reward for each movement action, default = 0.0
- `wall_penalty::Float64`: Penalty for hitting a wall, default = 0.0
- `discount_factor::Float64`: Discount factor, default = 0.9
- `observation_option::Symbol`: Observation option, default = :full
    - `:full`: return the full, joint observation
    - `:left_and_same`: number of moves for agent the left wall and number of agents in the same position
    - `:right_and_same`: number of moves for agent to the right wall and number of agents in the same position
    - `:boundaries_lr`: if agent is at leftmost or rightmost boundary
    - `:boundaries_ud`: if agent is at topmost or bottommost boundary
    - `:boundaries_both`: observation of left, right, top, and bottom boundaries (9 per agent)
- `observation_sigma::Float64`: Standard deviation of the observation noise, default = 0.0
- `observation_agent::Int`: Agent to observe, default = 1 (0 = all agents)
- `transition_prob::Float64`: Probability of transitioning to the next state, default = 0.6
- `transition_alternatives::Symbol`: Alternate transitions to consider when `transition_prob` is not 1.0, default = :all
    - `:all`: all transitions are alternatives except stay (if stay, no alternatives)
    - `:not_opposite`: all transitions are alternatives except stay and the opposite direction
- `init_state::Union{JointMeetState, Nothing}`: Initial state, default = nothing
"""
function JointMeetPOMDP(;
    map_str::String = """xxxxxoooxx
                         xxxxxoooxx
                         xxxxxoooxx
                         oooooooooo
                         oooooooooo
                         """,
    num_agents::Int = 2,
    meet_reward::Float64 = 1.0,
    meet_reward_locations::Vector{Int} = Vector{Int}(),
    step_penalty::Float64 = 0.0,
    wall_penalty::Float64 = 0.0,
    discount_factor::Float64 = 0.9,
    observation_option::Symbol = :full,
    observation_sigma::Float64 = 0.0,
    observation_agent::Int = 0,
    transition_prob::Float64 = 0.6,
    transition_alternatives::Symbol = :all,
    init_state::Union{JointMeetState, Nothing} = nothing
)

    # Remove whitespace from map_str but leaeve line breaks and check for valid charactors
    map_str = replace(map_str, r"[ \t]+" => "")
    if !all(c -> c ∈ ('x', 'o', '\n'), map_str)
        throw(ArgumentError("Invalid charactor in map_str. Only 'x', 'o', and '\\n' are allowed."))
    end
    
    if !(observation_option in (:full, :left_and_same, :right_and_same, :boundaries_lr, :boundaries_ud, :boundaries_both))
        throw(ArgumentError("Invalid observation option: $(observation_option)"))
    end
    
    if !(observation_agent in 0:num_agents)
        throw(ArgumentError("Invalid observation agent: $(observation_agent)"))
    end
    
    if !(0.0 <= transition_prob <= 1.0)
        throw(ArgumentError("Invalid transition probability: $(transition_prob)"))
    end
    
    if !(0.0 <= discount_factor <= 1.0)
        throw(ArgumentError("Invalid discount factor: $(discount_factor)"))
    end
    
    if !(0.0 <= observation_sigma)
        throw(ArgumentError("Invalid observation sigma: $(observation_sigma)"))
    end
    
    if !(transition_alternatives in (:all, :not_opposite))
        throw(ArgumentError("Invalid transition alternatives: $(transition_alternatives)"))
    end

    # Create metagraph
    mg = create_metagraph_from_map(map_str)

    if !isempty(meet_reward_locations) && !(all(l -> l in 1:get_prop(mg, :num_grid_pos), meet_reward_locations))
        throw(ArgumentError("Invalid meet reward locations: $(meet_reward_locations)"))
    end
    
    # Create distance matrix for the metagraph
    dist_matrix = floyd_warshall_shortest_paths(mg).dists
    
    return JointMeetPOMDP(
        mg, dist_matrix, num_agents, meet_reward, meet_reward_locations, step_penalty, 
        wall_penalty, discount_factor, 
        observation_option, observation_sigma, observation_agent,
        transition_prob, transition_alternatives, init_state
    )
end

"""
    create_metagraph_from_map(map_str::String)

Returns a `MetaDiGraph` representing the map. 'x' for walls, 'o' for open space.

Properties of the graph:
- `:nrows`: number of rows in the map
- `:ncols`: number of columns in the map
- `:num_grid_pos`: number of open spaces in the map
- `:node_mapping`: dictionary mapping (i, j) position in the map to node number
- `:node_pos_mapping`: dictionary mapping node number to (i, j) position in the map

Properties of the edges:
- `:action`: action associated with the edge (e.g. :north, :south, :east, :west)

# Example mat_str for the original TagPOMDP (the one in the original paper)
xxxxxoooxx\n
xxxxxoooxx\n
xxxxxoooxx\n
oooooooooo\n
oooooooooo\n
"""
function create_metagraph_from_map(map_str::String)
    lines = split(map_str, '\n')
    if lines[end] == ""
        pop!(lines)
    end

    @assert all(length(line) == length(lines[1]) for line in lines) "Map is not rectangular"

    nrows, ncols = length(lines), length(lines[1])
    num_o = count(c -> c == 'o', map_str)

    g = SimpleDiGraph(num_o)

    node_mapping = Dict{Tuple{Int, Int}, Int}()
    node_pos_mapping = Dict{Int, Tuple{Int, Int}}()
    node_counter = 1

    # Map each open area to a unique node number in the graph
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char == 'o'
                node_mapping[(i, j)] = node_counter
                node_pos_mapping[node_counter] = (i, j)
                node_counter += 1
            end
        end
    end

    # Create MetaGraph
    mg = MetaDiGraph(g)

    set_prop!(mg, :nrows, nrows)
    set_prop!(mg, :ncols, ncols)
    set_prop!(mg, :num_grid_pos, num_o)
    set_prop!(mg, :node_mapping, node_mapping)
    set_prop!(mg, :node_pos_mapping, node_pos_mapping)

    # Add edges based on possible moves and set action properties
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char == 'o'
                current_node = node_mapping[(i, j)]
                # North
                if i > 1 && lines[i - 1][j] == 'o'
                    north_node = node_mapping[(i - 1, j)]
                    add_edge!(mg, current_node, north_node)
                    set_prop!(mg, current_node, north_node, :action, :north)
                end
                # South
                if i < nrows && lines[i + 1][j] == 'o'
                    south_node = node_mapping[(i + 1, j)]
                    add_edge!(mg, current_node, south_node)
                    set_prop!(mg, current_node, south_node, :action, :south)
                end
                # East
                if j < ncols && lines[i][j + 1] == 'o'
                    east_node = node_mapping[(i, j + 1)]
                    add_edge!(mg, current_node, east_node)
                    set_prop!(mg, current_node, east_node, :action, :east)
                end
                # West
                if j > 1 && lines[i][j - 1] == 'o'
                    west_node = node_mapping[(i, j - 1)]
                    add_edge!(mg, current_node, west_node)
                    set_prop!(mg, current_node, west_node, :action, :west)
                end
            end
        end
    end
    return mg
end

"""
    map_str_from_metagraph(pomdp::JointMeetPOMDP)

Returns a string representing the map. 'x' for walls, 'o' for open space. Uses the
`node_mapping` property of the metagraph to determine which nodes are open spaces.
"""
function map_str_from_metagraph(pomdp::JointMeetPOMDP)
    nrows = get_prop(pomdp.mg, :nrows)
    ncols = get_prop(pomdp.mg, :ncols)
    node_mapping = get_prop(pomdp.mg, :node_mapping)
    lines = Vector{String}(undef, nrows)
    for i in 1:nrows
        line = Vector{Char}(undef, ncols)
        for j in 1:ncols
            if (i, j) in keys(node_mapping)
                line[j] = 'o'
            else
                line[j] = 'x'
            end
        end
        lines[i] = String(line)
    end
    return join(lines, '\n')
end

Base.length(pomdp::JointMeetPOMDP) = get_prop(pomdp.mg, :num_grid_pos) ^ pomdp.num_agents
POMDPs.isterminal(::JointMeetPOMDP, ::JointMeetState) = false
POMDPs.discount(pomdp::JointMeetPOMDP) = pomdp.discount_factor

function Base.show(io::IO, pomdp::JointMeetPOMDP)
    println(io, "JointMeetPOMDP")
    for name in fieldnames(typeof(pomdp))
        if name == :mg
            print(io, "\t", name, ": ")
            print(io, typeof(getfield(pomdp, name)), ", $(nv(getfield(pomdp, name))) nodes, $(ne(getfield(pomdp, name))) edges\n")
        elseif name == :dist_matrix
            d_mat_size = size(getfield(pomdp, name))
            print(io, "\t", name, ": ")
            print(io, typeof(getfield(pomdp, name)), "$d_mat_size\n")
        else
            print(io, "\t", name, ": ", getfield(pomdp, name), "\n")
        end
    end

    # Print the map as a string
    map_str = map_str_from_metagraph(pomdp)
    print(io, "\tmap:\n")
    lines = split(map_str, '\n')
    for line in lines
        print(io, "\t\t", line, "\n")
    end
end
