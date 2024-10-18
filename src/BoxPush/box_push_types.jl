"""
    BoxPushState
Type to represent the state of the BoxPushPOMDP.

- `small_box_pos::Vector{Int}`: Vector of grid positions of the small boxes. 
- `large_box_pos::Vector{Int}`: Vector of grid positions of the large boxes. (leftmost position)
- `agent_pos::Tuple{Int, Int}`: Grid position of the agents. Only considering two agents.
- `agent_orientation::Tuple{Int, Int}`: Tuple representing the agents' orientations.
"""
struct BoxPushState
    small_box_pos::Vector{Int}
    large_box_pos::Vector{Int}
    agent_pos::Tuple{Int, Int}
    agent_orientation::Tuple{Int, Int}
end

function BoxPushState(
    small_box_pos::Vector{Int},
    large_box_pos::Vector{Int},
    agent_pos::Tuple{Int, Int},
    agent_orientation::Tuple{Symbol, Symbol}
)
    agent_orientation_1 = AGENT_ORIENTATIONS[agent_orientation[1]]
    agent_orientation_2 = AGENT_ORIENTATIONS[agent_orientation[2]]
    return BoxPushState(small_box_pos, large_box_pos, agent_pos, 
        (agent_orientation_1, agent_orientation_2))
end

"""
    BoxPushPOMDP
    
The original implementation is from  "Improved Memory-Bounded Dynamic Programming for 
Decentralized POMDPs" by S. Seuken and S. Zilberstein. This type is used to represent the 
BoxPushPOMDP.

Fields:
- `mg::MetaDiGraph`: map of the BoxPushPOMDP.
- `map_option::Int`: Map option. See `box_push_constants.jl` for options.
- `discount_factor::Float64`: Discount factor. 
- `transition_prob::Float64`: Transition probability.
- `states::Vector{BoxPushState}`: Vector of possible states.
- `state_index::Dict{BoxPushState, Int}`: Dict mapping states to their index in the `states` vector.
- `box_goal_states::Dict{Symbol, BoxPushState}`: Dict mapping goal state names to the goal states.
- `small_box_goal_reward::Float64`: Reward for a small box reaching the goal.
- `large_box_goal_reward::Float64`: Reward for a large box reaching the goal.
- `step_penalty::Float64`: Penalty for each step taken (per agent).
- `wall_penalty::Float64`: Penalty for hitting a wall.
- `observation_agent::Int`: Agent used to generate observations. 0 == joint observation.
"""
struct BoxPushPOMDP <: POMDP{BoxPushState, Vector{Int}, Vector{Int}}
    mg::MetaDiGraph
    map_option::Int
    num_agents::Int
    discount_factor::Float64
    transition_prob::Float64
    states::Vector{BoxPushState}
    state_index::Dict{BoxPushState, Int}
    box_goal_states::Dict{Symbol, BoxPushState}
    small_box_goal_reward::Float64
    large_box_goal_reward::Float64
    step_penalty::Float64
    wall_penalty::Float64
    observation_agent::Int
    observation_prob::Float64
end

"""
    BoxPushPOMDP
Original implementation in "Improved Memory-Bounded Dynamic Programming for Decentralized 
POMDPs" by S. Seuken and S. Zilberstein.

Keywords:
- `map_option::Int`: Map option. See `box_push_constants.jl` for options. Default = 1.
- `discount_factor::Float64`: Discount factor. Default = 0.9.
- `transition_prob::Float64`: Transition probability. Default = 0.9.
- `observation_agent::Int`: Agent used to generate observations. 0 == joint observation.
    Default = 0.
- `small_box_goal_reward::Float64`: Reward for a small box reaching the goal. Default = 10.0.
- `large_box_goal_reward::Float64`: Reward for a large box reaching the goal. Default = 100.0.
- `step_penalty::Float64`: Penalty for each step taken (per agent). Default = -0.1.
- `wall_penalty::Float64`: Penalty for hitting a wall. Default = -5.0.
"""
function BoxPushPOMDP(;
    num_agents::Int = 2,
    map_option::Int = 1,
    discount_factor::Float64=0.9,
    transition_prob::Float64=0.9,
    observation_agent::Int = 0,
    small_box_goal_reward::Float64 = 10.0,
    large_box_goal_reward::Float64 = 100.0,
    step_penalty::Float64 = -0.1,
    wall_penalty::Float64 = -5.0,
    observation_prob::Float64 = 1.0
)
    if !(0 <= discount_factor <= 1)
        throw(ArgumentError("Invalid discount factor. Must be between 0 and 1."))
    end
    if !(0 < transition_prob <= 1)
        throw(ArgumentError("Invalid transition probability. Must be between 0 and 1."))
    end
    if !(map_option in keys(MAP_OPTIONS))
        throw(ArgumentError("Invalid map option."))
    end
    if !(observation_agent in 0:2)
        throw(ArgumentError("Invalid observation agent. Must be 0, 1, or 2."))
    end
    if num_agents != 2
        throw(ArgumentError("Invalid number of agents. Must be 2."))
    end
    
    # Create metagraph
    map_str = MAP_OPTIONS[map_option]
    map_str = replace(map_str, r"[ \t]+" => "")
    if !all(c -> c ∈ ('x', 'o', '\n', 'b', 'B', 'g', 'G', '1', '2'), map_str)
        throw(ArgumentError("Invalid charactor in map_str. Only 'x', 'o', 'b', 'B', 'g', 
            'G', '1', '2' and '\\n' are allowed."))
    end
    mg = create_box_push_metagraph_from_map(map_str)
    
    # Currently only supports 2 agents
    num_agents = 2
    
    state_vec, state_index, box_goal_states = get_states(mg, map_option)
    
    return BoxPushPOMDP(mg, map_option, num_agents, discount_factor, transition_prob, state_vec, 
        state_index, box_goal_states, small_box_goal_reward, large_box_goal_reward, 
        step_penalty, wall_penalty, observation_agent, observation_prob)
end

import Base: ==
function ==(s1::BoxPushState, s2::BoxPushState)
    small_box_pos_same = all(s1.small_box_pos .== s2.small_box_pos)
    if !small_box_pos_same return false end
    large_box_pos_same = all(s1.large_box_pos .== s2.large_box_pos)
    if !large_box_pos_same return false end
    agent_pos_same = s1.agent_pos == s2.agent_pos
    if !agent_pos_same return false end
    agent_orientation_same = s1.agent_orientation == s2.agent_orientation
    if !agent_orientation_same return false end
    return true
end

import Base: hash
function Base.hash(s::BoxPushState, h::UInt)
    h = hash(s.small_box_pos, h)
    h = hash(s.large_box_pos, h)
    h = hash(s.agent_pos, h)
    h = hash(s.agent_orientation, h)
    return h
end

POMDPs.isterminal(::BoxPushPOMDP, ::BoxPushState) = false
POMDPs.discount(pomdp::BoxPushPOMDP) = pomdp.discount_factor

"""
    get_states(mg::MetaDiGraph, map_option::Int)

Returns a tuple of the state vector, state index, and box goal states. The state vector is 
constructed based on the map option. For map options, see `box_push_constants.jl`.
"""
function get_states(mg::MetaDiGraph, map_option::Int)
    state_index = Dict{BoxPushState, Int}()
    box_goal_states = Dict{Symbol, BoxPushState}()
    orientations = [:north, :east, :south, :west]
    small_box_pos = get_prop(mg, :small_box_pos)
    large_box_pos = get_prop(mg, :large_box_pos)
    if map_option == 1
        agent_positions = [9, 10, 11, 12]
        state_vec = Vector{BoxPushState}(undef, 99)
        cnt = 1
        # Boxes at original positions
        for agent_1_pos in agent_positions
            for agent_2_pos in agent_positions
                if agent_2_pos <= agent_1_pos
                    # Agent 2 can't get to the left of agent 1 and can't be in the same position
                    continue
                end
                for agent_1_orientation in orientations
                    for agent_2_orientation in orientations
                        state_vec[cnt] = BoxPushState(
                            small_box_pos,
                            large_box_pos,
                            (agent_1_pos, agent_2_pos),
                            (agent_1_orientation, agent_2_orientation)
                        )
                        state_index[state_vec[cnt]] = cnt
                        cnt += 1
                    end
                end
            end
        end
        
        # Single small box at goal. For the expected reward, it only matters that a single
        # small box reached the goal.
        small_1_at_goal = BoxPushState(
            [1, small_box_pos[2]],
            large_box_pos,
            (5, 12),
            (1, 1)
        )
        state_vec[cnt] = small_1_at_goal
        state_index[small_1_at_goal] = cnt
        box_goal_states[:small_1_at_goal] = small_1_at_goal
        cnt += 1
        
        # Both small boxes at goals
        small_2_at_goal = BoxPushState(
            [1, 4],
            large_box_pos,
            (5, 8),
            (1, 1)
        )
        state_vec[cnt] = small_2_at_goal
        state_index[small_2_at_goal] = cnt
        box_goal_states[:small_2_at_goal] = small_2_at_goal
        cnt += 1
        
        # Large box at goal
        large_1_at_goal = BoxPushState(
            small_box_pos,
            [2],
            (6, 7),
            (1, 1)
        )
        state_vec[cnt] = large_1_at_goal
        state_index[large_1_at_goal] = cnt
        box_goal_states[:large_1_at_goal] = large_1_at_goal
        
        @assert cnt == 99 "Invalid number of states"
    elseif map_option == 2
        state_vec = Vector{BoxPushState}(undef, 93188)
        cnt = 1
        small_1_pos = [7, 8, 13, 14]
        small_2_pos = [11, 12, 17, 18]
        large_pos = [9, 15]
        agent_pos = collect(7:24)
        for small_1 in small_1_pos
            for small_2 in small_2_pos
                for large in large_pos
                    for agent_1 in agent_pos
                        if agent_1 == small_1 || agent_1 == small_2 ||
                            agent_1 == large || agent_1 == large + 1
                            continue
                        end
                        for agent_2 in agent_pos
                            if agent_1 == agent_2 ||
                                agent_2 == small_1 || agent_2 == small_2 ||
                                agent_2 == large || agent_2 == large + 1
                                continue
                            end
                            for a1_orient in orientations
                                for a2_orient in orientations
                                    state_vec[cnt] = BoxPushState(
                                        [small_1, small_2],
                                        [large],
                                        (agent_1, agent_2),
                                        (a1_orient, a2_orient)
                                    )
                                    state_index[state_vec[cnt]] = cnt
                                    cnt += 1
                                end
                            end
                        end
                    end
                end
            end
        end
        @assert cnt == 93184+1 "Invalid number of states: $cnt"
        
        # Single small box at goal
        small_1_at_goal = BoxPushState(
            [1, small_box_pos[2]],
            large_box_pos,
            (7, 23),
            (1, 1)
        )
        
        state_vec[cnt] = small_1_at_goal
        state_index[small_1_at_goal] = cnt
        box_goal_states[:small_1_at_goal] = small_1_at_goal
        cnt += 1
        
        # Single or both small boxes at top but not a goal(s)
        small_at_top = BoxPushState(
            [2, small_box_pos[2]],
            large_box_pos,
            (8, 23),
            (1, 1)
        )
        state_vec[cnt] = small_at_top
        state_index[small_at_top] = cnt
        box_goal_states[:small_at_top] = small_at_top
        cnt += 1
        
        # 2 Small boxes at goal
        small_2_at_goal = BoxPushState(
            [1, 6],
            large_box_pos,
            (7, 12),
            (1, 1)
        )
        state_vec[cnt] = small_2_at_goal
        state_index[small_2_at_goal] = cnt
        box_goal_states[:small_2_at_goal] = small_2_at_goal
        cnt += 1
        
        # Single large box at goal
        large_1_at_goal = BoxPushState(
            small_box_pos,
            [3],
            (9, 10),
            (1, 1)
        )
        state_vec[cnt] = large_1_at_goal
        state_index[large_1_at_goal] = cnt
        box_goal_states[:large_1_at_goal] = large_1_at_goal
        
        @assert cnt == 93188 "Invalid number of states: $cnt"
    else
        throw(ArgumentError("Invalid map option"))
    end
    return state_vec, state_index, box_goal_states
end

"""
    create_box_push_metagraph_from_map(map_str::String)

Returns a `MetaDiGraph` representing the map. 
- 'x' for walls
- 'o' for open space
- 'b' for small boxes
- 'B' for large boxes
- 'g' for small box goal
- 'G' for large box goal
- '1' for robot 1 initial position
- '2' for robot 2 initial position

Properties of the graph:
- `:nrows`: number of rows in the map
- `:ncols`: number of columns in the map
- `:num_grid_pos`: number of open spaces in the map
- `:node_mapping`: dictionary mapping (i, j) position in the map to node number
- `:node_pos_mapping`: dictionary mapping node number to (i, j) position in the map
- `:small_box_pos`: starting small box positions
- `:large_box_pos`: starting large box positions
- `:small_box_goals`: small box goal positions
- `:large_box_goals`: large box goal positions
- `:agent_start_pos`: tuple of agent start positions

Properties of the edges:
- `:action`: action associated with the edge (e.g. :north, :south, :east, :west)

Example map_str (the one in the original paper)
  gGGg\n
  bBBb\n
  1oo2\n
"""
function create_box_push_metagraph_from_map(map_str::String)
    map_str = replace(map_str, r"[ \t]+" => "")
    moveable_chars = ['o', 'b', 'B', 'g', 'G', '1', '2']
    
    lines = split(map_str, '\n')
    if lines[end] == ""
        pop!(lines)
    end

    @assert all(length(line) == length(lines[1]) for line in lines) "Map is not rectangular"

    nrows, ncols = length(lines), length(lines[1])
    num_o = count(c -> c == 'o', map_str)
    num_b = count(c -> c == 'b', map_str)
    num_B = count(c -> c == 'B', map_str)
    num_g = count(c -> c == 'g', map_str)
    num_G = count(c -> c == 'G', map_str)
    num_1 = count(c -> c == '1', map_str)
    num_2 = count(c -> c == '2', map_str)
    
    num_grid_pos = num_o + num_b + num_B + num_g + num_G + num_1 + num_2
    g = SimpleDiGraph(num_grid_pos)

    @assert num_B % 2 == 0 "Large boxes must take up 2 spaces"
    @assert num_G % 2 == 0 "Large box goals must take up 2 spaces"
    
    node_mapping = Dict{Tuple{Int, Int}, Int}()
    node_pos_mapping = Dict{Int, Tuple{Int, Int}}()
    small_box_goals = zeros(Int, num_g)
    large_box_goals = zeros(Int, Int(num_G / 2))
    small_box_pos = zeros(Int, num_b)
    large_box_pos = zeros(Int, Int(num_B / 2))
    node_counter = 0
    small_box_counter = 0
    large_box_counter = 0
    small_box_goal_counter = 0
    large_box_goal_counter = 0
    agent_pos_1 = 0
    agent_pos_2 = 0

    flag = false
    # Map each open area to a unique node number in the graph
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char in moveable_chars
                node_counter += 1
                node_mapping[(i, j)] = node_counter
                node_pos_mapping[node_counter] = (i, j)
                
                if char == 'g'
                    small_box_goal_counter += 1
                    small_box_goals[small_box_goal_counter] = node_counter
                elseif j != length(line) && char == 'G' && line[j + 1] == 'G'
                    if !((node_counter - 1) in large_box_goals)
                        large_box_goal_counter += 1
                        large_box_goals[large_box_goal_counter] = node_counter
                    end
                elseif char == 'b'
                    small_box_counter += 1
                    small_box_pos[small_box_counter] = node_counter
                elseif j != length(line) && char == 'B' && line[j + 1] == 'B'
                    if !((node_counter - 1) in large_box_pos)
                        large_box_counter += 1
                        large_box_pos[large_box_counter] = node_counter
                    end
                elseif char == '1'
                    agent_pos_1 = node_counter
                elseif char == '2'
                    agent_pos_2 = node_counter
                end 
            end
        end
    end

    if agent_pos_1 == 0 || agent_pos_2 == 0
        throw(ArgumentError("Invalid map. Must have two agents (as '1' and '2' in the map string)"))
    end
    if length(large_box_goals) != num_G / 2
        throw(ArgumentError("Invalid map. Large box goals are not correctly specified."))
    end
    if length(large_box_pos) != num_B / 2
        throw(ArgumentError("Invalid map. Large box positions are not correctly specified."))
    end
    
    # Create MetaGraph
    mg = MetaDiGraph(g)

    set_prop!(mg, :nrows, nrows)
    set_prop!(mg, :ncols, ncols)
    set_prop!(mg, :num_grid_pos, num_grid_pos)
    set_prop!(mg, :node_mapping, node_mapping)
    set_prop!(mg, :node_pos_mapping, node_pos_mapping)
    set_prop!(mg, :small_box_goals, small_box_goals)
    set_prop!(mg, :large_box_goals, large_box_goals)
    set_prop!(mg, :small_box_pos, small_box_pos)
    set_prop!(mg, :large_box_pos, large_box_pos)
    set_prop!(mg, :agent_start_pos, (agent_pos_1, agent_pos_2))

    # Add edges based on possible moves and set action properties
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char in moveable_chars
                current_node = node_mapping[(i, j)]
                
                # North
                if i > 1 && lines[i - 1][j] in moveable_chars
                    north_node = node_mapping[(i - 1, j)]
                    add_edge!(mg, current_node, north_node)
                    set_prop!(mg, current_node, north_node, :action, :north)
                end
                # South
                if i < nrows && lines[i + 1][j] in moveable_chars
                    south_node = node_mapping[(i + 1, j)]
                    add_edge!(mg, current_node, south_node)
                    set_prop!(mg, current_node, south_node, :action, :south)
                end
                # East
                if j < ncols && lines[i][j + 1] in moveable_chars
                    east_node = node_mapping[(i, j + 1)]
                    add_edge!(mg, current_node, east_node)
                    set_prop!(mg, current_node, east_node, :action, :east)
                end
                # West
                if j > 1 && lines[i][j - 1] in moveable_chars
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
    map_str_from_metagraph(pomdp::BoxPushPOMDP)

Returns a string representing the map. 'x' for walls, 'o' for open space. Uses the
`node_mapping` property of the metagraph to determine which nodes are open spaces.
"""
function map_str_from_metagraph(pomdp::BoxPushPOMDP)
    nrows = get_prop(pomdp.mg, :nrows)
    ncols = get_prop(pomdp.mg, :ncols)
    node_mapping = get_prop(pomdp.mg, :node_mapping)
    lines = Vector{String}(undef, nrows)
    for i in 1:nrows
        line = Vector{Char}(undef, ncols)
        for j in 1:ncols
            if (i, j) in keys(node_mapping)
                node = node_mapping[(i, j)]
                if node == get_prop(pomdp.mg, :agent_start_pos)[1]
                    line[j] = '1'
                elseif node == get_prop(pomdp.mg, :agent_start_pos)[2]
                    line[j] = '2'
                elseif node in get_prop(pomdp.mg, :small_box_pos)
                    line[j] = 'b'
                elseif node in get_prop(pomdp.mg, :small_box_goals)
                    line[j] = 'g'
                elseif node in get_prop(pomdp.mg, :large_box_pos)
                    line[j] = 'B'
                elseif node in get_prop(pomdp.mg, :large_box_goals)
                    line[j] = 'G'
                elseif node-1 in get_prop(pomdp.mg, :large_box_pos)
                    line[j] = 'B'
                elseif node-1 in get_prop(pomdp.mg, :large_box_goals)
                    line[j] = 'G'
                else
                    line[j] = 'o'
                end
            else
                line[j] = 'x'
            end
        end
        lines[i] = String(line)
    end
    return join(lines, '\n')
end

function Base.show(io::IO, pomdp::BoxPushPOMDP)
    println(io, "BoxPushPOMDP")
    for name in fieldnames(typeof(pomdp))
        if name == :mg
            print(io, "\t", name, ": ")
            print(io, typeof(getfield(pomdp, name)), ", $(nv(getfield(pomdp, name))) nodes, ",
                "$(ne(getfield(pomdp, name))) edges\n")
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

"""
    map_string_to_state(pomdp::BoxPushPOMDP, map_str::String, orientation::Any)
    map_string_to_state(mg::MetaGraph, map_str::String, orientation::Vector{K}) where K
    map_string_to_state(mg::MetaGraph, map_str::String, orientation::Tuple{Symbol, Symbol})
    map_string_to_state(mg::MetaGraph, map_str::String, orientation::Tuple{Int, Int}=(1, 1))

Convert a string representation of a BoxPush map to a BoxPushState.

# Arguments
- `pomdp::BoxPushPOMDP`: The BoxPushPOMDP instance.
- `map_str::String`: A string representation of the map. Use 'x' for walls, 'o' for open spaces, 
  'b' for small boxes, 'B' for large boxes (two adjacent 'B's), 'g' for small box goals, 
  'G' for large box goals (two adjacent 'G's), '1' for agent 1, and '2' for agent 2.
- `orientation::Union{Vector{K}, Tuple{Symbol, Symbol}, Tuple{Int, Int}}`: The orientation of the agents. 
  Can be a vector of two elements, a tuple of two symbols (e.g., (:north, :south)), or a tuple of two integers. 
  Default is (1, 1).

# Returns
- `BoxPushState`: The state corresponding to the given map string and orientation.

# Throws
- `ArgumentError`: If the map string contains invalid characters or if the map structure is invalid.
- `AssertionError`: If the map is not rectangular or if the number of agents, boxes, or goals is incorrect.

# Notes
- The function removes all whitespace from the input string before processing.
- Large boxes and large box goals must occupy two adjacent spaces in the map string.
- The top row of the map is considered inaccessible to the agents.
"""
function map_string_to_state(pomdp::BoxPushPOMDP, map_str::String, orientation::Any)
    return map_string_to_state(pomdp.mg, map_str, orientation)
end
function map_string_to_state(mg::MetaDiGraph, map_str::String, orientation::Vector{K}) where K
    @assert length(orientation) == 2 "Orientation must be length of 2"
    orient_tuple = Tuple(orientation)
    return map_string_to_state(mg, map_str, orient_tuple)
end
function map_string_to_state(mg::MetaDiGraph, map_str::String, orientation::Tuple{Symbol, Symbol})
    orient_int = (AGENT_ORIENTATIONS[orientation[1]], AGENT_ORIENTATIONS[orientation[2]])
    return map_string_to_state(mg, map_str, orient_int)
end
function map_string_to_state(mg::MetaDiGraph, map_str::String, orientation::Tuple{Int, Int})
    map_str = replace(map_str, r"[ \t]+" => "")
    if !all(c -> c ∈ ('x', 'o', '\n', 'b', 'B', 'g', 'G', '1', '2'), map_str)
        throw(ArgumentError("Invalid charactor in map_str. Only 'x', 'o', 'b', 'B', 'g', 
            'G', '1', '2' and '\\n' are allowed."))
    end
    
    lines = split(map_str, '\n')
    if lines[end] == ""
        pop!(lines)
    end

    @assert all(length(line) == length(lines[1]) for line in lines) "Map is not rectangular"
    
    num_b = count(c -> c == 'b', map_str)
    num_B = count(c -> c == 'B', map_str)
    num_G = count(c -> c == 'G', map_str)
    num_1 = count(c -> c == '1', map_str)
    num_2 = count(c -> c == '2', map_str)


    @assert num_1 == 1 "Map must have one agent (as '1' in the map string)"
    @assert num_2 == 1 "Map must have one agent (as '2' in the map string)"
    @assert num_B % 2 == 0 "Large boxes must take up 2 spaces"
    @assert num_G % 2 == 0 "Large box goals must take up 2 spaces"
    
    
    agent_pos_1, agent_pos_2 = (0, 0), (0, 0)
    small_box_pos = fill((0, 0), num_b)
    large_box_pos = fill((0, 0), Int(num_B / 2))
    
    small_box_counter, large_box_counter = 1, 1
    
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char == 'b'
                small_box_pos[small_box_counter] = (i, j)
                small_box_counter += 1
            elseif j != length(line) && char == 'B' && line[j + 1] == 'B'
                if !((i, j - 1) in large_box_pos)
                    large_box_pos[large_box_counter] = (i, j)
                    large_box_counter += 1
                end
            elseif char == '1'
                agent_pos_1 = (i, j)
            elseif char == '2'
                agent_pos_2 = (i, j)
            end 
        end
    end
    
    grid_to_node_map = get_prop(mg, :node_mapping)
    
    small_box_pos = [grid_to_node_map[coord] for coord in small_box_pos]
    large_box_pos = [grid_to_node_map[coord] for coord in large_box_pos]
    agent_pos_1 = grid_to_node_map[agent_pos_1]
    agent_pos_2 = grid_to_node_map[agent_pos_2]
    
    return BoxPushState(small_box_pos, large_box_pos, (agent_pos_1, agent_pos_2), orientation)
end

function num_at_goal_top(pomdp::BoxPushPOMDP, box_pos::Vector{Int}, goal_pos::Vector{Int})
    ncols = get_prop(pomdp.mg, :ncols)
    num_at_goal = 0
    num_at_top = 0
    for box in box_pos
        if box in goal_pos
            num_at_goal += 1
        elseif ceil(Int, box / ncols) == 1
            num_at_top += 1
        end
    end
    return num_at_goal, num_at_top
end

# Function to get the position in a given direction
function move_direction(pomdp::BoxPushPOMDP, pos::Int, orientation::Int, robot::Bool=true)
    # Get the neighboring positions from the graph
    for neighbor in neighbors(pomdp.mg, pos)
        edge_orientation = AGENT_ORIENTATIONS[get_prop(pomdp.mg, pos, neighbor, :action)]
        if edge_orientation == orientation
            if robot
                # Robot can't move to top row
                (i, _) = get_prop(pomdp.mg, :node_pos_mapping)[neighbor]
                if i == 1
                    continue
                end
            end
            return neighbor
        end
    end
    # If no valid movement, return the original position
    return pos
end
