"""
    Stochastic Mars Rover problem originally in "Achieveing Goals in Decentralized POMDPs"
    by C. Amato and S. Zilberstein
"""

struct StochasticMarsState
    agent_positions::Tuple
    experimented::Vector{Bool}
end

import Base: ==
function ==(s1::StochasticMarsState, s2::StochasticMarsState)
    same_pos = all(s1.agent_positions .== s2.agent_positions)
    same_exp_vec = all(s1.experimented .== s2.experimented)
    return same_pos && same_exp_vec
end

import Base: hash
function hash(s::StochasticMarsState, h::UInt)
    h = hash(s.agent_positions, h)  
    h = hash(s.experimented, h)     
    return h
end

struct StochasticMarsPOMDP <: POMDP{StochasticMarsState, Vector{Int}, Vector{Int}}
    mg::MetaDiGraph
    dist_matrix::Matrix{Float64}
    num_agents::Int
    transition_prob::Float64
    movement_penalty::Float64
    ruin_site_penalty::Float64
    redundancy_penalty::Float64
    drill_reward::Float64
    sample_reward::Float64
    init_state::Union{StochasticMarsState, Nothing}
    discount_factor::Float64
    observation_agent::Int
end

function StochasticMarsPOMDP(;
    map_str::String = """ds
                         sd
                      """,
    num_agents::Int=2,
    transition_prob::Float64=0.9,
    movement_penalty::Float64=-0.1,
    ruin_site_penalty::Float64=-10.0,
    redundancy_penalty::Float64=-1.0,
    drill_reward::Float64=6.0,
    sample_reward::Float64=2.0,
    init_state::Union{StochasticMarsState, Nothing} = nothing,
    discount_factor::Float64=0.9,
    observation_agent::Int=0
)

    map_str = replace(map_str, r"[ \t]+" => "")
    if !all(c -> c ∈ ('x', 'o', '\n', 's', 'd'), map_str)
        throw(ArgumentError("Invalid charactor in map_str. Only 'x', 'o', 's', 'd' and '\\n' are allowed."))
    end

    if !(0 <= transition_prob <= 1)
        throw(ArgumentError("Invalid transition probability. Must be between 0 and 1."))
    end
    if !(0 <= discount_factor <= 1)
        throw(ArgumentError("Invalid discount factor. Must be between 0 and 1."))
    end
    
    # Create metagraph
    mg = create_mars_metagraph_from_map(map_str)

    dist_matrix = floyd_warshall_shortest_paths(mg).dists
    
    return StochasticMarsPOMDP(
        mg, dist_matrix, num_agents, transition_prob, 
        movement_penalty, ruin_site_penalty, redundancy_penalty, drill_reward, 
        sample_reward, init_state, discount_factor, 
        observation_agent)
end

"""
    create_mars_metagraph_from_map(map_str::String)

Returns a `MetaDiGraph` representing the map. 'x' for walls, 'o' for open space, 's' for
sample sites, 'd' for drill sites.

Properties of the graph:
- `:nrows`: number of rows in the map
- `:ncols`: number of columns in the map
- `:num_grid_pos`: number of open spaces in the map
- `:node_mapping`: dictionary mapping (i, j) position in the map to node number
- `:node_pos_mapping`: dictionary mapping node number to (i, j) position in the map

Properties of the edges:
- `:action`: action associated with the edge (e.g. :north, :south, :east, :west)

Properties of the nodes:
- `:is_sample_site`: whether the node is a sample site
- `:is_drill_site`: whether the node is a drill site

# Example map_str (the one in the original paper)
  ds\n
  sd\n
"""
function create_mars_metagraph_from_map(map_str::String)
    moveable_chars = ['o', 's', 'd']
    
    lines = split(map_str, '\n')
    if lines[end] == ""
        pop!(lines)
    end

    @assert all(length(line) == length(lines[1]) for line in lines) "Map is not rectangular"

    nrows, ncols = length(lines), length(lines[1])
    num_o = count(c -> c == 'o', map_str)
    num_s = count(c -> c == 's', map_str)
    num_d = count(c -> c == 'd', map_str)

    g = SimpleDiGraph(num_o + num_s + num_d)

    node_mapping = Dict{Tuple{Int, Int}, Int}()
    node_pos_mapping = Dict{Int, Tuple{Int, Int}}()
    ds_mapping = Dict{Int, Int}()
    node_counter = 1
    site_counter = 1

    # Map each open area to a unique node number in the graph
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char in moveable_chars
                node_mapping[(i, j)] = node_counter
                node_pos_mapping[node_counter] = (i, j)
                
                if char == 'd' || char == 's'
                    ds_mapping[node_counter] = site_counter
                    site_counter += 1
                end
                
                node_counter += 1
            end
        end
    end

    # Create MetaGraph
    mg = MetaDiGraph(g)

    set_prop!(mg, :nrows, nrows)
    set_prop!(mg, :ncols, ncols)
    set_prop!(mg, :num_grid_pos, num_o + num_s + num_d)
    set_prop!(mg, :num_sample_drill_sites, num_s + num_d)
    set_prop!(mg, :node_mapping, node_mapping)
    set_prop!(mg, :node_pos_mapping, node_pos_mapping)
    set_prop!(mg, :ds_mapping, ds_mapping)

    # Add edges based on possible moves and set action properties
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char in moveable_chars
                current_node = node_mapping[(i, j)]
                set_prop!(mg, current_node, :is_sample_site, char == 's')
                set_prop!(mg, current_node, :is_drill_site, char == 'd')
                
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
    map_str_from_metagraph(pomdp::StochasticMarsPOMDP)

Returns a string representing the map. 'x' for walls, 'o' for open space. Uses the
`node_mapping` property of the metagraph to determine which nodes are open spaces.
"""
function map_str_from_metagraph(pomdp::StochasticMarsPOMDP)
    nrows = get_prop(pomdp.mg, :nrows)
    ncols = get_prop(pomdp.mg, :ncols)
    node_mapping = get_prop(pomdp.mg, :node_mapping)
    lines = Vector{String}(undef, nrows)
    for i in 1:nrows
        line = Vector{Char}(undef, ncols)
        for j in 1:ncols
            if (i, j) in keys(node_mapping)
                if get_prop(pomdp.mg, node_mapping[(i, j)], :is_sample_site)
                    line[j] = 's'
                elseif get_prop(pomdp.mg, node_mapping[(i, j)], :is_drill_site)
                    line[j] = 'd'
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

function Base.show(io::IO, pomdp::StochasticMarsPOMDP)
    println(io, "StochasticMarsPOMDP")
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

function Base.length(pomdp::StochasticMarsPOMDP)
    num_robot_pos = get_prop(pomdp.mg, :num_grid_pos) ^ pomdp.num_agents
    num_rock_status = 2 ^ get_prop(pomdp.mg, :num_sample_drill_sites)
    return num_robot_pos * num_rock_status
end

POMDPs.isterminal(::StochasticMarsPOMDP, ::StochasticMarsState) = false
POMDPs.discount(pomdp::StochasticMarsPOMDP) = pomdp.discount_factor

POMDPs.states(pomdp::StochasticMarsPOMDP) = pomdp

function Base.iterate(pomdp::StochasticMarsPOMDP, ii::Int=1)
    if ii > length(pomdp)
        return nothing
    end
    s = getindex(pomdp, ii)
    return (s, ii + 1)
end

function Base.getindex(pomdp::StochasticMarsPOMDP, si::Int)
    @assert si <= length(pomdp) "Index out of bounds"
    @assert si > 0 "Index out of bounds"

    num_grid_pos = get_prop(pomdp.mg, :num_grid_pos)
    num_sample_drill_sites = get_prop(pomdp.mg, :num_sample_drill_sites)

    # Reconstruct state_vec as in stateindex
    pos_vec = [num_grid_pos for _ in 1:pomdp.num_agents]
    drill_stat_vec = [2 for _ in 1:num_sample_drill_sites]
    state_vec = vcat(pos_vec, drill_stat_vec)

    # Convert linear index back to multidimensional indices
    indices = ind2sub_dims(si, state_vec)

    # Extract agent positions
    agent_positions = tuple(indices[1:pomdp.num_agents]...)

    # Extract experimented statuses
    ex_indices = indices[pomdp.num_agents+1:end]
    experimented = [ex == 2 for ex in ex_indices]

    return StochasticMarsState(agent_positions, experimented)
end


function Base.getindex(pomdp::StochasticMarsPOMDP, si_range::UnitRange{Int})
    return [getindex(pomdp, si) for si in si_range]
end

function Base.firstindex(::StochasticMarsPOMDP)
    return 1
end
function Base.lastindex(pomdp::StochasticMarsPOMDP)
    return length(pomdp)
end

function POMDPs.stateindex(pomdp::StochasticMarsPOMDP, s::StochasticMarsState)
    num_grid_pos = get_prop(pomdp.mg, :num_grid_pos)
    for (ii, r_pos) in enumerate(s.agent_positions)
        @assert r_pos > 0 && r_pos <= num_grid_pos "Robot $ii invalid position"
    end
    num_sample_drill_sites = get_prop(pomdp.mg, :num_sample_drill_sites)
    if length(s.experimented) != num_sample_drill_sites
        throw(ArgumentError("Invalid state: length of experimented must be equal to the number of sample drill sites"))
    end
    
    pos_vec = [num_grid_pos for _ in 1:pomdp.num_agents]
    drill_stat_vec = [2 for _ in 1:num_sample_drill_sites]
    state_vec = vcat(pos_vec, drill_stat_vec)
    
    # Turn s.experimented into 1:2 Vector
    ex_vec = [s.experimented[i] ? 2 : 1 for i in 1:length(s.experimented)]
    si = LinearIndices(Tuple(state_vec))[s.agent_positions..., ex_vec...]
    return si
end

# Uniform over grid positions but experimented is vector of false
function POMDPs.initialstate(pomdp::StochasticMarsPOMDP)
    if !isnothing(pomdp.init_state)
        return Deterministic(pomdp.init_state)
    else
        poss_states = Vector{StochasticMarsState}()
        num_grid_pos = get_prop(pomdp.mg, :num_grid_pos)
        num_poss_state_pos = num_grid_pos ^ pomdp.num_agents
        for i in 1:num_poss_state_pos
            push!(poss_states, pomdp[i])
        end
        probs = normalize(ones(length(poss_states)), 1)
        return SparseCat(poss_states, probs)
    end
end

const MARS_SINGLE_ACTIONS_DICT = Dict(:north => 1, :east => 2, :south => 3, :west => 4, :sample => 5, :drill => 6)
const MARS_SINGLE_ACTION_NAMES = Dict(1 => "North", 2 => "East", 3 => "South", 4 => "West", 5 => "Sample", 6 => "Drill")

function POMDPs.actions(pomdp::StochasticMarsPOMDP)
     acts = [action_from_index(pomdp, ai) for ai in 1:(length(MARS_SINGLE_ACTIONS_DICT)^pomdp.num_agents)]
     return acts
end

function POMDPs.actionindex(pomdp::StochasticMarsPOMDP, a::Tuple{Vararg{Int}})
    return POMDPs.actionindex(pomdp, collect(a))
end
function POMDPs.actionindex(pomdp::StochasticMarsPOMDP, a::Vector{Int})
    for ai in a
        @assert ai >= 1 "Invalid action index"
        @assert ai <= length(MARS_SINGLE_ACTIONS_DICT) "Invalid action index"
    end
    return LinearIndices(Tuple(length(MARS_SINGLE_ACTIONS_DICT) for _ in 1:pomdp.num_agents))[a...]
end

function POMDPs.actionindex(pomdp::StochasticMarsPOMDP, a::Tuple{Vararg{Symbol}})
    for ai in a
        @assert ai in keys(MARS_SINGLE_ACTIONS_DICT) "Invalid action symbol"
    end
    return POMDPs.actionindex(pomdp, [MARS_SINGLE_ACTIONS_DICT[aj] for aj in a])
end 

function action_from_index(pomdp::StochasticMarsPOMDP, ai::Int)
    @assert ai >= 1 "Invalid action index"
    @assert ai <= length(MARS_SINGLE_ACTIONS_DICT)^pomdp.num_agents "Invalid action index"
    action_cart_ind = CartesianIndices(Tuple(length(MARS_SINGLE_ACTIONS_DICT) for _ in 1:pomdp.num_agents))[ai]
    return Vector{Int}(collect(Tuple(action_cart_ind)))
end

function POMDPs.transition(pomdp::StochasticMarsPOMDP, s::StochasticMarsState, a::Vector{Symbol})
    return POMDPs.transition(pomdp, s, [MARS_SINGLE_ACTIONS_DICT[aj] for aj in a])
end
function POMDPs.transition(pomdp::StochasticMarsPOMDP, s::StochasticMarsState, a::Vector{Int})
    exp_vec = deepcopy(s.experimented)
    
    r_pos′ = Vector{Vector{Int}}(undef, pomdp.num_agents)
    prob′ = Vector{Vector{Float64}}(undef, pomdp.num_agents)
    for ii in 1:pomdp.num_agents
        aᵢ = a[ii]
        sᵢ = s.agent_positions[ii]
        
        if aᵢ == MARS_SINGLE_ACTIONS_DICT[:drill] || aᵢ == MARS_SINGLE_ACTIONS_DICT[:sample]
            rock_num = get_prop(pomdp.mg, :ds_mapping)[sᵢ]
            exp_vec[rock_num] = true
            r_pos′[ii] = [sᵢ]
            prob′[ii] = [1.0]
            continue
        else
            new_grid_pos = move_direction(pomdp, sᵢ, aᵢ)
            r_pos′[ii] = [new_grid_pos, sᵢ]
            prob′[ii] = [pomdp.transition_prob, 1.0 - pomdp.transition_prob]
        end
    end

    # If all rocks are experimented, we reset. We can do this here instead of waiting until 
    # the next state because it is deterministic for performing experiments.
    if all(exp_vec)
        return initialstate(pomdp)
    end
    
    new_states = []
    new_probs = []
    for (pos, prob) in zip(Iterators.product(r_pos′...), Iterators.product(prob′...))
        new_state = StochasticMarsState(pos, exp_vec)
        if new_state in new_states
            idx = findfirst(new_state == new_state_i for new_state_i in new_states)
            new_probs[idx] += prod(prob)
            continue
        else
            push!(new_states, new_state)
            push!(new_probs, prod(prob))
        end
    end
    return SparseCat(new_states, normalize(new_probs, 1))
end

function move_direction(pomdp::StochasticMarsPOMDP, v::Int, a::Int)
    if a == MARS_SINGLE_ACTIONS_DICT[:drill] || a == MARS_SINGLE_ACTIONS_DICT[:sample]
        return v
    end
    
    neighs = neighbors(pomdp.mg, v)
    for n_i in neighs
        if MARS_SINGLE_ACTIONS_DICT[get_prop(pomdp.mg, v, n_i, :action)] == a
            return n_i
        end
    end
    return v
end

function POMDPs.observations(pomdp::StochasticMarsPOMDP)
    num_grids = get_prop(pomdp.mg, :num_grid_pos)
    num_single_obs = num_grids * 2  # Each agent's possible observations
    if pomdp.observation_agent == 0
        # Joint observations: all combinations of per-agent observations
        single_obs_range = 1:num_single_obs
        return vec([collect(obs) for obs in Iterators.product(ntuple(_ -> single_obs_range, pomdp.num_agents)...)])
    else
        # Single agent observations
        return [[oi] for oi in 1:num_single_obs]
    end
end

function POMDPs.obsindex(pomdp::StochasticMarsPOMDP, o::Vector{Int})
    num_grids = get_prop(pomdp.mg, :num_grid_pos)
    num_single_obs = num_grids * 2
    if pomdp.observation_agent == 0
        # Map vector of observations to a unique index
        obs_tuple = Tuple(num_single_obs for _ in 1:pomdp.num_agents)
        return LinearIndices(obs_tuple)[o...]
    else
        return o[1]
    end
end

function POMDPs.observation(pomdp::StochasticMarsPOMDP, a::Vector{Int}, sp::StochasticMarsState)
    if pomdp.observation_agent == 0
        return Deterministic(joint_observation(pomdp, a, sp))
    else
        return Deterministic([single_observation(pomdp, a, sp, pomdp.observation_agent)])
    end
end

function joint_observation(pomdp::StochasticMarsPOMDP, ::Any, sp::StochasticMarsState)
    return [single_observation(pomdp, nothing, sp, ii) for ii in 1:pomdp.num_agents]
end

function single_observation(pomdp::StochasticMarsPOMDP, ::Any, sp::StochasticMarsState, obs_agent::Int)
    num_grids = get_prop(pomdp.mg, :num_grid_pos)
    agent_pos = sp.agent_positions[obs_agent]
    exp_num = get_prop(pomdp.mg, :ds_mapping)[agent_pos]
    exp_val = sp.experimented[exp_num] ? 2 : 1
    return LinearIndices((num_grids, 2))[agent_pos, exp_val]
end

function POMDPs.reward(pomdp::StochasticMarsPOMDP, s::StochasticMarsState, a::Vector{Symbol})
    return POMDPs.reward(pomdp, s, [MARS_SINGLE_ACTIONS_DICT[aj] for aj in a])
end
function POMDPs.reward(pomdp::StochasticMarsPOMDP, s::StochasticMarsState, a::Vector{Int})
    rew = 0.0

    all_agents_same_pos = true
    all_actions_drill = true
    all_action_sample = true
    all_actions_drill_or_sample = true
    for i in 1:pomdp.num_agents
        aᵢ = a[i]
        if !(aᵢ == MARS_SINGLE_ACTIONS_DICT[:drill] || aᵢ == MARS_SINGLE_ACTIONS_DICT[:sample])
            all_actions_drill_or_sample = false
        end
        if !(aᵢ == MARS_SINGLE_ACTIONS_DICT[:drill])
            all_actions_drill = false
        end
        if s.agent_positions[i] != s.agent_positions[1]
            all_agents_same_pos = false
        end
        if !(aᵢ == MARS_SINGLE_ACTIONS_DICT[:sample])
            all_action_sample = false
        end
    end
    
    if all_agents_same_pos
        pos_of_agent = s.agent_positions[1]
        exp_num = get_prop(pomdp.mg, :ds_mapping)[pos_of_agent]
        if !s.experimented[exp_num]
            if all_actions_drill    
                # If at the drill site AND not experimented, then we get the drill reward
                if get_prop(pomdp.mg, pos_of_agent, :is_drill_site)
                    return pomdp.drill_reward
                else
                    return pomdp.ruin_site_penalty
                end
            elseif all_action_sample
                if get_prop(pomdp.mg, pos_of_agent, :is_sample_site)
                    return pomdp.sample_reward
                end
            elseif all_actions_drill_or_sample
                return pomdp.ruin_site_penalty
            end
        end
    end
    
    # If we made it here, then the rest of the rewards are additive for each agent
    temp_exp_status = deepcopy(s.experimented)
    for i in 1:pomdp.num_agents
        aᵢ = a[i]
        if !(aᵢ in [MARS_SINGLE_ACTIONS_DICT[:drill], MARS_SINGLE_ACTIONS_DICT[:sample]])
            rew += pomdp.movement_penalty
        else
            exp_num = get_prop(pomdp.mg, :ds_mapping)[s.agent_positions[i]]
            if temp_exp_status[exp_num]
                rew += pomdp.redundancy_penalty
            else
                if aᵢ == MARS_SINGLE_ACTIONS_DICT[:drill]
                    rew += pomdp.ruin_site_penalty
                else
                    rew += pomdp.sample_reward
                end
            end
            temp_exp_status[exp_num] = true
        end
    end
    return rew
end


const SAMPLE_SITE_COLOR = RGB(0.7, 0.7, 0.8)  # Orange for sample sites
const EXPERIMENTED_SITE_COLOR = RGB(0.6, 0.6, 0.6)  # Gray for experimented sites
const DRILL_SITE_COLOR = RGB(0.4, 0.4, 0.8)   # Blue for drill sites

const MARS_OFFSET_MAGNITUDE = 0.15  # Offset for overlapping agents

"""
    POMDPTools.render(pomdp::StochasticMarsPOMDP, step::NamedTuple)

Render function for `StochasticMarsPOMDP` that plots the environment, agents, and sites.
"""
function POMDPTools.render(pomdp::StochasticMarsPOMDP, step::NamedTuple; title_str="", title_font_size=30)
    
    plt = plot(; 
        title=title_str, titlefont=font(title_font_size),
        legend=false, ticks=false, showaxis=false, grid=false, aspectratio=:equal, 
        size=(800, 800)
    )
    
    px_p_tick = px_per_tick(plt)
    fnt_size = Int(floor(px_p_tick / 35))
    
    plt = plot_stochastic_mars!(plt, pomdp)
    
    plot_belief_flag = true

    plt = add_legend!(plt, pomdp)
    
    # Plot agents if the state is provided
    if !isnothing(get(step, :s, nothing))
        s = step.s
        agent_positions = s.agent_positions
        experimented = s.experimented

        plt = plot_agents!(plt, pomdp, agent_positions)
        plt = plot_sites_status!(plt, pomdp, experimented)
    end

    # Annotate the action if provided
    if !isnothing(get(step, :a, nothing))
        # Determine font size based on plot size
        num_cols = get_prop(pomdp.mg, :ncols)
        num_rows = get_prop(pomdp.mg, :nrows)
        xc = (num_cols + 1) / 2
        yc = -(num_rows + 0.75)
        a = step.a
        action_names = action_name(pomdp, a)
        # Capitalize the first letter of each action
        action_names = [uppercase(first(name)) * lowercase(name[2:end]) for name in action_names]
        action_text = join(action_names, "-")
        action_text = replace(action_text, "[" => "", "]" => "")
        action_text = latexstring("\$\\textrm{$(replace(action_text, " " => "~"))}\$")
        plt = annotate!(plt, xc, yc, (text(action_text, :black, :center, fnt_size)))
    end

    # Plot the belief if provided
    if !isnothing(get(step, :b, nothing)) && plot_belief_flag
        b = step.b
        plt = plot_belief!(plt, pomdp, b; belief_threshold=0.01)
    end
    
    plt = dynamic_plot_resize!(plt; max_size=(800, 800))
    return plt
end

"""
    plot_stochastic_mars(pomdp::StochasticMarsPOMDP; size::Tuple{Int, Int}=(600, 600))

Function to plot the Stochastic Mars environment grid based on the map.
"""
function plot_stochastic_mars!(plt, pomdp::StochasticMarsPOMDP)
    # Get the map string and dimensions
    map_str = map_str_from_metagraph(pomdp)
    lines = split(map_str, '\n')
    nrows = length(lines)
    ncols = length(lines[1])

    num_cols = get_prop(pomdp.mg, :ncols)
    num_rows = get_prop(pomdp.mg, :nrows)

    # Plot the grid cells
    for xi in 1:nrows
        for yj in 1:ncols
            char = lines[xi][yj]
            color = :white  # Default color for open space

            if char == 'x'      # Wall
                color = :black
            elseif char == 's'  # Sample site
                color = SAMPLE_SITE_COLOR
            elseif char == 'd'  # Drill site
                color = DRILL_SITE_COLOR
            else
                color = :white
            end
            # Draw the cell
            plt = plot!(plt, rect(0.5, 0.5, yj, -xi); color=color, alpha=0.5)
        end
    end
    
    # Plot a blank section at the bottom for action text
    x_mid = ((0.5 + num_cols * 1.0) - 0.5) / 2.0
    plt = plot!(plt, rect(0.0, 0.5, x_mid, -0.3 - num_rows); 
        linecolor=:white, color=:white)

    # Plot a blank section at the top for the title
    plt = plot!(plt, rect(0.0, 0.30, 0.0, -0.5); 
        linecolor=:white, color=:white)
        
    return plt
end

"""
    plot_agents!(plt, pomdp::StochasticMarsPOMDP, agent_positions::Tuple)

Function to plot agents at their positions.
"""
function plot_agents!(plt, pomdp::StochasticMarsPOMDP, agent_positions::Tuple)
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)
    num_agents = length(agent_positions)
    offset_angle = 2*π / num_agents
    for i in 1:num_agents
        pos = agent_positions[i]
        (xi, yj) = node_pos_mapping[pos]

        # Offset if agents are in the same position
        offset = (0.0, 0.0)
        c = ROBOT_COLORS[i]

        if sum(agent_positions .== pos) > 1
            angle = π/2 + (i-1)*offset_angle
            offset = MARS_OFFSET_MAGNITUDE .* (cos(angle), sin(angle))
        end

        x_agent = yj + offset[1]
        y_agent = -xi + offset[2]

        plt = plot_agent!(plt, x_agent, y_agent; color=c, body_size=0.2)
    end
    return plt
end



"""
    plot_sites_status!(plt, pomdp::StochasticMarsPOMDP, experimented::Vector{Bool})

Function to indicate the status of sample and drill sites (experimented or not).
"""
function plot_sites_status!(plt, pomdp::StochasticMarsPOMDP, experimented::Vector{Bool})
    ds_mapping = get_prop(pomdp.mg, :ds_mapping)
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)
    for (pos, site_num) in ds_mapping
        is_experimented = experimented[site_num]
        (xi, yj) = node_pos_mapping[pos]
        if is_experimented
            # Draw an "X" at the top right corner of the cell to indicate it has been experimented
            x_size = 0.15  # Size of the X
            y_offset = 0.33  # Offset from the bottom of the cell
            x_offset = 0.4
            plt = plot!(plt, [yj - x_size/2 + x_offset, yj + x_size/2 + x_offset], 
                [-xi + y_offset, -xi + y_offset + x_size]; color=:black, linewidth=4)
            plt = plot!(plt, [yj - x_size/2 + x_offset, yj + x_size/2 + x_offset], 
                [-xi + y_offset + x_size, -xi + y_offset]; color=:black, linewidth=4)
        end
        # Plot the site number in the top left corner of the cell
        px_p_tick = px_per_tick(plt)
        fnt_size = Int(floor(px_p_tick / 18))
        site_text = string(site_num)
        plt = annotate!(plt, yj - 0.42, -xi + 0.42, (text(site_text, :black, :center, fnt_size)))
    end
    return plt
end

"""
    plot_belief!(plt, pomdp::StochasticMarsPOMDP, b::Any; belief_threshold::Float64=0.01)

Function to plot the belief over agent positions onto the existing plot.
"""
function plot_belief!(plt, pomdp::StochasticMarsPOMDP, b::Any; belief_threshold::Float64=0.01)
    # Convert belief to a vector of probabilities and corresponding states
    if b isa DiscreteBelief
        probs = b.b
        state_list = b.state_list
    elseif b isa SparseCat
        probs = b.probs
        state_list = b.vals
    elseif b isa Deterministic
        probs = [1.0]
        state_list = [b.val]
    elseif b isa ParticleCollection
        state_num_part = Dict{StochasticMarsState, Int}()
        for part in b.particles
            state_num_part[part] = get(state_num_part, part, 0) + 1
        end
        state_list = Vector{StochasticMarsState}()
        probs = Vector{Float64}()
        for (k, v) in state_num_part
            push!(state_list, k)
            push!(probs, v / length(b.particles))
        end
    else
        error("Unsupported belief type")
    end

    num_agents = pomdp.num_agents
    num_cells = get_prop(pomdp.mg, :num_grid_pos)
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)

    # Initialize belief grids for agents
    agent_beliefs = [zeros(num_cells) for _ in 1:num_agents]
    site_beliefs = zeros(num_cells)
    
    # Aggregate belief over the state space
    for (prob, s) in zip(probs, state_list)
        for i in 1:num_agents
            agent_beliefs[i][s.agent_positions[i]] += prob
        end
        for (site_num, is_experimented) in enumerate(s.experimented)
            if is_experimented
                site_beliefs[site_num] += prob
            end
        end
    end

    px_p_tick = px_per_tick(plt)
    
    # Plot agent position beliefs
    for i in 1:num_agents
        for pos in 1:num_cells
            belief = agent_beliefs[i][pos]
            if belief >= belief_threshold
                (xi, yj) = node_pos_mapping[pos]
                x_agent = yj - 0.38 + 0.25 * (i - 1)
                y_agent = -xi - 0.38
                c = ROBOT_COLORS[i]
                alpha_scale = max(belief, 0.1)
                plt = plot!(plt, circ(x_agent, y_agent, 0.11); color=c, alpha=alpha_scale)
                # Plot belief percentage as text
                
                fnt_size = Int(floor(px_p_tick / 20))
                belief_text = "$(round(belief, digits=2))"
                plt = annotate!(plt, x_agent, y_agent, (text(belief_text, :black, :center, fnt_size)))
            end
        end
    end
    
    # Plot the belief at the top at each grid point
    for pos in 1:num_cells
        belief = site_beliefs[pos]
        (xi, yj) = node_pos_mapping[pos]
        fnt_size = Int(floor(px_p_tick / 17))
        belief_text = "$(round(belief, digits=2))"
        plt = annotate!(plt, yj - 0.0, -xi + 0.42, (text(belief_text, :black, :center, fnt_size)))
        
    end
    
    return plt
end

"""
    add_legend!(plt::Plot, pomdp::StochasticMarsPOMDP)

Function to add a legend to the plot.
"""
function add_legend!(plt, pomdp::StochasticMarsPOMDP)
    # Coordinates for the legend
    
    legend_x = -1.  # Adjust as needed
    legend_x_center = (-1.0 + 0.5) / 2.0 - 0.1
    legend_y_start = -0.65  # Start at the top
    y_spacing = 0.2  # Spacing between entries
    
    px_p_tick = px_per_tick(plt)
    fnt_size = Int(floor(px_p_tick / 17))
    
    plt = annotate!(plt, legend_x_center, legend_y_start, text("Legend", :black, :bold, :center, fnt_size))
    
    legend_y_start = -1.0  # Start at the top
    curr_y = legend_y_start
    
    fnt_size = Int(floor(px_p_tick / 30))
    
    rect_shape = rect(0.07, 0.07, legend_x, curr_y)
    plt = plot!(plt, rect_shape; color=SAMPLE_SITE_COLOR, alpha=0.5)
    plt = annotate!(plt, legend_x + 0.12, curr_y, text("Sample Site", :black, :left, fnt_size))

    rect_shape = rect(0.07, 0.07, legend_x + 0.8, curr_y)
    plt = plot!(plt, rect_shape; color=DRILL_SITE_COLOR, alpha=0.5)
    plt = annotate!(plt, legend_x + 0.12 + 0.8, curr_y, text("Drill Site", :black, :left, fnt_size))
        
    # # Add agent legend
    num_agents = pomdp.num_agents
    for i in 1:num_agents
        y_space_mult = floor(Int, (i - 1) / 2)
        curr_y = legend_y_start - (1 + y_space_mult) * y_spacing
        
        x_space = i % 2 == 1 ? 0.0 : 0.8
        
        # Plot agent symbol
        plt = plot!(plt, circ(legend_x + x_space, curr_y, 0.07); color=ROBOT_COLORS[i])
        plt = annotate!(plt, legend_x + x_space + 0.12, curr_y, text("Agent $i", :black, :left, fnt_size))
    end


    # Add belief explanations
    y_pos = curr_y - y_spacing  - 0.5
    x_pos = legend_x_center

    plt = plot!(plt, rect(0.5, 0.5, x_pos, y_pos); color=:white)

    fnt_size = Int(floor(px_p_tick / 40))
    plt = annotate!(plt, x_pos - 0.47, y_pos + 0.42, text("Site #", :black, :left, fnt_size))

    plt = annotate!(plt, x_pos, y_pos + 0.42, text("Belief\nExp State", :black, :center, fnt_size))

    plt = annotate!(plt, x_pos + 0.47, y_pos + 0.42, text("Exp'd", :black, :right, fnt_size))
    
    plt = plot_agent!(plt, x_pos, y_pos; color=:white, body_size=0.2)
    plt = annotate!(plt, x_pos, y_pos, text("True Agent\nState", :black, :center, fnt_size))
    
    x_pos_p = x_pos - 0.38
    y_pos_p = y_pos - 0.38
    for i in 1:num_agents
        x_agent = x_pos_p + 0.25 * (i - 1)
        y_agent = y_pos_p 
        plt = plot!(plt, circ(x_agent, y_agent, 0.11); color=ROBOT_COLORS[i], alpha=0.3)
    end
    
    plt = annotate!(plt, x_pos, y_pos_p, text("Belief of Agent State", :black, :center, fnt_size))
    
    return plt
end

"""
    action_name(pomdp::StochasticMarsPOMDP, a::Int)

Helper function to get the action names from the action index.
"""
function action_name(::StochasticMarsPOMDP, a::Vector{Int})
    action_syms = Dict(v => k for (k, v) in MARS_SINGLE_ACTIONS_DICT)
    action_names = [string(action_syms[ai]) for ai in a]
    return action_names
end
