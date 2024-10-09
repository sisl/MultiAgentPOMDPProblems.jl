"""
    observations(pomdp::JointMeetPOMDP)
    
Observation options:
- :full - Full observation of position
    - If joint (observation_agent=0), then joint full observation
    - If observation_agent != 0, then observation is own position and number of agents in 
      the same grid location.
- :left_and_same: number of moves for agent the left wall and number of agents in the same position
- :right_and_same: number of moves for agent to the right wall and number of agents in the same position
- :boundaries_lr: if agent is at leftmost or rightmost boundary
- :boundaries_ud: if agent is at topmost or bottommost boundary
- :boundaries_both: observation of left, right, top, and bottom boundaries (9 per agent)
"""
function POMDPs.observations(pomdp::JointMeetPOMDP)
    obs_size = get_obs_size(pomdp)
    if pomdp.observation_agent == 0
        return vec([[ci.I...] for ci in CartesianIndices(Tuple(obs_size for _ in 1:pomdp.num_agents))])
    else
        return [[oi] for oi in 1:obs_size]
    end
end

function get_obs_size(pomdp::JointMeetPOMDP)
    num_agents = pomdp.num_agents
    obs_size = 0
    if pomdp.observation_option == :full
        num_grids = get_prop(pomdp.mg, :num_grid_pos)
        if pomdp.observation_agent == 0
            # Full observation for all agents
            obs_size = num_grids
        else
            obs_size = num_grids * num_agents
        end
    elseif pomdp.observation_option == :left_and_same || pomdp.observation_option == :right_and_same
        obs_size = get_prop(pomdp.mg, :ncols) * num_agents
        
    elseif pomdp.observation_option == :boundaries_lr || pomdp.observation_option == :boundaries_ud
        obs_size = 4 # e.g. left, right, nothing, both
    elseif pomdp.observation_option == :boundaries_both
        obs_size = 16 # (left, right, both, none) x (top, bottom, both, none)
    else
        throw(ArgumentError("Invalid observation option: $(pomdp.observation_option)"))
    end
    return obs_size
end

function POMDPs.obsindex(pomdp::JointMeetPOMDP, o::Vector{Int})
    obs_size = get_obs_size(pomdp)
    @assert all(1 <= oi <= obs_size for oi in o) "Invalid observation index"
    if pomdp.observation_agent == 0
        return LinearIndices(Tuple(obs_size for _ in 1:pomdp.num_agents))[o...]
    else
        return o[1]
    end
end

function POMDPs.observation(pomdp::JointMeetPOMDP, a::Tuple{Vararg}, sp::JointMeetState)
    return POMDPs.observation(pomdp, collect(a), sp)
end
function POMDPs.observation(pomdp::JointMeetPOMDP, a::Vector{Symbol}, sp::JointMeetState)
    return POMDPs.observation(pomdp, [JOINTMEET_ACTIONS_DICT[ai] for ai in a], sp)
end
function POMDPs.observation(pomdp::JointMeetPOMDP, a::Vector{Int}, sp::JointMeetState)
    if pomdp.observation_option == :full
        if pomdp.observation_agent == 0
            return joint_observation(pomdp, a, sp)
        else
            return single_observation(pomdp, a, sp, pomdp.observation_agent)
        end
    elseif pomdp.observation_option == :left_and_same
        if pomdp.observation_agent == 0
            return joint_distance_from_wall(pomdp, a, sp, :west)
        else
            return distance_from_wall(pomdp, a, sp, pomdp.observation_agent, :west)
        end
    elseif pomdp.observation_option == :right_and_same
        if pomdp.observation_agent == 0
            return joint_distance_from_wall(pomdp, a, sp, :east)
        else
            return distance_from_wall(pomdp, a, sp, pomdp.observation_agent, :east)
        end
    elseif pomdp.observation_option == :boundaries_lr
        if pomdp.observation_agent == 0
            return joint_left_right_boundary_observation(pomdp, a, sp)
        else
            return left_right_boundary_observation(pomdp, a, sp, pomdp.observation_agent)
        end
    elseif pomdp.observation_option == :boundaries_ud
        if pomdp.observation_agent == 0
            return joint_top_bottom_boundary_observation(pomdp, a, sp)
        else
            return top_bottom_boundary_observation(pomdp, a, sp, pomdp.observation_agent)
        end
    elseif pomdp.observation_option == :boundaries_both
        if pomdp.observation_agent == 0
            return joint_all_boundary_observation(pomdp, a, sp)
        else
            return both_boundary_observation(pomdp, a, sp, pomdp.observation_agent)
        end
    else
        throw(ArgumentError("Invalid observation option: $(pomdp.observation_option)"))
    end
end

"""
    joint_observation(pomdp::JointMeetPOMDP, a::Int, sp::JointMeetState)
    
Full observation of all agents.
"""
function joint_observation(::JointMeetPOMDP, ::Any, sp::JointMeetState)
    return SparseCat([collect(sp.r_positions)], 1.0)
end

"""
    joint_distance_from_wall(pomdp::JointMeetPOMDP, a::Int, sp::JointMeetState, direction::Symbol)
    
Observation of the distance from the wall for all agents. Distribution is truncated normal
with mean at the distance from the wall and standard deviation `pomdp.observation_sigma`.

Observations for agents are independent, so we can calculate the observation for each agent
and then combine them.
"""
function joint_distance_from_wall(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState, direction::Symbol)
    
    if direction == :east || direction == :west
        prop = :ncols
    else
        prop = :nrows
    end
    
    obs_tuple = (get_prop(pomdp.mg, prop), pomdp.num_agents)
    max_dist = get_prop(pomdp.mg, prop)
    
    agent_from_wall = zeros(Int, pomdp.num_agents)
    agent_num_same = zeros(Int, pomdp.num_agents)
    agent_obs = zeros(Int, pomdp.num_agents)
    for agent_num in 1:pomdp.num_agents
        agent_pos = sp.r_positions[agent_num]
        
        other_positions = [o_pos for (ii, o_pos) in enumerate(sp.r_positions) if ii != agent_num]
        num_same_p = count(agent_pos .== other_positions) + 1
        
        cnt = 0
        able_to_move = true
        while able_to_move
            able_to_move = false
            neighs = neighbors(pomdp.mg, agent_pos)
            for n_i in neighs
                if get_prop(pomdp.mg, agent_pos, n_i, :action) == direction
                    agent_pos = n_i
                    cnt += 1
                    able_to_move = true
                    break
                end
            end
        end
        dist_from_wall = cnt + 1
        agent_from_wall[agent_num] = dist_from_wall
        agent_num_same[agent_num] = num_same_p
        agent_obs[agent_num] = LinearIndices(obs_tuple)[dist_from_wall, num_same_p]
    end
    
    if pomdp.observation_sigma == 0
        return SparseCat([agent_obs], 1.0)
    else
        # For all possible observations, calculate the probability of each observation
        
        obs_per_agent = fill(Vector{Int}(undef, max_dist), pomdp.num_agents)
        obs_prob_per_agent = fill(Vector{Float64}(undef, max_dist), pomdp.num_agents)
        
        for oi in 1:max_dist
            for agent_i in 1:pomdp.num_agents
                obs_pdf = truncated(Normal(agent_from_wall[agent_i], pomdp.observation_sigma), 1.0, max_dist)
                obs_prob = cdf(obs_pdf, oi + 0.5) - cdf(obs_pdf, oi - 0.5)
                obs_idx = LinearIndices(obs_tuple)[oi, agent_num_same[agent_i]]
                obs_per_agent[agent_i][oi] = obs_idx
                obs_prob_per_agent[agent_i][oi] = obs_prob
            end
        end
        
        poss_obs = vec([collect(obs) for obs in Iterators.product(obs_per_agent...)])
        probs = prod.(vec(collect(Iterators.product(obs_prob_per_agent...))))
        
        non_zero_probs = findall(x -> x > 0, probs)
        
        return SparseCat(poss_obs[non_zero_probs], probs[non_zero_probs])
    end
end

"""
    distance_from_wall(pomdp::JointMeetPOMDP, a::Int, sp::JointMeetState, agent_num::Int, direction::Symbol)
    
Observation of the distance from the wall for a given agent. Distribution is truncated normal
with mean at the distance from the wall and standard deviation `pomdp.observation_sigma`.
"""
function distance_from_wall(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState, agent_num::Int, direction::Symbol)
    
    if direction == :east || direction == :west
        prop = :ncols
    else
        prop = :nrows
    end
    
    agent_pos = sp.r_positions[agent_num]
    cnt = 0
    able_to_move = true
    while able_to_move
        able_to_move = false
        neighs = neighbors(pomdp.mg, agent_pos)
        for n_i in neighs
            if get_prop(pomdp.mg, agent_pos, n_i, :action) == direction
                agent_pos = n_i
                cnt += 1
                able_to_move = true
                break
            end
        end
    end
    
    r_pos = sp.r_positions[agent_num]
    other_positions = [o_pos for (ii, o_pos) in enumerate(sp.r_positions) if ii != agent_num]
    num_robots_in_same_pos = count(r_pos .== other_positions)
    
    obs_option_tuple = Tuple(get_prop(pomdp.mg, prop) for _ in 1:pomdp.num_agents)
    if pomdp.observation_sigma == 0
        obs = LinearIndices(obs_option_tuple)[cnt+1, num_robots_in_same_pos+1]
        return SparseCat([[obs]], 1.0)
    else
        max_dist = get_prop(pomdp.mg, prop)
        obs_dist = truncated(Normal(cnt+1, pomdp.observation_sigma), 1.0, max_dist)
        
        obs_list = Vector{Vector{Int}}()
        obs_probs = Vector{Float64}()
        for o in 1:max_dist
            prob = cdf(obs_dist, o+0.5) - cdf(obs_dist, o-0.5)
            if prob > 0
                obs_lin_idx = LinearIndices(obs_option_tuple)[o, num_robots_in_same_pos+1]
                push!(obs_list, [obs_lin_idx])
                push!(obs_probs, prob)
            end
        end
        return SparseCat(obs_list, normalize(obs_probs, 1.0))
    end
end


"""
    single_observation(pomdp::JointMeetPOMDP, a::Int, sp::JointMeetState, agent_num::Int)
    
Full observation of agent's own position and number of robots in the same position.
"""
function single_observation(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState, agent_num::Int)
    
    # If the agent is not in another grid position, return the grid position as the observation
    r_pos = sp.r_positions[agent_num]
    other_positions = [o_pos for (ii, o_pos) in enumerate(sp.r_positions) if ii != agent_num]
    num_robots_in_same_pos = count(r_pos .== other_positions)
    
    obs_indices = LinearIndices((get_prop(pomdp.mg, :num_grid_pos), pomdp.num_agents))
    obs = obs_indices[r_pos, num_robots_in_same_pos+1]
    return SparseCat([[obs]], 1.0)
end

"""
    at_boundary(pomdp::JointMeetPOMDP, sp::JointMeetState, agent_num::Int, direction::Symbol)
    
Return boolean of whether the agent is at a boundary (designated by `direction`).
"""
function at_boundary(
    pomdp::JointMeetPOMDP, sp::JointMeetState, agent_num::Int, direction::Symbol
)
    agent_pos = sp.r_positions[agent_num]
    neighs = neighbors(pomdp.mg, agent_pos)
    able_to_move = false
    for n_i in neighs
        if get_prop(pomdp.mg, agent_pos, n_i, :action) == direction
            able_to_move = true
            break
        end
    end
    return !able_to_move
end

function at_top_boundary(pomdp::JointMeetPOMDP, sp::JointMeetState, agent_num::Int)
    return at_boundary(pomdp, sp, agent_num, :north)
end

function at_bottom_boundary(pomdp::JointMeetPOMDP, sp::JointMeetState, agent_num::Int)
    return at_boundary(pomdp, sp, agent_num, :south)
end

function at_left_boundary(pomdp::JointMeetPOMDP, sp::JointMeetState, agent_num::Int)
    return at_boundary(pomdp, sp, agent_num, :west)
end

function at_right_boundary(pomdp::JointMeetPOMDP, sp::JointMeetState, agent_num::Int)
    return at_boundary(pomdp, sp, agent_num, :east)
end

function get_discrete_boundary_obs(bool_array::Vector{Bool})
    obs = 0
    for b in bool_array
        obs = obs * 2 + (b ? 1 : 0)
    end
    obs += 1  # Adjusting index to start from 1
    return obs
end

function joint_left_right_boundary_observation(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState)
    num_agents = pomdp.num_agents
    agent_obs_array = Vector{Int}(undef, num_agents)
    for agent_num in 1:num_agents
        at_left = at_left_boundary(pomdp, sp, agent_num)
        at_right = at_right_boundary(pomdp, sp, agent_num)
        agent_obs_array[agent_num] = get_discrete_boundary_obs([at_left, at_right])  # Returns 1 to 4
    end
    return SparseCat([agent_obs_array], 1.0)
end

function left_right_boundary_observation(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState, agent_num::Int)
    at_left = at_left_boundary(pomdp, sp, agent_num)
    at_right = at_right_boundary(pomdp, sp, agent_num)
    return SparseCat([[get_discrete_boundary_obs([at_left, at_right])]], 1.0)
end

function joint_top_bottom_boundary_observation(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState)
    num_agents = pomdp.num_agents
    agent_obs_array = Vector{Int}(undef, num_agents)
    for agent_num in 1:num_agents
        at_top = at_top_boundary(pomdp, sp, agent_num)
        at_bottom = at_bottom_boundary(pomdp, sp, agent_num)
        agent_obs_array[agent_num] = get_discrete_boundary_obs([at_top, at_bottom])
    end
    return SparseCat([agent_obs_array], 1.0)
end

function top_bottom_boundary_observation(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState, agent_num::Int)
    at_top = at_top_boundary(pomdp, sp, agent_num)
    at_bottom = at_bottom_boundary(pomdp, sp, agent_num)
    return SparseCat([[get_discrete_boundary_obs([at_top, at_bottom])]], 1.0)
end

function get_agent_boundary_lr_tb_obs(pomdp::JointMeetPOMDP, sp::JointMeetState, agent_num::Int)
    at_left = at_left_boundary(pomdp, sp, agent_num)
    at_right = at_right_boundary(pomdp, sp, agent_num)
    at_top = at_top_boundary(pomdp, sp, agent_num)
    at_bottom = at_bottom_boundary(pomdp, sp, agent_num)
    return get_discrete_boundary_obs([at_left, at_right, at_top, at_bottom])
end

function joint_all_boundary_observation(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState)
    num_agents = pomdp.num_agents
    agent_obs_array = Vector{Int}(undef, num_agents)
    for agent_num in 1:num_agents
        agent_obs_array[agent_num] = get_agent_boundary_lr_tb_obs(pomdp, sp, agent_num)
    end
    return SparseCat([agent_obs_array], 1.0)
end

function both_boundary_observation(pomdp::JointMeetPOMDP, ::Any, sp::JointMeetState, agent_num::Int)
    obs = get_agent_boundary_lr_tb_obs(pomdp, sp, agent_num)
    return SparseCat([[obs]], 1.0)
end
