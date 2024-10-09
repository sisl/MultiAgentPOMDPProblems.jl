function POMDPs.observations(pomdp::BoxPushPOMDP)
    if pomdp.observation_prob == 1.0
        obs_size = length(BOX_PUSH_OBSERVATIONS)
    else
        obs_size = length(BOX_PUSH_MOD_OBSERVATIONS)
    end
    if pomdp.observation_agent == 0
        return vec([[ci.I...] for ci in CartesianIndices((obs_size, obs_size))])
    else
        return [[i] for i in 1:obs_size]
    end
end

function POMDPs.obsindex(pomdp::BoxPushPOMDP, o::Vector{Int})
    if pomdp.observation_prob == 1.0
        obs_size = length(BOX_PUSH_OBSERVATIONS)
    else
        obs_size = length(BOX_PUSH_MOD_OBSERVATIONS)
    end
    @assert all(1 <= oi <= obs_size for oi in o) "Invalid observation index"
    if pomdp.observation_agent == 0
        return LinearIndices((obs_size, obs_size))[o...]
    else
        return o[1]
    end
end

function POMDPs.observation(pomdp::BoxPushPOMDP, ::Any, sp::BoxPushState)
    if pomdp.observation_agent == 0
        obs = [single_agent_observation(pomdp, sp, i) for i in 1:2]
        if pomdp.observation_prob == 1.0
            return SparseCat([obs], [1.0])
        else
            poss_obs = [
                obs,
                [obs[1], BOX_PUSH_MOD_OBSERVATIONS[:no_obs]], 
                [BOX_PUSH_MOD_OBSERVATIONS[:no_obs], obs[2]],
                [BOX_PUSH_MOD_OBSERVATIONS[:no_obs], BOX_PUSH_MOD_OBSERVATIONS[:no_obs]]
            ]
            poss_obs_probs = [
                pomdp.observation_prob^2,
                pomdp.observation_prob * (1 - pomdp.observation_prob),
                (1 - pomdp.observation_prob) * pomdp.observation_prob,
                (1 - pomdp.observation_prob)^2
            ]
            return SparseCat(poss_obs, poss_obs_probs)
        end
    else
        if pomdp.observation_prob == 1.0
            return SparseCat([[single_agent_observation(pomdp, sp, pomdp.observation_agent)]], [1.0])
        else
            obs = [[single_agent_observation(pomdp, sp, pomdp.observation_agent)], [BOX_PUSH_MOD_OBSERVATIONS[:no_obs]]]
            return SparseCat(obs, [pomdp.observation_prob, 1.0 - pomdp.observation_prob])
        end
    end
end

function single_agent_observation(pomdp::BoxPushPOMDP, sp::BoxPushState, agent_idx::Int)
    if pomdp.observation_prob == 1.0
        obs_dict = BOX_PUSH_OBSERVATIONS
    else
        obs_dict = BOX_PUSH_MOD_OBSERVATIONS
    end
    
    orient = sp.agent_orientation[agent_idx]
    pos = sp.agent_pos[agent_idx]
    
    sp_n = pos
    for neighbor in neighbors(pomdp.mg, pos)
        edge_orientation = AGENT_ORIENTATIONS[get_prop(pomdp.mg, pos, neighbor, :action)]
        if edge_orientation == orient
            sp_n = neighbor
        end
    end
    
    # No valid movement = wall
    if sp_n == pos
        return obs_dict[:wall]
    end
    
    # Check for other agent
    if sp_n in sp.agent_pos
        return obs_dict[:agent]
    end
    
    # Check for small box
    if sp_n in sp.small_box_pos
        return obs_dict[:small_box]
    end
    
    # Check for large box
    if sp_n in sp.large_box_pos || sp_n in (sp.large_box_pos .+ 1)
        return obs_dict[:large_box]
    end
    
    # Made it here, so it is empty
    return obs_dict[:empty]
end
