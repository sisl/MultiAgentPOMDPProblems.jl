POMDPs.reward(::JointMeetPOMDP, ::JointMeetState, ::Tuple{}) = throw(ArgumentError("Invalid action tuple (empty)"))
function POMDPs.reward(pomdp::JointMeetPOMDP, si::JointMeetState, a::Tuple{Vararg{Int}})
    return POMDPs.reward(pomdp, si, collect(a))
end
function POMDPs.reward(pomdp::JointMeetPOMDP, si::JointMeetState, a::Tuple{Vararg{Symbol}})
    return POMDPs.reward(pomdp, si, [JOINTMEET_ACTIONS_DICT[ai] for ai in a])
end
function POMDPs.reward(pomdp::JointMeetPOMDP, s::JointMeetState, a::Vector{Int})
    rew = pomdp.step_penalty
    
    # All robots in the same position and they all choose to stay -> return the meet_reward
    if all(r_pos == s.r_positions[1] for r_pos in s.r_positions)
        if isempty(pomdp.meet_reward_locations) || s.r_positions[1] in pomdp.meet_reward_locations
            rew += pomdp.meet_reward
        end
    end
    
    # Determine if action would hit a wall for each agent. If so, return the wall_penalty
    if pomdp.wall_penalty != 0.0
        for (r_pos, a_i) in zip(s.r_positions, a)
            if a_i == JOINTMEET_ACTIONS_DICT[:stay]
                continue
            end
            poss_actions = [JOINTMEET_ACTIONS_DICT[get_prop(pomdp.mg, r_pos, n_i, :action)] for n_i in neighbors(pomdp.mg, r_pos)]
            if a_i ∉ poss_actions
                rew += pomdp.wall_penalty
            end
        end
    end
    
    return rew
end
