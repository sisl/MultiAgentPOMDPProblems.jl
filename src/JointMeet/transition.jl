      
function POMDPs.transition(pomdp::JointMeetPOMDP, si::JointMeetState, a::Tuple)
    return POMDPs.transition(pomdp, si, collect(a))
end
function POMDPs.transition(pomdp::JointMeetPOMDP, s::JointMeetState, a::Vector{Symbol})
    return POMDPs.transition(pomdp, s, [JOINTMEET_ACTIONS_DICT[ai] for ai in a])
end
function POMDPs.transition(pomdp::JointMeetPOMDP, s::JointMeetState, a::Vector{Int})
    
    if pomdp.transition_prob == 1.0
        r_pos′ = Vector{Int}(undef, pomdp.num_agents)
        for (i, r_pos) in enumerate(s.r_positions)
            r_pos′[i] = move_direction(pomdp, r_pos, a[i])
        end
        
        new_state = JointMeetState(Tuple(r_pos′))
        return Deterministic(new_state)
    end
    
    r_pos′ = Vector{Vector{Int}}(undef, pomdp.num_agents)
    prob′ = Vector{Vector{Float64}}(undef, pomdp.num_agents)
    for ii in 1:pomdp.num_agents
        aᵢ = a[ii]
        sᵢ = s.r_positions[ii]
        if aᵢ == JOINTMEET_ACTIONS_DICT[:stay]
            r_pos′[ii] = [sᵢ]
            prob′[ii] = [1.0]
            continue
        end
        
        primary_agent_pos = move_direction(pomdp, sᵢ, aᵢ)
        r_pos′[ii] = [primary_agent_pos]
        prob′[ii] = [pomdp.transition_prob]
        
        alt_transitions = get_alt_transitions(pomdp, aᵢ)
        
        for a_alt in alt_transitions
            s′ = move_direction(pomdp, sᵢ, a_alt)
            push!(r_pos′[ii], s′)
            push!(prob′[ii], (1.0 - pomdp.transition_prob) / length(alt_transitions))
        end
    end
    
    new_states = []
    new_probs = []
    for (pos, prob) in zip(Iterators.product(r_pos′...), Iterators.product(prob′...))
        new_state = JointMeetState(pos)
        if new_state in new_states
            idx = findfirst(new_state == new_state_i for new_state_i in new_states)
            new_probs[idx] += prod(prob)
            continue
        else
            push!(new_states, new_state)
            push!(new_probs, prod(prob))
        end
    end
    return SparseCat(new_states, normalize(new_probs, 1.0))
end


"""
    move_direction(pomdp::JointMeetPOMDP, v::Int, a::Int)

Move the robot in the direction of the action. Finds the neighbors of the current node
and checks if the action is valid (edge with corresponding action exists). If so, returns
the node index of the valid action.
"""
function move_direction(pomdp::JointMeetPOMDP, v::Int, a::Int)
    if a == JOINTMEET_ACTIONS_DICT[:stay]
        return v
    end
    
    neighs = neighbors(pomdp.mg, v)
    for n_i in neighs
        if JOINTMEET_ACTIONS_DICT[get_prop(pomdp.mg, v, n_i, :action)] == a
            return n_i
        end
    end
    return v
end

const ALT_TRANSITIONS_NOT_OPPOSITE = Dict(1 => (2, 4), 2 => (1, 3), 3 => (2, 4), 4 => (1, 3), 5 => ())

function get_alt_transitions(pomdp::JointMeetPOMDP, a::Int)
    if a == JOINTMEET_ACTIONS_DICT[:stay]
        return ()
    end
   
    if pomdp.transition_alternatives == :all
        # Set diff of all actions minus the current
        return setdiff(collect(1:length(JOINTMEET_ACTIONS_DICT)), [a])
    elseif pomdp.transition_alternatives == :not_opposite
        return ALT_TRANSITIONS_NOT_OPPOSITE[a]
    else
        throw(ArgumentError("Invalid transition alternatives: $(pomdp.transition_alternatives)"))
    end
end
