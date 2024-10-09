POMDPs.states(pomdp::JointMeetPOMDP) = pomdp

function Base.iterate(pomdp::JointMeetPOMDP, ii::Int=1)
    if ii > length(pomdp)
        return nothing
    end
    s = getindex(pomdp, ii)
    return (s, ii + 1)
end

function Base.getindex(pomdp::JointMeetPOMDP, si::Int)
    @assert si <= length(pomdp) "Index out of bounds"
    @assert si > 0 "Index out of bounds"
    num_grid_pos = get_prop(pomdp.mg, :num_grid_pos)
    env_state_tuple = Tuple(CartesianIndices(Tuple(num_grid_pos for _ in 1:pomdp.num_agents))[si])
    return JointMeetState(env_state_tuple)
end

function Base.getindex(pomdp::JointMeetPOMDP, si_range::UnitRange{Int})
    return [getindex(pomdp, si) for si in si_range]
end

function Base.firstindex(::JointMeetPOMDP)
    return 1
end
function Base.lastindex(pomdp::JointMeetPOMDP)
    return length(pomdp)
end

function POMDPs.stateindex(pomdp::JointMeetPOMDP, s::JointMeetState)
    num_grid_pos = get_prop(pomdp.mg, :num_grid_pos)
    for (ii, r_pos) in enumerate(s.r_positions)
        @assert r_pos > 0 && r_pos <= num_grid_pos "Robot $ii invalid position"
    end

    si = LinearIndices(Tuple(num_grid_pos for _ in 1:pomdp.num_agents))[s.r_positions...]
    return si
end

# Uniform over the entire state space 
function POMDPs.initialstate(pomdp::JointMeetPOMDP)
    if !isnothing(pomdp.init_state)
        return Deterministic(pomdp.init_state)
    else
        probs = normalize(ones(length(pomdp)), 1)
        return SparseCat(ordered_states(pomdp), probs)
    end
end
