
POMDPs.states(pomdp::BoxPushPOMDP) = pomdp

function Base.length(pomdp::BoxPushPOMDP)
    if pomdp.map_option == 1
        # 4(4*3) + 4(4*2) + 4(4*1) + 3 = 99
        # 96 robot positions/orientations + 3 box goals
        # 99 states
        return length(pomdp.states)
    elseif pomdp.map_option == 2
        # 14(14-1)*4^2 = 2912 robot positions/orientations
        # 4*4*2 = 32 box positions (not including goal)
        # Small boxes at goal/top: 3
        # Large box at goal: 1
        # Total states: 2912*32 + 3 + 1 = 93188
        return length(pomdp.states)
    else
        throw(ArgumentError("Invalid map option"))
    end
end

function Base.iterate(pomdp::BoxPushPOMDP, ii::Int=1)
    if ii > length(pomdp)
        return nothing
    end
    s = getindex(pomdp, ii)
    return (s, ii + 1)
end

function Base.getindex(pomdp::BoxPushPOMDP, si::Int)
    @assert si <= length(pomdp) "Index out of bounds"
    @assert si > 0 "Index out of bounds"
    return pomdp.states[si]
end

function Base.getindex(pomdp::BoxPushPOMDP, si_range::UnitRange{Int})
    return [getindex(pomdp, si) for si in si_range]
end

function Base.firstindex(::BoxPushPOMDP)
    return 1
end

function Base.lastindex(pomdp::BoxPushPOMDP)
    return length(pomdp)
end

function POMDPs.stateindex(pomdp::BoxPushPOMDP, s::BoxPushState)
    return pomdp.state_index[s]
end

function POMDPs.initialstate(pomdp::BoxPushPOMDP)
    init_state = BoxPushState(
        get_prop(pomdp.mg, :small_box_pos),
        get_prop(pomdp.mg, :large_box_pos),
        get_prop(pomdp.mg, :agent_start_pos),
        (:east, :west)
    )
    return Deterministic(init_state)
end
