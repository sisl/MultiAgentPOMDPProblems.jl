const JOINTMEET_ACTIONS_DICT = Dict(:north => 1, :east => 2, :south => 3, :west => 4, :stay => 5)
const JOINTMEET_ACTION_NAMES = Dict(1 => "North", 2 => "East", 3 => "South", 4 => "West", 5 => "Stay")

function POMDPs.actions(pomdp::JointMeetPOMDP)
    acts_size = length(JOINTMEET_ACTIONS_DICT)
    return [action_from_index(pomdp, ai) for ai in 1:acts_size^pomdp.num_agents]
end

function POMDPs.actionindex(pomdp::JointMeetPOMDP, a::Vector{Int})
    for ai in a
        @assert ai >= 1 "Invalid action index"
        @assert ai <= length(JOINTMEET_ACTIONS_DICT) "Invalid action index"
    end
    return LinearIndices(Tuple(length(JOINTMEET_ACTIONS_DICT) for _ in 1:pomdp.num_agents))[a...]
end
POMDPs.actionindex(::JointMeetPOMDP, ::Tuple{}) = throw(ArgumentError("Invalid action tuple (empty)"))
function POMDPs.actionindex(pomdp::JointMeetPOMDP, a::Tuple{Vararg{Int}})
    return POMDPs.actionindex(pomdp, collect(a))
end
function POMDPs.actionindex(pomdp::JointMeetPOMDP, a::Tuple{Vararg{Symbol}})
    return POMDPs.actionindex(pomdp, [JOINTMEET_ACTIONS_DICT[ai] for ai in a])
end 

function action_from_index(pomdp::JointMeetPOMDP, ai::Int)
    @assert ai >= 1 "Invalid action index"
    @assert ai <= length(JOINTMEET_ACTIONS_DICT)^pomdp.num_agents "Invalid action index"
    action_card_ind = CartesianIndices(Tuple(length(JOINTMEET_ACTIONS_DICT) for _ in 1:pomdp.num_agents))[ai]
    return Vector{Int}(collect(Tuple(action_card_ind)))
end
