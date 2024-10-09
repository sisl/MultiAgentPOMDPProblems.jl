function POMDPs.actions(::BoxPushPOMDP)
    acts_size = length(BOX_PUSH_ACTIONS)
    return vec([[ci.I...] for ci in CartesianIndices((acts_size, acts_size))])
 end
 
 function POMDPs.actionindex(::BoxPushPOMDP, a::Tuple{Int, Int})
    return actionindex(pomdp, [a[1], a[2]])
 end
 function POMDPs.actionindex(::BoxPushPOMDP, a::Vector{Int})
     acts_size = length(BOX_PUSH_ACTIONS)
     @assert all(1 <= ai <= acts_size for ai in a) "Invalid action"
     return LinearIndices((acts_size, acts_size))[a...]
 end
 
 function POMDPs.actionindex(pomdp::BoxPushPOMDP, a::Vector{Symbol})
    return POMDPs.actionindex(pomdp, (a[1], a[2]))
 end
 function POMDPs.actionindex(pomdp::BoxPushPOMDP, a::Tuple{Symbol, Symbol})
     @assert a[1] in keys(BOX_PUSH_ACTIONS) "Invalid action symbol"
     @assert a[2] in keys(BOX_PUSH_ACTIONS) "Invalid action symbol"
     return POMDPs.actionindex(pomdp, [BOX_PUSH_ACTIONS[a[1]], BOX_PUSH_ACTIONS[a[2]]])
 end
 
 function action_name(pomdp::BoxPushPOMDP, ai::Int)
     a = actions(pomdp)[ai]
     return Tuple(BOX_PUSH_ACTION_NAMES[ai] for ai in a)
 end
 
 function list_actions(pomdp::BoxPushPOMDP)
     println("Actions:")
     for ai in 1:length(actions(pomdp))
         println("  $ai: $(action_name(pomdp, ai))")
     end
 end
