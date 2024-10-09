module MultiAgentPOMDPProblems

using POMDPs
using POMDPTools

using LinearAlgebra
using MetaGraphs
using Graphs
using Plots
using LaTeXStrings
using Distributions

using ParticleFilters: ParticleCollection

include("vis_utils.jl")

# Helper function to convert linear index to multidimensional indices
function ind2sub_dims(si::Int, dims::Vector{Int})
    indices = zeros(Int, length(dims))
    si0 = si - 1  # 1-based indexing in Julia
    for i = 1:length(dims)
        d = dims[i]
        indices[i] = (si0 % d) + 1
        si0 = si0 ÷ d
    end
    return indices
end

function sub2ind_dims(indices::Vector{Int}, dims::Vector{Int})
    idx = 0
    mult = 1
    for i in 1:length(dims)
        idx += (indices[i] - 1) * mult
        mult *= dims[i]
    end
    return idx + 1  # 1-based indexing
end

# Multi-agent Tiger POMDP
include("multi_tiger_pomdp.jl")

export MultiTigerPOMDP
export TigerState, TigerLeft, TigerRight
export TigerAction, OpenLeft, OpenRight, Listen
export TigerObservation, HearLeft, HearRight

# Broadcast Channel POMDP
include("broadcast_channel.jl")

export BroadcastChannelPOMDP
export BroadcastAction, BroadcastSend, BroadcastListen
export BroadcastState, BufferEmpty, BufferFull
export BroadcastObservation, BroadcastCollision, BroadcastSuccess, BroadcastNothing

# Meeting in a Grid, JointMeetPOMDP

include("JointMeet/JointMeetPOMDP.jl")

export JointMeetPOMDP, JointMeetState

include("stochastic_mars.jl")

export StochasticMarsPOMDP, StochasticMarsState

include("multi_wireless.jl")

export WirelessPOMDP
export WirelessAction, WirelessListen, WirelessSend
export WirelessSourceState, WirelessIdle, WirelessPacket
export ChannelObservation, ChannelIdle, ChannelCollision, ChannelSingleTx
export WirelessState, WirelessObservation

include("BoxPush/BoxPushPOMDP.jl")

export BoxPushPOMDP, BoxPushState

end # module
