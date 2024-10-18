using Test

using POMDPs
using POMDPTools
using MetaGraphs
using Random
using Aqua
using MultiAgentPOMDPProblems

const RUN_TESTS = isempty(ARGS) ? ["all"] : ARGS

@testset verbose=true "MultiAgentPOMDPProblems" begin
    if "MultiTigerPOMDP" in RUN_TESTS || "all" in RUN_TESTS
        @info "Running MultiTigerPOMDP tests"
        include("multi_tiger_tests.jl")
    end
    
    if "BroadcastChannelPOMDP" in RUN_TESTS || "all" in RUN_TESTS
        @info "Running BroadcastChannelPOMDP tests"
        include("broadcast_test.jl")
    end
    
    if "JointMeetPOMDP" in RUN_TESTS || "all" in RUN_TESTS
        @info "Running JointMeetPOMDP tests"
        include("jointmeet_test.jl")
    end
    
    if "StochasticMarsPOMDP" in RUN_TESTS || "all" in RUN_TESTS
        @info "Running StochasticMarsPOMDP tests"
        include("stochastic_mars_test.jl")
    end
    
    if "WirelessPOMDP" in RUN_TESTS || "all" in RUN_TESTS
        @info "Running WirelessPOMDP tests"
        include("wireless_network_test.jl")
    end
    
    if "BoxPushPOMDP" in RUN_TESTS || "all" in RUN_TESTS
        @info "Running BoxPushPOMDP tests"
        include("boxpush_test.jl")
    end
    
    Aqua.test_all(MultiAgentPOMDPProblems)
    
end
