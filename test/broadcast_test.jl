
@testset "BroadcastChannelPOMDP" begin

    @testset "Constructors" begin
        pomdp = BroadcastChannelPOMDP()
        @test pomdp isa BroadcastChannelPOMDP
        pomdp = BroadcastChannelPOMDP(; num_agents=3, buffer_fill_prob=[0.9])
        @test pomdp.num_agents == 3
        pomdp = BroadcastChannelPOMDP(; observation_agent=1)
        @test pomdp.observation_agent == 1
        
        @test_throws ArgumentError BroadcastChannelPOMDP(num_agents=3, observation_agent=4)
        @test_throws ArgumentError BroadcastChannelPOMDP(num_agents=3, observation_agent=-1)
        @test_throws ArgumentError BroadcastChannelPOMDP(num_agents=3, buffer_fill_prob=[0.5, 0.5])
        @test_throws ArgumentError BroadcastChannelPOMDP(num_agents=3, correct_obs_prob=[0.5, 0.5])
    end

    pomdp = BroadcastChannelPOMDP()
    pomdp3 = BroadcastChannelPOMDP(; num_agents=3, buffer_fill_prob=[0.9, 0.1, 0.1])

    @testset "States" begin
        s = states(pomdp)
        @test length(s) == 2^pomdp.num_agents
        @test all(state isa Vector{BroadcastState} for state in s)
        @test stateindex(pomdp, [BufferEmpty, BufferEmpty]) == 1
        @test stateindex(pomdp, [BufferFull, BufferFull]) == 4
        
        s = states(pomdp3)
        @test length(s) == 2^pomdp3.num_agents
        @test all(state isa Vector{BroadcastState} for state in s)
        @test stateindex(pomdp3, [BufferEmpty, BufferEmpty, BufferEmpty]) == 1
        @test stateindex(pomdp3, [BufferFull, BufferFull, BufferFull]) == 8
        
        init_state = initialstate(pomdp)
        @test init_state isa POMDPTools.SparseCat
        @test length(init_state.vals) == 1
        @test init_state.probs[1] == 1.0
        
        @test POMDPTools.Testing.has_consistent_initial_distribution(pomdp)
    end

    @testset "Actions" begin
        a = actions(pomdp)
        @test length(a) == 2^pomdp.num_agents
        @test all(action isa Vector{BroadcastAction} for action in a)
        @test actionindex(pomdp, [BroadcastSend, BroadcastSend]) == 1
        @test actionindex(pomdp, [BroadcastListen, BroadcastListen]) == 4
        
        a = actions(pomdp3)
        @test length(a) == 2^pomdp3.num_agents
        @test all(action isa Vector{BroadcastAction} for action in a)
        @test actionindex(pomdp3, [BroadcastSend, BroadcastSend, BroadcastSend]) == 1
        @test actionindex(pomdp3, [BroadcastListen, BroadcastListen, BroadcastListen]) == 8
    end

    @testset "Transitions" begin
        t = transition(pomdp, [BufferEmpty, BufferFull], [BroadcastListen, BroadcastSend])
        @test t isa POMDPTools.SparseCat
        s_idx = findfirst(==([BufferEmpty, BufferFull]), t.vals)
        @test t.probs[s_idx] == (1.0 - pomdp.buffer_fill_prob[1]) * (pomdp.buffer_fill_prob[2])
        s_idx = findfirst(==([BufferFull, BufferEmpty]), t.vals)
        @test t.probs[s_idx] == pomdp.buffer_fill_prob[1] * (1 - pomdp.buffer_fill_prob[2])
        s_idx = findfirst(==([BufferFull, BufferFull]), t.vals)
        @test t.probs[s_idx] == pomdp.buffer_fill_prob[1] * pomdp.buffer_fill_prob[2]
        s_idx = findfirst(==([BufferEmpty, BufferEmpty]), t.vals)
        @test t.probs[s_idx] == (1.0 - pomdp.buffer_fill_prob[1]) * (1.0 - pomdp.buffer_fill_prob[2])
        
        t = transition(pomdp, [BufferFull, BufferFull], [BroadcastSend, BroadcastSend])
        @test t isa POMDPTools.SparseCat
        @test length(t.vals) == 1
        @test t.vals[1] == [BufferFull, BufferFull]
        @test t.probs[1] == 1.0
        
        @test POMDPTools.Testing.has_consistent_transition_distributions(pomdp)
    end

    @testset "Observations" begin
        pomdp = BroadcastChannelPOMDP(; correct_obs_prob=[0.8, 0.9])
        o = observations(pomdp)
        @test length(o) == 5^pomdp.num_agents
        @test all(obs isa Vector{Tuple{BroadcastObservation,BroadcastState}} for obs in o)
        
        @test obsindex(pomdp, observations(pomdp)[1]) == 1
        @test obsindex(pomdp, observations(pomdp)[end]) == length(observations(pomdp))
        obs_idxs = rand(1:length(observations(pomdp)), 10)
        @test all(obsindex(pomdp, observations(pomdp)[oi]) == oi for oi in obs_idxs)
        
        o3 = observations(pomdp3)
        @test length(o3) == 5^pomdp3.num_agents
        @test all(obs isa Vector{Tuple{BroadcastObservation,BroadcastState}} for obs in o3)
        
        pomdp_designated = BroadcastChannelPOMDP(; observation_agent=1)
        o_designated = observations(pomdp_designated)
        @test length(o_designated) == 5
        @test all(obs isa Vector{Tuple{BroadcastObservation,BroadcastState}} for obs in o_designated)
        
        @test_throws MethodError observation(pomdp, actions(pomdp)[1], states(pomdp)([1]))
        
        @test obsindex(pomdp, observations(pomdp)[1]) == 1
        @test obsindex(pomdp, observations(pomdp)[end]) == length(observations(pomdp))
        obs_idxs = rand(1:length(observations(pomdp)), 10)
        @test all(obsindex(pomdp, observations(pomdp)[oi]) == oi for oi in obs_idxs)
       
        o = observation(pomdp, [BufferEmpty, BufferEmpty], [BroadcastListen, BroadcastListen], [BufferEmpty, BufferEmpty])
        @test o isa POMDPTools.SparseCat
        @test length(o.vals) == 1
        @test o.probs[1] == 1.0
        @test o.vals[1][1] == (BroadcastNothing, BufferEmpty)
        @test o.vals[1][2] == (BroadcastNothing, BufferEmpty)
        
        o = observation(pomdp, [BufferFull, BufferEmpty], [BroadcastSend, BroadcastSend], [BufferFull, BufferFull])
        @test length(o.vals) == 2
        o_idx = findfirst(x -> x[1] == (BroadcastCollision, BufferFull), o.vals)
        @test o.probs[o_idx] == (1.0 - pomdp.correct_obs_prob[1])
        o_idx = findfirst(x -> x[1] == (BroadcastSuccess, BufferFull), o.vals)
        @test o.probs[o_idx] == pomdp.correct_obs_prob[1]
        
        o = observation(pomdp, [BufferFull, BufferFull], [BroadcastSend, BroadcastSend], [BufferFull, BufferFull])
        o_idxs = findall(x -> x[1] == (BroadcastSuccess, BufferFull), o.vals)
        @test length(o_idxs) == 2
        o_idx = findfirst(x -> x[2] == (BroadcastSuccess, BufferFull), o.vals[o_idxs])
        @test o.probs[o_idxs[o_idx]] ≈ (1 - pomdp.correct_obs_prob[1]) * (1 - pomdp.correct_obs_prob[2])
        o_idx = findfirst(x -> x[2] == (BroadcastCollision, BufferFull), o.vals[o_idxs])
        @test o.probs[o_idxs[o_idx]] ≈ (1 - pomdp.correct_obs_prob[1]) * pomdp.correct_obs_prob[2]
        
        o_idxs = findall(x -> x[1] == (BroadcastCollision, BufferFull), o.vals)
        @test length(o_idxs) == 2
        o_idx = findfirst(x -> x[2] == (BroadcastSuccess, BufferFull), o.vals[o_idxs])
        @test o.probs[o_idxs[o_idx]] ≈ pomdp.correct_obs_prob[1] * (1 - pomdp.correct_obs_prob[2])
        o_idx = findfirst(x -> x[2] == (BroadcastCollision, BufferFull), o.vals[o_idxs])
        @test o.probs[o_idxs[o_idx]] ≈ pomdp.correct_obs_prob[1] * pomdp.correct_obs_prob[2]
        
        @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
    end
    
    @testset "Rewards" begin
        r = reward(pomdp, [BufferEmpty, BufferEmpty], [BroadcastListen, BroadcastSend])
        @test r == 0.0
        r = reward(pomdp, [BufferFull, BufferFull], [BroadcastSend, BroadcastSend])
        @test r == 0.0
        r = reward(pomdp, [BufferFull, BufferFull], [BroadcastSend, BroadcastListen])
        @test r == pomdp.success_reward
        r = reward(pomdp, [BufferEmpty, BufferFull], [BroadcastSend, BroadcastListen])
        @test r == 0.0
    end
    
    @testset "Simulation" begin
        pomdp = BroadcastChannelPOMDP(num_agents=2)
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
        
        pomdp = BroadcastChannelPOMDP(; observation_agent=1)
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
    end
    
    @testset "Visualization" begin
        pomdp = BroadcastChannelPOMDP(; observation_agent=1)
        b = initialstate(pomdp)
        s = rand(b)
        a = rand(actions(pomdp))
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
        
        pomdp = BroadcastChannelPOMDP(; num_agents=3, buffer_fill_prob=[0.8, 0.1, 0.1], observation_agent=0)
        b = initialstate(pomdp)
        s = rand(b)
        a = rand(actions(pomdp))
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
    end
end
