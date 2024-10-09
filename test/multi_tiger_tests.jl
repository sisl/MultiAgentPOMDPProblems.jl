@testset "MultiTigerPOMDP" begin

    @testset "Constructors" begin
        pomdp = MultiTigerPOMDP()
        @test pomdp isa MultiTigerPOMDP
        pomdp = MultiTigerPOMDP(; num_agents = 3)
        @test pomdp.num_agents == 3
        pomdp = MultiTigerPOMDP(; observation_agent=1)
        @test pomdp.observation_agent == 1
    end
    pomdp = MultiTigerPOMDP()
    
    @testset "States" begin
        @test states(pomdp) == [TigerLeft, TigerRight]
        @test stateindex(pomdp, TigerLeft) == 1
        @test stateindex(pomdp, TigerRight) == 2
        init_state = initialstate(pomdp)
        @test init_state isa POMDPTools.Uniform
        
        @test POMDPTools.Testing.has_consistent_initial_distribution(pomdp)
    end
        
    @testset "Actions" begin
        acts = actions(pomdp)
        @test length(acts) == 3^pomdp.num_agents
        @test all(a[1] in [OpenLeft, OpenRight, Listen] for a in acts)
        @test all(a[2] in [OpenLeft, OpenRight, Listen] for a in acts)
    end

    @testset "Transitions" begin
        t = transition(pomdp, TigerLeft, [Listen, Listen])
        @test t isa POMDPTools.SparseCat
        @test t.vals == [TigerLeft]
        @test t.probs == [1.0]
    
        t = transition(pomdp, TigerRight, [Listen, Listen])
        @test t isa POMDPTools.SparseCat
        @test t.vals == [TigerRight]
        @test t.probs == [1.0]
        
        t = transition(pomdp, TigerLeft, [Listen, OpenLeft])
        @test t isa POMDPTools.SparseCat
        @test t.vals == [TigerLeft, TigerRight]
        @test t.probs == [0.5, 0.5]
        
        @test POMDPTools.Testing.has_consistent_transition_distributions(pomdp)
    end
       
    @testset "Observations" begin
        obs = observations(pomdp)
        @test length(obs) == 2^pomdp.num_agents
        @test all(o[1] in [HearLeft, HearRight] for o in obs)
        @test all(o[2] in [HearLeft, HearRight] for o in obs)
        
        @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        @test pomdp.observation_agent == 0
        o = observation(pomdp, [Listen, Listen], TigerLeft)
        @test o isa POMDPTools.SparseCat
        hear_left_left_index = findfirst(x -> x == [HearLeft, HearLeft], o.vals)
        @test o.probs[hear_left_left_index] ≈ pomdp.correct_obs_prob^2
        hear_left_right_index = findfirst(x -> x == [HearLeft, HearRight], o.vals)
        @test o.probs[hear_left_right_index] ≈ (pomdp.correct_obs_prob * (1 - pomdp.correct_obs_prob))
        hear_right_right_index = findfirst(x -> x == [HearRight, HearRight], o.vals)
        @test o.probs[hear_right_right_index] ≈ (1 - pomdp.correct_obs_prob)^2
        o = observation(pomdp, [Listen, OpenLeft], TigerLeft)
        @test o isa POMDPTools.Uniform
        
        
        pomdp = MultiTigerPOMDP(; observation_agent = 1)
        o = observation(pomdp, [Listen, Listen], TigerLeft)
        hear_left_index = findfirst(x -> x == [HearLeft], o.vals)
        @test o.probs[hear_left_index] ≈ pomdp.correct_obs_prob
        hear_right_index = findfirst(x -> x == [HearRight], o.vals)
        @test o.probs[hear_right_index] ≈ (1 - pomdp.correct_obs_prob)
        o = observation(pomdp, [Listen, OpenLeft], TigerLeft)
        @test o isa POMDPTools.Uniform
    end
    
    @testset "Rewards" begin
        r = reward(pomdp, TigerLeft, [Listen, Listen])
        @test r == pomdp.listen_cost * pomdp.num_agents
    
        r = reward(pomdp, TigerLeft, [Listen, OpenLeft])
        @test r == (pomdp.listen_cost + pomdp.open_wrong_penalty)
        
        r = reward(pomdp, TigerLeft, [OpenLeft, OpenLeft])
        @test r == pomdp.open_wrong_penalty / pomdp.num_agents
        
        r = reward(pomdp, TigerLeft, [Listen, OpenRight])
        @test r == (pomdp.listen_cost + pomdp.open_correct_reward)
        
        r = reward(pomdp, TigerLeft, [OpenRight, OpenRight])
        @test r == pomdp.open_correct_reward * pomdp.num_agents
    end
   
    @testset "Simulation Step" begin
        pomdp = MultiTigerPOMDP()
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
        
        pomdp = MultiTigerPOMDP(; observation_agent=1)
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
    end

    @testset "Visualization" begin
        pomdp = MultiTigerPOMDP()
        b = initialstate(pomdp)
        s = rand(b)
        a = [1, 1]
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
        
        pomdp = MultiTigerPOMDP(; num_agents=3)
        b = initialstate(pomdp)
        s = rand(b)
        a = [:OpenLeft, :Listen, :OpenRight]
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
    end
    
end
