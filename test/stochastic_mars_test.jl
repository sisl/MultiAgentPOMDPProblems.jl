
@testset "StochasticMarsPOMDP" begin

    @testset "Constructors" begin
        # Test default constructor
        pomdp = StochasticMarsPOMDP()
        @test pomdp isa StochasticMarsPOMDP
        @test pomdp.num_agents == 2
        @test pomdp.transition_prob ≈ 0.9
        @test pomdp.observation_agent == 0

        # Test custom constructor
        map_str = """ds
                     sd"""
        pomdp = StochasticMarsPOMDP(
            map_str=map_str,
            num_agents=3,
            transition_prob=0.8,
            movement_penalty=-0.2,
            observation_agent=1
        )
        @test pomdp.num_agents == 3
        @test pomdp.transition_prob ≈ 0.8
        @test pomdp.movement_penalty ≈ -0.2
        @test pomdp.observation_agent == 1
        @test get_prop(pomdp.mg, :nrows) == 2
        @test get_prop(pomdp.mg, :ncols) == 2

        # Test invalid arguments
        @test_throws ArgumentError StochasticMarsPOMDP(map_str="invalid")
        @test_throws ArgumentError StochasticMarsPOMDP(transition_prob=1.1)
        @test_throws ArgumentError StochasticMarsPOMDP(discount_factor=1.1)
    end
    
    @testset "States" begin
        pomdp = StochasticMarsPOMDP(map_str="ds\nsd")
        
        @test length(pomdp) == 16 * 16 # 4^2 positions * 4^2 experiment states
        @test pomdp[1] isa StochasticMarsState
        @test stateindex(pomdp, pomdp[1]) == 1
        @test stateindex(pomdp, pomdp[end]) == length(pomdp)

        init_dist = initialstate(pomdp)
        @test init_dist isa POMDPTools.SparseCat
        @test length(init_dist.vals) == 16 # Uniform over positions (not experiment states)
        
        s1 = StochasticMarsState((1, 2), [false, true, false, true])
        s2 = StochasticMarsState((1, 2), [false, true, false, true])
        @test s1 == s2
        
        s3 = StochasticMarsState((1, 2), [false, false, false, true])
        @test s1 != s3
        
        pomdp = StochasticMarsPOMDP(; num_agents=3, map_str="ddd\nsss\ndss")
        @test length(pomdp) == (9^3) * (2^9)
        
        # @test POMDPTools.Testing.has_consistent_initial_distribution(pomdp)
    end

    @testset "Actions" begin
        pomdp = StochasticMarsPOMDP(map_str="ds\nsd", num_agents=2)
        
        acts = actions(pomdp)
        @test length(acts) == 6^2  # 6 actions per agent

        @test acts[1] isa Vector{Int}
        
        @test actionindex(pomdp, (1, 1)) == 1
        @test actionindex(pomdp, (:north, :north)) == 1
        @test_throws AssertionError actionindex(pomdp, (0, 1))
        @test_throws AssertionError actionindex(pomdp, (:invalid, :north))
    end 
    
    @testset "Transitions" begin
        pomdp = StochasticMarsPOMDP(map_str="ds\nsd", transition_prob=0.8)
        
        pos = (1, 2)
        exp_vec = [false, false, false, false]
        
        s = StochasticMarsState(pos, exp_vec)
        t = transition(pomdp, s, [:east, :south])

        @test t isa POMDPTools.SparseCat
        @test length(t.vals) == 4

        # Check probabilities for specific transitions
        s_idx = findfirst(==(StochasticMarsState((2, 4), exp_vec)), t.vals)
        @test isapprox(t.probs[s_idx], pomdp.transition_prob^2)
        
        s_idx = findfirst(==(StochasticMarsState((1, 4), exp_vec)), t.vals)
        @test isapprox(t.probs[s_idx], pomdp.transition_prob * (1 - pomdp.transition_prob))

        # Test drill action
        s = StochasticMarsState((1, 2), exp_vec)
        t = transition(pomdp, s, [:drill, :sample])
        @test t isa POMDPTools.SparseCat
        @test length(t.vals) == 1
        @test t.vals[1] == StochasticMarsState((1, 2), [true, true, false, false])
        
        # Test reset when all rocks are experimented
        s = StochasticMarsState((4, 4), [true, true, true, false])
        t = transition(pomdp, s, [:drill, :drill])
        @test t isa POMDPTools.SparseCat
        @test all(t.probs .== 1.0 / length(t.vals))
    end
    
    @testset "Observations" begin
        pomdp = StochasticMarsPOMDP(;
            map_str="ds\nsd",
            num_agents=2,
            observation_agent=0
        )
        
        num_grids = 4 # Based on map_str        
        state_vec = [num_grids, num_grids, 2, 2] # 2 agents
        
        @test length(observations(pomdp)) == prod(state_vec)
        @test observations(pomdp)[1] isa Vector{Int}

        sp = StochasticMarsState((1, 2), [true, false, true, false])
        o = observation(pomdp, [1, 1], sp)
        @test o isa Deterministic
        exp_obs = [2, 1] # Based on sp and true = 2 and false = 1
        obs_1 = LinearIndices((num_grids, 2))[1, 2]
        obs_2 = LinearIndices((num_grids, 2))[2, 1]
        @test o.val == [obs_1, obs_2]

        # @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        # Test single agent observation
        pomdp = StochasticMarsPOMDP(;
            map_str="ds\nsd", 
            observation_agent=2
        )
        state_vec = [num_grids, 2] 
        @test length(observations(pomdp)) == prod(state_vec)
        @test observations(pomdp)[1] isa Vector{Int}
        
        o = observation(pomdp, [1, 1], sp)
        @test o isa Deterministic
        obs_2 = LinearIndices((num_grids, 2))[2, 1]
        @test o.val == [obs_2]

        # @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
    end
    
    @testset "Rewards" begin
        pomdp = StochasticMarsPOMDP(
            map_str="ds\nsd",
            movement_penalty=-0.1,
            ruin_site_penalty=-10.0,
            redundancy_penalty=-1.0,
            drill_reward=6.0,
            sample_reward=2.0
        )

        # Test movement penalty
        s = StochasticMarsState((1, 2), [false, false, false, false])
        a = [:east, :south]
        @test reward(pomdp, s, a) == pomdp.movement_penalty * 2
        
        # Test drill reward
        s = StochasticMarsState((4, 4), [false, false, false, false])
        a = [:drill, :drill]
        @test reward(pomdp, s, a) == pomdp.drill_reward

        # Test sample reward
        s = StochasticMarsState((2, 2), [false, false, false, false])
        a = [:sample, :sample]
        @test reward(pomdp, s, a) == pomdp.sample_reward

        s = StochasticMarsState((1, 1), [false, false, false, false])
        a = [:sample, :sample]
        @test reward(pomdp, s, a) == pomdp.sample_reward + pomdp.redundancy_penalty
        
        # Test ruin site penalty
        s = StochasticMarsState((2, 2), [false, false, false, false])
        a = [:drill, :drill]
        @test reward(pomdp, s, a) == pomdp.ruin_site_penalty

        # Test redundancy penalty
        s = StochasticMarsState((1, 1), [true, false, false, false])
        a = [:drill, :drill]
        @test reward(pomdp, s, a) == pomdp.redundancy_penalty * 2
        
        # Other combinations
        s = StochasticMarsState((1, 2), [false, false, false, false])
        a = [:sample, :sample]
        @test reward(pomdp, s, a) == pomdp.sample_reward * 2
        
        s = StochasticMarsState((1, 2), [false, false, false, false])
        a = [:sample, :drill]
        @test reward(pomdp, s, a) == pomdp.sample_reward + pomdp.ruin_site_penalty
        
        s = StochasticMarsState((1, 2), [true, false, false, false])
        a = [:drill, :sample]
        @test reward(pomdp, s, a) == pomdp.redundancy_penalty + pomdp.sample_reward
        
        s = StochasticMarsState((1, 2), [true, false, false, false])
        a = [:sample, :north]
        @test reward(pomdp, s, a) == pomdp.redundancy_penalty + pomdp.movement_penalty
        
        s = StochasticMarsState((2, 4), [false, false, true, false])
        a = [:east, :drill]
        @test reward(pomdp, s, a) == pomdp.movement_penalty + pomdp.ruin_site_penalty
        
        s = StochasticMarsState((2, 3), [false, true, false, false])
        a = [:east, :sample]
        @test reward(pomdp, s, a) == pomdp.movement_penalty + pomdp.sample_reward
    end
    
    @testset "Simulation" begin
        pomdp = StochasticMarsPOMDP()
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
        
        pomdp = StochasticMarsPOMDP(; observation_agent=1)
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
    end
    
    @testset "Visualization" begin
        pomdp = StochasticMarsPOMDP()
        b = initialstate(pomdp)
        s = rand(b)
        a = [1, 1]
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
        
        pomdp = StochasticMarsPOMDP(; num_agents=3, map_str="ddd\nsss\ndss")
        b = initialstate(pomdp)
        s = rand(b)
        a = [1, 1, 1]
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
    end
end
