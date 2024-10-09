@testset "JointMeetPOMDP" begin

    @testset "Constructors" begin
        pomdp = JointMeetPOMDP()
        @test pomdp isa JointMeetPOMDP
        @test pomdp.num_agents == 2
        @test pomdp.observation_option == :full
        @test pomdp.observation_agent == 0

        
        map_str::String = """oooo\noooo\noooo\noooo"""
        pomdp = JointMeetPOMDP(; num_agents=3, map_str=map_str)
        @test pomdp.num_agents == 3
        @test get_prop(pomdp.mg, :nrows) == 4
        @test get_prop(pomdp.mg, :ncols) == 4

        pomdp = JointMeetPOMDP(; observation_option=:left_and_same, observation_agent=1)
        @test pomdp.observation_option == :left_and_same
        @test pomdp.observation_agent == 1

        @test_throws ArgumentError JointMeetPOMDP(; num_agents=2, observation_agent=3)
        @test_throws ArgumentError JointMeetPOMDP(; observation_option=:invalid)
        @test_throws ArgumentError JointMeetPOMDP(; transition_prob=1.1)
        @test_throws ArgumentError JointMeetPOMDP(; discount_factor=1.1)
        @test_throws ArgumentError JointMeetPOMDP(; observation_sigma=-1.0)
    end

    pomdp = JointMeetPOMDP(; map_str="""oooo\noooo\noooo\noooo""")

    @testset "States" begin
        @test length(pomdp) == get_prop(pomdp.mg, :num_grid_pos)^pomdp.num_agents
        @test pomdp[1] isa JointMeetState
        @test stateindex(pomdp, pomdp[1]) == 1
        @test stateindex(pomdp, pomdp[end]) == length(pomdp)

        init_state = initialstate(pomdp)
        @test init_state isa POMDPTools.SparseCat
        @test length(init_state.vals) == length(pomdp) 
        
        @test POMDPTools.Testing.has_consistent_initial_distribution(pomdp)
    end
        
    @testset "Actions" begin
        acts = actions(pomdp)
        @test length(acts) == 5^pomdp.num_agents # north, east, south, west, stay per agent
        @test acts[1] isa Vector{Int}
        @test actionindex(pomdp, (1, 1)) == 1
        @test actionindex(pomdp, (:north, :north)) == 1
        @test_throws AssertionError actionindex(pomdp, (0, 1))
        @test_throws KeyError actionindex(pomdp, (:invalid, :north))
    end

    @testset "Transitions" begin
        map_str::String = """oo\noo"""
        pomdp = JointMeetPOMDP(; 
            num_agents=2, 
            map_str=map_str, 
            transition_prob=0.6, 
            transition_alternatives=:all
        )
        
        s = JointMeetState((1, 2))
        t = transition(pomdp, s, (:east, :south))
        @test t isa POMDPTools.SparseCat
        @test length(t.vals) == 9
        
        s_idx = findfirst(==(JointMeetState((2, 4))), t.vals)
        @test isapprox(t.probs[s_idx], 0.6^2)
        s_idx = findfirst(==(JointMeetState((2, 2))), t.vals)
        @test isapprox(t.probs[s_idx], 0.6 * (1 - 0.6) / 4 * 3)
        s_idx = findfirst(==(JointMeetState((3, 1))), t.vals)
        @test isapprox(t.probs[s_idx], ((1 - 0.6) / 4) ^ 2)
        
        pomdp = JointMeetPOMDP(; 
            num_agents=2, 
            map_str=map_str, 
            transition_prob=0.6, 
            transition_alternatives=:not_opposite
        )
        
        s = JointMeetState((1, 2))
        t = transition(pomdp, s, (:east, :south))
        @test t isa POMDPTools.SparseCat
        @test length(t.vals) == 9
        s_idx = findfirst(==(JointMeetState((2, 4))), t.vals)
        @test isapprox(t.probs[s_idx], 0.6^2)
        s_idx = findfirst(==(JointMeetState((2, 2))), t.vals)
        @test isapprox(t.probs[s_idx], 0.6 * (1 - 0.6) / 2  * 1)
        s_idx = findfirst(==(JointMeetState((3, 1))), t.vals)
        @test isapprox(t.probs[s_idx], ((1 - 0.6) / 2) ^ 2)
        
        pomdp = JointMeetPOMDP(; 
            num_agents=2, 
            map_str=map_str, 
            transition_prob=1.0
        )
        
        s = JointMeetState((1, 2))
        t = transition(pomdp, s, (:east, :south))
        @test t isa Deterministic
        @test t.val == JointMeetState((2, 4))
    end
    
    @testset "Observations" begin
        map_str = """ooo
                     ooo
                     ooo"""
        # Joint full observation
        pomdp = JointMeetPOMDP(; 
            map_str=map_str,
            observation_option=:full,
            observation_agent=0
        )
        @test length(observations(pomdp)) == get_prop(pomdp.mg, :num_grid_pos) ^ pomdp.num_agents
        sp = JointMeetState((1, 2))
        o = observation(pomdp, [1, 1], sp)
        @test o isa SparseCat
        @test o.vals[1] isa Vector{Int}
        @test all(o.vals[1] .== sp.r_positions)
        
        # @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        # Full observation for agent 1
        pomdp = JointMeetPOMDP(; 
            map_str=map_str,
            observation_option=:full,
            observation_agent=1
        )
        @test length(observations(pomdp)) == get_prop(pomdp.mg, :num_grid_pos) * pomdp.num_agents
        sp = JointMeetState((3, 1))
        o = observation(pomdp, [2, 3], sp)
        @test o isa SparseCat
        @test o.vals[1] isa Vector{Int}
        @test o.vals[1] == [3]
        sp = JointMeetState((1, 1))
        o = observation(pomdp, [2, 2], sp)
        @test o isa SparseCat
        @test o.vals[1] isa Vector{Int}
        @test o.vals[1] == [get_prop(pomdp.mg, :num_grid_pos) + 1]
        
        # @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        # Left and same observation (joint)
        pomdp = JointMeetPOMDP(; 
            map_str=map_str,
            observation_option=:left_and_same,
            observation_sigma=0.0,
            observation_agent=0
        )
        @test length(observations(pomdp)) == (get_prop(pomdp.mg, :ncols) * pomdp.num_agents) ^ pomdp.num_agents
        sp = JointMeetState((1, 2))
        o = observation(pomdp, [1, 1], sp)
        @test o isa SparseCat
        
        sp = JointMeetState((2, 1))
        o = observation(pomdp, [1, 2], sp)
        @test o isa SparseCat
        @test o.vals[1] isa Vector{Int}
        @test o.vals[1] == [2, 1]
        
        pomdp = JointMeetPOMDP(; 
            map_str=map_str,
            observation_option=:left_and_same,
            observation_sigma=0.2,
            observation_agent=0
        )
        o = observation(pomdp, [1, 1], sp)
        @test o isa SparseCat
        @test length(o.vals) == get_prop(pomdp.mg, :ncols) ^ pomdp.num_agents
        
        # Boundaries lr (agent 1)
        map_str = """oooxo
                     oooxo
                     oooxx
                     oooxo"""
        pomdp = JointMeetPOMDP(; 
            map_str=map_str,
            observation_option=:boundaries_lr,
            observation_agent=1
        )
        @test length(observations(pomdp)) == 4
        sp = JointMeetState((1, 1))
        ol = observation(pomdp, [1, 1], sp)
        @test ol isa SparseCat
        sp = JointMeetState((3, 1))
        or = observation(pomdp, [1, 1], sp)
        sp = JointMeetState((4, 1))
        ob = observation(pomdp, [1, 1], sp)
        sp = JointMeetState((2, 1))
        on = observation(pomdp, [1, 1], sp)
        @test length(unique([ol.vals[1], or.vals[1], ob.vals[1], on.vals[1]])) == 4
        sp = JointMeetState((5, 1))
        ol2 = observation(pomdp, [1, 1], sp)
        @test all(ol.vals[1] .== ol2.vals[1])
        
        # @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        # Boundaries lr (joint)
        pomdp = JointMeetPOMDP(; 
            map_str=map_str,
            observation_option=:boundaries_lr,
            observation_agent=0
        )
        @test length(observations(pomdp)) == 4 ^ pomdp.num_agents
        sp = JointMeetState((1, 4))
        o = observation(pomdp, [1, 1], sp)
        @test o isa SparseCat
        sp = JointMeetState((5, 8))
        o1 = observation(pomdp, [1, 1], sp)
        @test o1 isa SparseCat
        @test o1.vals[1] == o.vals[1]
        @test o1.vals[1] isa Vector{Int}
        
        # @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        # Boundaries both (agent 1)
        pomdp = JointMeetPOMDP(; 
            map_str=map_str,
            observation_option=:boundaries_both,
            observation_agent=1
        )
        @test length(observations(pomdp)) == 16
        sp = JointMeetState((1, 1)) # left, top
        olt = observation(pomdp, [1, 1], sp)
        @test olt isa SparseCat
        @test olt.vals[1] isa Vector{Int}
        sp = JointMeetState((2, 1)) # none, top
        ont = observation(pomdp, [1, 1], sp)
        sp = JointMeetState((3, 1)) # right, top
        ort = observation(pomdp, [1, 1], sp)
        sp = JointMeetState((4, 1)) # both, top
        obt = observation(pomdp, [1, 1], sp)
        sp = JointMeetState((5, 1)) # left, none
        oln = observation(pomdp, [1, 1], sp)
        sp = JointMeetState((8, 1)) # both, both
        obb = observation(pomdp, [1, 1], sp)
        @test length(unique([olt.vals[1], ont.vals[1], ort.vals[1], obt.vals[1], oln.vals[1], obb.vals[1]])) == 6
        
        # @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        # Boundaries both (joint)
        pomdp = JointMeetPOMDP(; 
            map_str=map_str,
            observation_option=:boundaries_both,
            observation_agent=0
        )
        @test length(observations(pomdp)) == 16 ^ pomdp.num_agents
        sp = JointMeetState((1, 1))
        o = observation(pomdp, [1, 1], sp)
        @test o isa SparseCat
        @test o.vals[1] isa Vector{Int}
        
        # @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
    end
   
    
    @testset "Rewards" begin
        step_penalty = -0.123
        wall_penalty = -5.432
        meet_reward = 9.876
        pomdp = JointMeetPOMDP(; 
            map_str="""oo\noo""",
            step_penalty=step_penalty,
            wall_penalty=wall_penalty,
            meet_reward=meet_reward
        )
        
        s = JointMeetState((1, 1))
        a = (:north, :north)
        @test reward(pomdp, s, a) == step_penalty + wall_penalty*2 + meet_reward
        
        s = JointMeetState((1, 2))
        a = (:east, :south)
        @test reward(pomdp, s, a) == step_penalty
        
        s = JointMeetState((1, 2))
        a = (:stay, :north)
        @test reward(pomdp, s, a) == step_penalty + wall_penalty
        
        pomdp = JointMeetPOMDP(; 
            map_str="""ooo\nooo\nooo""",
            step_penalty=step_penalty,
            wall_penalty=wall_penalty,
            meet_reward=meet_reward,
            meet_reward_locations=[1, 9]
        )
        
        s = JointMeetState((1, 1))
        a = (:north, :north)
        @test reward(pomdp, s, a) == step_penalty + wall_penalty*2 + meet_reward
        
        s = JointMeetState((2, 2))
        a = (:west, :south)
        @test reward(pomdp, s, a) == step_penalty
    end
    
    @testset "Simulation" begin
        pomdp = JointMeetPOMDP()
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
        
        pomdp = JointMeetPOMDP(; observation_agent=1)
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
    end
    
    @testset "Visualization" begin
        pomdp = JointMeetPOMDP(; observation_option=:boundaries_both, observation_agent=1)
        b = initialstate(pomdp)
        s = rand(b)
        a = rand(actions(pomdp))
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
        
        pomdp = JointMeetPOMDP(; observation_option=:full, observation_agent=0)
        pomdp = StochasticMarsPOMDP(; num_agents=3, map_str="ddd\nsss\ndss")
        b = initialstate(pomdp)
        s = rand(b)
        a = rand(actions(pomdp))
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
    end
    
end
