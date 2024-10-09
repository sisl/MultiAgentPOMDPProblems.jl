@testset "WirelessPOMDP" begin

    @testset "Constructors" begin
        # Test default constructor
        pomdp = WirelessPOMDP()
        @test pomdp isa WirelessPOMDP
        @test pomdp.num_agents == 2
        @test pomdp.discount_factor ≈ 0.99
        @test pomdp.idle_to_packet_prob ≈ 0.0470
        @test pomdp.packet_to_idle_prob ≈ 0.0741
        @test pomdp.observation_prob ≈ 0.9
        @test pomdp.observation_agent == 0

        # Test custom constructor
        pomdp = WirelessPOMDP(
            num_agents=3,
            discount_factor=0.9,
            idle_to_packet_prob=0.05,
            packet_to_idle_prob=0.07,
            observation_prob=0.8,
            observation_agent=2
        )
        @test pomdp.num_agents == 3
        @test pomdp.discount_factor ≈ 0.9
        @test pomdp.idle_to_packet_prob ≈ 0.05
        @test pomdp.packet_to_idle_prob ≈ 0.07
        @test pomdp.observation_prob ≈ 0.8
        @test pomdp.observation_agent == 2

        # Test invalid arguments
        @test_throws ArgumentError WirelessPOMDP(idle_to_packet_prob=-0.1)
        @test_throws ArgumentError WirelessPOMDP(packet_to_idle_prob=1.1)
        @test_throws ArgumentError WirelessPOMDP(observation_prob=-0.5)
        @test_throws ArgumentError WirelessPOMDP(observation_agent=4, num_agents=3)
        @test_throws ArgumentError WirelessPOMDP(discount_factor=1.1)
    end

    @testset "States" begin
        num_agents = 2
        pomdp = WirelessPOMDP(num_agents=num_agents)
        num_source_states = length(instances(WirelessSourceState))  # 2 source states
        num_queue_lengths = 4  # Queue lengths from 0 to 3
        total_states = num_source_states^num_agents * num_queue_lengths^num_agents
        @test length(pomdp) == total_states
        @test pomdp[1] isa WirelessState
        @test stateindex(pomdp, pomdp[1]) == 1
        @test stateindex(pomdp, pomdp[end]) == length(pomdp)

        init_dist = initialstate(pomdp)
        @test init_dist isa POMDPTools.SparseCat
        @test length(init_dist.vals) == num_source_states^num_agents  # All combinations of source states, queue lengths are zeros
        @test all(sum(s.queue_length) == 0 for s in init_dist.vals)

        s1 = WirelessState([WirelessIdle, WirelessPacket], [1, 2])
        s2 = WirelessState([WirelessIdle, WirelessPacket], [1, 2])
        @test s1 == s2

        s3 = WirelessState([WirelessIdle, WirelessIdle], [1, 2])
        @test s1 != s3

        @test POMDPTools.Testing.has_consistent_initial_distribution(pomdp)
    end

    @testset "Actions" begin
        num_agents = 2
        pomdp = WirelessPOMDP(num_agents=num_agents)
        num_actions_per_agent = length(instances(WirelessAction))  # 2 actions: WirelessListen and WirelessSend
        total_actions = num_actions_per_agent^num_agents
        acts = actions(pomdp)
        @test length(acts) == total_actions
        @test acts[1] isa Vector{WirelessAction}

        @test actionindex(pomdp, [WirelessListen, WirelessListen]) == 1
        @test actionindex(pomdp, (1, 1)) == 1
        
        @test_throws AssertionError actionindex(pomdp, (WirelessListen, WirelessListen, WirelessListen))
        @test_throws MethodError actionindex(pomdp, (WirelessListen, 3))

        a = MultiAgentPOMDPProblems.action_from_index(pomdp, 1)
        @test a == (WirelessListen, WirelessListen)
    end

    @testset "Transitions" begin
        pomdp = WirelessPOMDP(
            num_agents=2,
            idle_to_packet_prob=0.01,
            packet_to_idle_prob=0.02
        )
        s = WirelessState([WirelessIdle, WirelessIdle], [0, 0])
        a = (WirelessListen, WirelessListen)
        sp_dist = transition(pomdp, s, a)

        @test sp_dist isa POMDPTools.SparseCat
        @test length(sp_dist.vals) > 0
        total_prob = sum(sp_dist.probs)
        @test isapprox(total_prob, 1.0; atol=1e-6)

        # Check that resulting states are valid
        for sp in sp_dist.vals
            @test length(sp.source_state) == 2
            @test all(ss in instances(WirelessSourceState) for ss in sp.source_state)
            @test length(sp.queue_length) == 2
            @test all(0 ≤ ql ≤ 3 for ql in sp.queue_length)
        end

        st = WirelessState([WirelessIdle, WirelessIdle], [0, 0])
        st_idx = findfirst(==(st), sp_dist.vals)
        @test isapprox(sp_dist.probs[st_idx], (1-0.01)*(1-0.01); atol=1e-6) # Values from above
        
        st = WirelessState([WirelessIdle, WirelessPacket], [0, 0])
        st_idx = findfirst(==(st), sp_dist.vals)
        @test isapprox(sp_dist.probs[st_idx], (1-0.01)*0.01; atol=1e-6)
        
        s = WirelessState([WirelessPacket, WirelessIdle], [0, 0])
        a = (WirelessListen, WirelessListen)
        sp_dist = transition(pomdp, s, a)
        st = WirelessState([WirelessIdle, WirelessPacket], [1, 0])
        st_idx = findfirst(==(st), sp_dist.vals)
        @test isapprox(sp_dist.probs[st_idx], 0.02*0.01; atol=1e-6) # Values from above
        
        pomdp3 = WirelessPOMDP(; num_agents=3)
        rng = MersenneTwister(42)
        for _ in 1:10
            q = rand(rng, 0:3, 3)
            s = WirelessState([WirelessIdle, WirelessIdle, WirelessIdle], q)
            a = [WirelessListen, WirelessListen, WirelessListen]
            td_dist = transition(pomdp3, s, a)
            for sp in td_dist.vals
                for (ii, qi) in enumerate(sp.queue_length)
                    @test 0 <= qi <= 3
                    @test qi == q[ii]
                end
            end
            s = WirelessState([WirelessPacket, WirelessPacket, WirelessPacket], q)
            a = [WirelessListen, WirelessListen, WirelessListen]
            td_dist = transition(pomdp3, s, a)
            for sp in td_dist.vals
                max_q = maximum(sp.queue_length)
                @test max_q <= 3
                for (ii, qi) in enumerate(sp.queue_length)
                    @test 0 <= qi <= 3
                    if q[ii] < 3
                        @test qi <= q[ii] + 1
                    else
                        @test qi == 3
                    end
                end
            end
        end
        #TODO: Expand to test for specific transition rules (collision, success, etc.)
    end

    @testset "Observations" begin
        pomdp = WirelessPOMDP(num_agents=2, observation_prob=0.8)
        s = WirelessState([WirelessIdle, WirelessPacket], [0, 1])
        a = (WirelessSend, WirelessListen)
        o_dist = observation(pomdp, a, s)

        @test o_dist isa POMDPTools.SparseCat
        @test length(o_dist.vals) == 3 ^ pomdp.num_agents
        @test o_dist.vals[1] isa Vector{WirelessObservation}
        total_prob = sum(o_dist.probs)
        @test isapprox(total_prob, 1.0; atol=1e-6)

        @test o_dist.vals[1] isa Vector{WirelessObservation}
        
        # Check that observations are valid
        for o in o_dist.vals
            @test length(o) == pomdp.num_agents
            for obs in o
                @test obs isa WirelessObservation
                @test obs.channel_observation in instances(ChannelObservation)
                @test obs.queue_empty isa Bool
            end
        end

        # Test consistent observation distributions
        @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        pomdp = WirelessPOMDP(num_agents=2, observation_prob=0.8, observation_agent=1)
        s = WirelessState([WirelessIdle, WirelessPacket], [0, 1])
        a = (WirelessSend, WirelessListen)
        o_dist = observation(pomdp, a, s)
        @test length(o_dist.vals) == 3
        @test o_dist.vals[1] isa Vector{WirelessObservation}
        @test length(o_dist.vals[1]) == 1
        @test o_dist.vals[1][1] isa WirelessObservation
        
        @test POMDPTools.Testing.has_consistent_observation_distributions(pomdp)
        
        #TODO: Expand to test for specific observations
    end

    @testset "Rewards" begin
        pomdp = WirelessPOMDP(num_agents=2)
        s = WirelessState([WirelessIdle, WirelessPacket], [2, 1])
        a = actionindex(pomdp, (WirelessSend, WirelessListen))
        r = reward(pomdp, s, a)
        @test r == -sum(s.queue_length)
        @test r == -(2 + 1)

        s = WirelessState([WirelessPacket, WirelessPacket], [3, 3])
        r = reward(pomdp, s, a)
        @test r == -6
    end
    
    @testset "Simulation Step" begin
        pomdp = WirelessPOMDP()
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
        
        pomdp = WirelessPOMDP(; observation_agent=1)
        s = initialstate(pomdp)
        rng = MersenneTwister(42)
        policy = RandomPolicy(pomdp; rng=rng, updater=DiscreteUpdater(pomdp))
        
        sim = HistoryRecorder(; rng=rng,max_steps=10)
        simulate(sim, pomdp, policy)
    end
    
    @testset "Visualization" begin
        pomdp = WirelessPOMDP(; observation_agent=1)
        b = initialstate(pomdp)
        s = rand(b)
        a = rand(actions(pomdp))
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
        
        pomdp = WirelessPOMDP(; num_agents=3, observation_agent=0)
        b = initialstate(pomdp)
        s = rand(b)
        a = rand(actions(pomdp))
        step = (s=s, b=b, a=a)
        plt = render(pomdp, step)
    end
end
