@testset "BoxPushPOMDP" begin
    
    @testset "Constructors" begin
        pomdp = BoxPushPOMDP()
        @test pomdp isa BoxPushPOMDP
        @test discount(pomdp) == 0.9
        @test pomdp.map_option == 1
        @test pomdp.transition_prob == 0.9
        @test pomdp.small_box_goal_reward == 10.0
        @test pomdp.large_box_goal_reward == 100.0
        @test pomdp.step_penalty == -0.1
        @test pomdp.wall_penalty == -5.0
        @test pomdp.observation_agent == 0
        
        
        pomdp1 = BoxPushPOMDP(; map_option=1)
        @test get_prop(pomdp1.mg, :num_grid_pos) == 12
        @test all(get_prop(pomdp1.mg, :small_box_goals) .== [1, 4])
        @test all(get_prop(pomdp1.mg, :large_box_goals) .== [2])
        @test all(get_prop(pomdp1.mg, :agent_start_pos) .== (9, 12))
        @test all(get_prop(pomdp1.mg, :small_box_pos) .== [5, 8])
        @test all(get_prop(pomdp1.mg, :large_box_pos) .== [6])
        
        pomdp2 = BoxPushPOMDP(; map_option=2)
        @test get_prop(pomdp2.mg, :num_grid_pos) == 24
        @test get_prop(pomdp2.mg, :small_box_pos) == [14, 17]
        @test get_prop(pomdp2.mg, :large_box_pos) == [15]
        @test get_prop(pomdp2.mg, :agent_start_pos) == (20, 23)
        @test get_prop(pomdp2.mg, :small_box_goals) == [1, 6]
        @test get_prop(pomdp2.mg, :large_box_goals) == [3]
        
        
        @test_throws ArgumentError BoxPushPOMDP(; map_option=3)
        @test_throws ArgumentError BoxPushPOMDP(; discount_factor=1.5)
        @test_throws ArgumentError BoxPushPOMDP(; transition_prob=1.5)
        @test_throws ArgumentError BoxPushPOMDP(; observation_agent=3)
    end
    
    pomdp_1 = BoxPushPOMDP(; map_option=1, observation_agent=0)
    pomdp_2 = BoxPushPOMDP(; map_option=2, observation_agent=0)
    
    init_state_1 = BoxPushState([5, 8], [6], (9, 12), (:east, :west))
    init_state_2 = BoxPushState([14, 17], [15], (20, 23), (:east, :west))
    
    @testset "States" begin
        @test length(pomdp_1) == 4*4*3 + 4*4*2 + 4*4*1 + 3
        @test length(pomdp_2) == 14*(14-1)*4^2 * 4*4*2 + 3 + 1
        
        s1 = BoxPushState([1, 4], [2], (9, 12), (:east, :west))
        s2 = BoxPushState([1, 4], [2], (9, 12), (:east, :west))
        @test s1 == s2
        @test s1 != BoxPushState([1, 4], [2], (9, 12), (:east, :east))
        
        s1_map_fcn = MultiAgentPOMDPProblems.map_string_to_state(pomdp_1, "bBBb\noooo\n1oo2", (:east, :west))
        @test s1_map_fcn == s1
        
        @test stateindex(pomdp_1, pomdp_1[1]) == 1
        @test stateindex(pomdp_1, pomdp_1[37]) == 37
        @test stateindex(pomdp_1, pomdp_1[end]) == 99
        
        @test stateindex(pomdp_2, pomdp_2[1]) == 1
        @test stateindex(pomdp_2, pomdp_2[51]) == 51
        @test stateindex(pomdp_2, pomdp_2[end]) == 93188
        
        init_state_d = initialstate(pomdp_1)
        @test init_state_d isa Deterministic
        @test init_state_d.val == init_state_1
            
        init_state_d = initialstate(pomdp_2)
        @test init_state_d isa Deterministic
        @test init_state_d.val == init_state_2
    end
    
    @testset "Actions" begin
        @test length(actions(pomdp_1)) == 16
        @test length(actions(pomdp_2)) == 16
        
        @test actions(pomdp_1)[1] isa Vector{Int}
        @test length(actions(pomdp_2)[1]) == 2
        @test actionindex(pomdp_1, (:turn_left, :turn_left)) == 1
        @test actionindex(pomdp_1, (:stay, :stay)) == 16
        @test actionindex(pomdp_1, [1, 1]) == 1
        @test actionindex(pomdp_1, [4, 4]) == 16
    end
    
    ori_dict = MultiAgentPOMDPProblems.AGENT_ORIENTATIONS
    str_state = MultiAgentPOMDPProblems.map_string_to_state
    
    @testset "Transitions" begin
        
        @test get_prop(pomdp_1.mg, :large_box_goals) == [2]
        @test get_prop(pomdp_2.mg, :large_box_goals) == [3]
        
        # If in goal state, we transition to initial state
        s = pomdp_1.box_goal_states[:small_1_at_goal]
        td = transition(pomdp_1, s, (:turn_left, :turn_left))
        @test td.val == initialstate(pomdp_1).val

        s = pomdp_2.box_goal_states[:small_at_top]
        td = transition(pomdp_2, s, (:turn_left, :move_forward))
        @test td.val == initialstate(pomdp_2).val
        
        # Robot's can't move to the top/goal location (only possible with map_option=2)
        s = str_state(pomdp_2, "goGGog\n12oooo\nobBBbo\noooooo", (:north, :north))
        td = transition(pomdp_2, s, (:move_forward, :move_forward))
        @test td isa SparseCat
        @test length(td.vals) == 1
        @test td.vals[1] isa BoxPushState
        @test td.vals[1].agent_pos == (7, 8)
        @test td.vals[1].agent_orientation == (ori_dict[:north], ori_dict[:north])
        
        # Robots can't push large box by themselves
        s = str_state(pomdp_1, "gGGg\nbBBb\no12o\n", (:east, :north))
        td = transition(pomdp_1, s, (:turn_left, :move_forward))
        @test td isa SparseCat
        @test length(td.vals) == 2 # Whether the left robot turned or not
        @test all(val.agent_pos == (10, 11) for val in td.vals)
        s_idx = findfirst(x -> x.agent_orientation == (ori_dict[:north], ori_dict[:north]), td.vals)
        @test td.probs[s_idx] == pomdp_1.transition_prob
        
        # Robots can't push small boxes east/west respectively (map_option=2)
        s = str_state(pomdp_2, "goGGog\n1boob2\nooBBoo\noooooo", (:east, :west))
        td = transition(pomdp_2, s, (:move_forward, :move_forward))
        @test td isa SparseCat
        @test length(td.vals) == 1 # Whether the left robot turned or not
        @test all(val.agent_pos == (7, 12) for val in td.vals)
       
        # Actions of turn_left, turn_right, robot(s) don't move
        s = init_state_1
        td = transition(pomdp_1, s, (:turn_left, :turn_right))
        @test all(val.agent_pos == init_state_1.agent_pos for val in td.vals)
        s = init_state_2
        td = transition(pomdp_2, s, (:turn_right, :stay))
        @test all(val.agent_pos == init_state_2.agent_pos for val in td.vals)
        
        # Robots can't move past each other (can't swap positions)
        s = str_state(pomdp_1, "gGGg\nbBBb\no12o\n", (:east, :west))
        td = transition(pomdp_1, s, (:move_forward, :move_forward))
        @test td isa SparseCat
        @test length(td.vals) == 1
        @test td.vals[1].agent_pos == (10, 11)
        
        # Test movement of large box
        s = str_state(pomdp_1, "gGGg\nbBBb\no12o\n", (:north, :north))
        td = transition(pomdp_1, s, (:move_forward, :move_forward))
        @test td isa SparseCat
        @test length(td.vals) == 2 # Both move with tp*tp prob or one does and it stays
        s_idx = findfirst(x -> x.agent_pos == (10, 11), td.vals)
        @test td.probs[s_idx] == (1 - pomdp_1.transition_prob^2)
        s_idx = findfirst(x -> x == pomdp_1.box_goal_states[:large_1_at_goal], td.vals)
        @test td.probs[s_idx] == pomdp_1.transition_prob^2
        
        s = str_state(pomdp_2, "goGGog\noooooo\nobBBbo\noo12oo", (:north, :north))
        td = transition(pomdp_2, s, (:move_forward, :move_forward))
        @test td isa SparseCat
        @test length(td.vals) == 2
        s_idx = findfirst(x -> x.agent_pos == (21, 22), td.vals)
        @test td.probs[s_idx] == (1 - pomdp_2.transition_prob^2)
        s_idx = findfirst(x -> x.large_box_pos == [9], td.vals)
        @test td.probs[s_idx] == pomdp_2.transition_prob^2
        
        # Test movement of small box to goal
        s = str_state(pomdp_1, "gGGg\nbBBb\n1oo2\n", (:north, :north))
        td = transition(pomdp_1, s, (:move_forward, :move_forward))
        @test td isa SparseCat
        @test length(td.vals) == 3
        s_idx = findfirst(x -> x.agent_pos == (9, 12), td.vals)
        @test td.probs[s_idx] == (1 - pomdp_1.transition_prob)^2
        s_idx = findfirst(x -> x == pomdp_1.box_goal_states[:small_2_at_goal], td.vals)
        @test td.probs[s_idx] == pomdp_1.transition_prob^2
        s_idx = findfirst(x -> x == pomdp_1.box_goal_states[:small_1_at_goal], td.vals)
        @test td.probs[s_idx] == 2 * pomdp_1.transition_prob * (1 - pomdp_1.transition_prob)
        
        # Test movement of small box to top
        s = str_state(pomdp_2, "goGGog\noboobo\no1BB2o\noooooo", (:north, :north))
        td = transition(pomdp_2, s, (:move_forward, :move_forward))
        @test td isa SparseCat
        @test length(td.vals) == 2
        s_idx = findfirst(x -> x.agent_pos == (14, 17), td.vals)
        @test td.probs[s_idx] == (1 - pomdp_2.transition_prob)^2
        s_idx = findfirst(x -> x == pomdp_2.box_goal_states[:small_at_top], td.vals)
        @test td.probs[s_idx] == 1 - (1 - pomdp_2.transition_prob)^2
       
        POMDPTools.Testing.has_consistent_transition_distributions(pomdp_1)
    end
    
    obs_dict = MultiAgentPOMDPProblems.BOX_PUSH_OBSERVATIONS
    
    @testset "Observations" begin
        pomdp_1_0 = BoxPushPOMDP(; map_option=1, observation_agent=0)
        pomdp_2_0 = BoxPushPOMDP(; map_option=2, observation_agent=0)
        
        obs = observations(pomdp_1_0)
        @test length(obs) == 5^2
        @test all(obi isa Vector{Int} for obi in obs)
        @test all(all(1 <= obij <= 5 for obij in obi) for obi in obs)
        
        obs = observations(pomdp_2_0)
        @test length(obs) == 5^2
        @test all(obi isa Vector{Int} for obi in obs)
        @test all(all(1 <= obij <= 5 for obij in obi) for obi in obs)
        
        s = str_state(pomdp_1_0, "gGGg\nbBBb\n1oo2\n", (:east, :west))
        od = observation(pomdp_1_0, (:turn_left, :turn_left), s)
        @test od isa Deterministic
        @test od.val == [obs_dict[:empty], obs_dict[:empty]]

        s = str_state(pomdp_1_0, "gGGg\nbBBb\n1oo2\n", (:north, :south))
        od = observation(pomdp_1_0, (:turn_left, :turn_left), s)
        @test od.val == [obs_dict[:small_box], obs_dict[:wall]]
        
        s = str_state(pomdp_2_0, "goGGog\noooooo\nobBBbo\noo12oo", (:east, :north))
        od = observation(pomdp_2_0, (:turn_left, :turn_left), s)
        @test od isa Deterministic
        @test od.val == [obs_dict[:agent], obs_dict[:large_box]]
        
        
        pomdp_1_1 = BoxPushPOMDP(; map_option=1, observation_agent=1)
        pomdp_2_2 = BoxPushPOMDP(; map_option=2, observation_agent=2)
        
        obs = observations(pomdp_1_1)
        @test length(obs) == 5
        @test all(obi isa Vector{Int} for obi in obs)
        @test all(all(1 <= obij <= 5 for obij in obi) for obi in obs)
            
        obs = observations(pomdp_2_2)
        @test length(obs) == 5
        @test all(obi isa Vector{Int} for obi in obs)
        @test all(all(1 <= obij <= 5 for obij in obi) for obi in obs)
            
        s = str_state(pomdp_1_1, "gGGg\nbBBb\n1oo2\n", (:west, :west))
        od = observation(pomdp_1_1, (:turn_left, :turn_left), s)
        @test od isa Deterministic
        @test od.val == [obs_dict[:wall]]
        
        s = str_state(pomdp_2_2, "goGGog\noooooo\nobBBbo\noo12oo", (:north, :west))
        od = observation(pomdp_2_2, (:turn_left, :turn_left), s)
        @test od isa Deterministic
        @test od.val == [obs_dict[:agent]]
        
        POMDPTools.Testing.has_consistent_observation_distributions(pomdp_1_0)
        POMDPTools.Testing.has_consistent_observation_distributions(pomdp_1_1)
    end
    
    @testset "Rewards" begin
        step_p = pomdp_1.step_penalty
        small_r = pomdp_1.small_box_goal_reward
        large_r = pomdp_1.large_box_goal_reward
        hit_p = pomdp_1.wall_penalty
        
        # Test step and no other penalty/reward
        s = str_state(pomdp_1, "gGGg\nbBBb\n1oo2\n", (:east, :west))
        @test isapprox(reward(pomdp_1, s, (:turn_left, :turn_left)), 2 * step_p)
        
        s = str_state(pomdp_1, "gGGg\nbBBb\n1oo2\n", (:east, :west))
        @test isapprox(reward(pomdp_1, s, (:move_forward, :move_forward)), 2 * step_p)
        
        # Test small box reward
        s = str_state(pomdp_1, "gGGg\nbBBb\n1oo2\n", (:north, :west))
        @test isapprox(reward(pomdp_1, s, (:move_forward, :move_forward)), small_r + 2 * step_p)
        
        s = str_state(pomdp_1, "gGGg\nbBBb\n1oo2\n", (:north, :north))
        @test isapprox(reward(pomdp_1, s, (:move_forward, :move_forward)), 2 * small_r + 2 * step_p)
        
        # Test large box reward
        s = str_state(pomdp_1, "gGGg\nbBBb\no12o\n", (:north, :north))
        @test isapprox(reward(pomdp_1, s, (:move_forward, :move_forward)), large_r + 2 * step_p)
        
        # Hit wall penalty
        s = str_state(pomdp_1, "gGGg\nbBBb\n1oo2\n", (:east, :east))
        @test isapprox(reward(pomdp_1, s, (:move_forward, :move_forward)), hit_p + 2 * step_p)
        
        # Hit agent penalty
        s = str_state(pomdp_1, "gGGg\nbBBb\n12oo\n", (:north, :west))
        @test isapprox(reward(pomdp_1, s, (:move_forward, :move_forward)), small_r + hit_p + 2 * step_p)
        
        # Hit each other
        s = str_state(pomdp_1, "gGGg\nbBBb\no12o\n", (:east, :west))
        @test isapprox(reward(pomdp_1, s, (:move_forward, :move_forward)), hit_p + hit_p + 2 * step_p)
        
        # Try to move large box alone
        s = str_state(pomdp_1, "gGGg\nbBBb\no12o\n", (:north, :north))
        @test isapprox(reward(pomdp_1, s, (:turn_left, :move_forward)), hit_p + 2 * step_p)
        
        # Try to move small box through wall
        s = str_state(pomdp_2, "goGGog\noooooo\nb1BB2b\noooooo", (:west, :east))
        @test isapprox(reward(pomdp_2, s, (:move_forward, :move_forward)), 2 * hit_p + 2 * step_p)
        
        # Move to occupied space
        s = str_state(pomdp_2, "goGGog\noooooo\nobBBbo\no12ooo", (:north, :west))
        @test isapprox(reward(pomdp_2, s, (:move_forward, :move_forward)), hit_p + 2 * step_p)
        
        # Move small to top (not goal)
        s = str_state(pomdp_2, "goGGog\noboobo\no1BB2o\noooooo", (:north, :north))
        @test isapprox(reward(pomdp_2, s, (:move_forward, :move_forward)), 2 * step_p)
        
    end
    
    @testset "Simulation" begin
        pomdp_sim1 = BoxPushPOMDP()
        s1 = initialstate(pomdp_sim1)
        rng1 = MersenneTwister(42)
        policy1 = RandomPolicy(pomdp_sim1; rng=rng1, updater=DiscreteUpdater(pomdp_sim1))
        
        sim1 = HistoryRecorder(; rng=rng1,max_steps=10)
        simulate(sim1, pomdp_sim1, policy1)
        
        pomdp_sim2 = BoxPushPOMDP(; observation_agent=1)
        s2 = initialstate(pomdp_sim2)
        rng2 = MersenneTwister(42)
        policy2 = RandomPolicy(pomdp_sim2; rng=rng2, updater=DiscreteUpdater(pomdp_sim2))
        
        sim2 = HistoryRecorder(; rng=rng2,max_steps=10)
        simulate(sim2, pomdp_sim2, policy2)
    end
    
    @testset "Visualization" begin
        pomdp = BoxPushPOMDP(; map_option=2)
        b = initialstate(pomdp)
        s = b.val
        step = (s=s, b=b)
        plt = render(pomdp, step)
        up = DiscreteUpdater(pomdp)
        act = (:move_forward, :move_forward)
        plt = render(pomdp, (s=s, b=b, a=act))
        
        (sp, o, r) = @gen(:sp, :o, :r)(pomdp, s, act)
        bp = SparseCat(update(up, b, act, o))
        b = bp
        s = sp
        act = (:turn_left, :turn_right)
        plt = render(pomdp, (s=s, b=b, a=act))
        
    end
end
