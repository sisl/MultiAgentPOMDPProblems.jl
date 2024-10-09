
function POMDPs.transition(pomdp::BoxPushPOMDP, s::BoxPushState, a::Tuple{Symbol, Symbol})
    return POMDPs.transition(pomdp, s, [BOX_PUSH_ACTIONS[a[1]], BOX_PUSH_ACTIONS[a[2]]])
end
function POMDPs.transition(pomdp::BoxPushPOMDP, s::BoxPushState, a::Vector{Int})
    # If we are at a goal state, we reset to the initial state
    if s in values(pomdp.box_goal_states)
        return initialstate(pomdp)
    end
    
    tp = pomdp.transition_prob  # Transition probability
    
    # Initialize a dictionary to hold next states and their probabilities
    next_state_probs = Dict{BoxPushState, Float64}()

    # Get possible outcomes for each agent
    possible_agent_outcomes = Vector{Vector{Tuple{Int, Int, Float64}}}(undef, 2)
    for i in 1:2
        ai = a[i]
        current_pos = s.agent_pos[i]
        current_orientation = s.agent_orientation[i]
        possible_outcomes = Vector{Tuple{Int, Int, Float64}}()

        if ai == BOX_PUSH_ACTIONS[:turn_left] || ai == BOX_PUSH_ACTIONS[:turn_right]
            # Success: orientation changes
            new_orient = new_orientation(current_orientation, ai)
            push!(possible_outcomes, (current_pos, new_orient, tp))
            # Failure: stays the same
            push!(possible_outcomes, (current_pos, current_orientation, 1 - tp))
        elseif ai == BOX_PUSH_ACTIONS[:move_forward]
            # Success: attempts to move forward
            new_pos = move_direction(pomdp, current_pos, current_orientation)
            if new_pos != current_pos
                push!(possible_outcomes, (new_pos, current_orientation, tp))
            else
                # Movement blocked (e.g., wall)
                push!(possible_outcomes, (current_pos, current_orientation, tp))
            end
            # Failure: stays in place
            push!(possible_outcomes, (current_pos, current_orientation, 1 - tp))
        elseif ai == BOX_PUSH_ACTIONS[:stay]
            # Only one possible outcome
            push!(possible_outcomes, (current_pos, current_orientation, 1.0))
        else
            error("Invalid action")
        end

        possible_agent_outcomes[i] = possible_outcomes
    end
    
    # Generate all combinations of agent outcomes
    for outcome1 in possible_agent_outcomes[1]
        pos1, orient1, prob1 = outcome1
        for outcome2 in possible_agent_outcomes[2]
            pos2, orient2, prob2 = outcome2

            total_prob = prob1 * prob2

            # Initialize positions to check for conflicts
            intended_pos1 = pos1
            intended_pos2 = pos2

            # Check for agents moving into each other's positions (swap positions) or ending up in the same position
            swap_positions = (intended_pos1 == s.agent_pos[2] && intended_pos2 == s.agent_pos[1])
            same_position = (intended_pos1 == intended_pos2)
            
            if swap_positions || same_position
                # Agents hit each other and stay in their original positions
                intended_pos1 = s.agent_pos[1]
                intended_pos2 = s.agent_pos[2]
            end

            # Deep copy box positions to avoid mutation
            new_small_box_pos = copy(s.small_box_pos)
            new_large_box_pos = copy(s.large_box_pos)

            # Handle box movements
            # Agents attempting to push small boxes
            for (agent_idx, pos, orient, ai) in zip(1:2, [intended_pos1, intended_pos2], 
                [orient1, orient2], a)
                if ai == BOX_PUSH_ACTIONS[:move_forward] && pos != s.agent_pos[agent_idx]
                    # Agent moved forward successfully
                    idx_in_small_boxes = findfirst(x -> x == pos, s.small_box_pos)
                    if idx_in_small_boxes !== nothing
                        # Agent is attempting to push a small box
                        box_movement_allowed = false
                        # Determine box movement direction based on the box and agent
                        if (agent_idx == 1 && idx_in_small_boxes == 1) && 
                            (orient == AGENT_ORIENTATIONS[:north] || 
                            orient == AGENT_ORIENTATIONS[:west])
                            # First small box can be moved up or left by agent 1
                            box_movement_allowed = true
                        elseif (agent_idx == 2 && idx_in_small_boxes == 2) && 
                            (orient == AGENT_ORIENTATIONS[:north] || 
                            orient == AGENT_ORIENTATIONS[:east])
                            # Second small box can be moved up or right by agent 2
                            box_movement_allowed = true
                        end

                        if box_movement_allowed
                            # Attempt to move the box
                            new_box_pos = move_direction(pomdp, pos, orient, false)
                            if new_box_pos != pos && !(new_box_pos in s.small_box_pos || 
                                new_box_pos in s.large_box_pos || new_box_pos == intended_pos1 || 
                                new_box_pos == intended_pos2)
                                # Move the box
                                new_small_box_pos[idx_in_small_boxes] = new_box_pos
                            else
                                # Can't move the box; agent stays in original position
                                if agent_idx == 1
                                    intended_pos1 = s.agent_pos[agent_idx]
                                else
                                    intended_pos2 = s.agent_pos[agent_idx]
                                end
                            end
                        else
                            # Can't move the box; agent stays in original position
                            if agent_idx == 1
                                intended_pos1 = s.agent_pos[agent_idx]
                            else
                                intended_pos2 = s.agent_pos[agent_idx]
                            end
                        end
                    end
                end
            end

            # Agents attempting to push the large box
            agent1_pushing_large = false
            agent2_pushing_large = false
            if (a[1] == BOX_PUSH_ACTIONS[:move_forward] && 
                pos1 != s.agent_pos[1])
                if intended_pos1 == s.large_box_pos[1] || intended_pos1 == s.large_box_pos[1] + 1
                    agent1_pushing_large = true
                end
            end
            if (a[2] == BOX_PUSH_ACTIONS[:move_forward] && 
                pos2 != s.agent_pos[2])
                if intended_pos2 == s.large_box_pos[1] || intended_pos2 == s.large_box_pos[1] + 1
                    agent2_pushing_large = true
                end
            end

            # Check if both agents are pushing the large box correctly
            if agent1_pushing_large && agent2_pushing_large
                # Both agents are pushing the large box from below and facing north
                if (orient1 == AGENT_ORIENTATIONS[:north]) && (orient2 == AGENT_ORIENTATIONS[:north])
                    # Attempt to move the large box
                    new_large_box_pos_candidate = move_direction(pomdp, s.large_box_pos[1], 
                        AGENT_ORIENTATIONS[:north], false)
                    if (new_large_box_pos_candidate != s.large_box_pos[1] && 
                        !(new_large_box_pos_candidate in s.small_box_pos || 
                        new_large_box_pos_candidate in [intended_pos1, intended_pos2]))
                        # Move the large box
                        new_large_box_pos[1] = new_large_box_pos_candidate
                    else
                        # Can't move the large box; agents stay in original positions
                        intended_pos1 = s.agent_pos[1]
                        intended_pos2 = s.agent_pos[2]
                    end
                else
                    # Agents are not facing north; can't move the large box
                    intended_pos1 = s.agent_pos[1]
                    intended_pos2 = s.agent_pos[2]
                end
            elseif agent1_pushing_large || agent2_pushing_large
                # Only one agent is pushing the large box; can't move it
                if agent1_pushing_large
                    intended_pos1 = s.agent_pos[1]
                else
                    intended_pos2 = s.agent_pos[2]
                end
            end

            # Ensure agents didn't end up in the same position after adjustments
            if intended_pos1 == intended_pos2
                continue
            end
            
            # Create the new state
            num_small_at_goal, num_small_at_top = num_at_goal_top(pomdp, new_small_box_pos, 
                get_prop(pomdp.mg, :small_box_goals))
            num_large_at_goal, num_large_at_top = num_at_goal_top(pomdp, new_large_box_pos, 
                get_prop(pomdp.mg, :large_box_goals))
            
            modify_state = false
            if num_small_at_goal > 0
                state_symbol = Symbol("small_$(num_small_at_goal)_at_goal")
                modify_state = true
            elseif num_small_at_top > 0
                state_symbol = Symbol("small_at_top")
                modify_state = true
            elseif num_large_at_goal > 0
                state_symbol = Symbol("large_$(num_large_at_goal)_at_goal")
                modify_state = true
            elseif num_large_at_top > 0
                state_symbol = Symbol("large_at_top")
                modify_state = true
            end
            
            if modify_state
                new_state = pomdp.box_goal_states[state_symbol]
            else
                new_state = BoxPushState(new_small_box_pos, new_large_box_pos, 
                    (intended_pos1, intended_pos2), (orient1, orient2))    
            end

            # Update the probability
            next_state_probs[new_state] = get(next_state_probs, new_state, 0.0) + total_prob
        end
    end
    
    # Normalize probabilities (if necessary)
    total = sum(values(next_state_probs))
    if total > 0
        state_vec = Vector{BoxPushState}(undef, length(next_state_probs))
    prob_vec = Vector{Float64}(undef, length(next_state_probs))
        for (ii, (state, prob)) in enumerate(next_state_probs)
            state_vec[ii] = state
            prob_vec[ii] = prob / total
        end
    else
        # If no valid next states (should not happen), return the current state with probability 1
        @warn "`transition`: No valid next states found. Returning current state."
        next_state_probs[s] = 1.0
    end

    # Return the distribution
    return SparseCat(state_vec, prob_vec)
end

# Function to get the new orientation after turning left or right
function new_orientation(current_orientation::Int, action::Int)
    if action == BOX_PUSH_ACTIONS[:turn_left]
        return mod1(current_orientation - 1, 4)
    elseif action == BOX_PUSH_ACTIONS[:turn_right]
        return mod1(current_orientation + 1, 4)
    else
        return current_orientation  # No change
    end
end
