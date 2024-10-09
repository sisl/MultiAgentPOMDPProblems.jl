function POMDPs.reward(pomdp::BoxPushPOMDP, s::BoxPushState, a::Tuple{Int, Int})
    return POMDPs.reward(pomdp, s, [a[1], a[2]])
end
function POMDPs.reward(pomdp::BoxPushPOMDP, s::BoxPushState, a::Tuple{Symbol, Symbol})
    return POMDPs.reward(pomdp, s, [BOX_PUSH_ACTIONS[a[1]], BOX_PUSH_ACTIONS[a[2]]])
end
function POMDPs.reward(pomdp::BoxPushPOMDP, s::BoxPushState, a::Vector{Symbol})
    return POMDPs.reward(pomdp, s, [BOX_PUSH_ACTIONS[a[1]], BOX_PUSH_ACTIONS[a[2]]])
end
function POMDPs.reward(pomdp::BoxPushPOMDP, s::BoxPushState, a::Vector{Int})
    rew = 0.0
    
    small_box_goals = get_prop(pomdp.mg, :small_box_goals)
    large_box_goals = get_prop(pomdp.mg, :large_box_goals)
    grid_to_node_mapping = get_prop(pomdp.mg, :node_mapping)
    node_to_grid_mapping = get_prop(pomdp.mg, :node_pos_mapping)
    
    # If current state is a goal state, then reward is 0.0
    small_at_goal, small_at_top = num_at_goal_top(pomdp, s.small_box_pos, small_box_goals)
    large_at_goal, large_at_top = num_at_goal_top(pomdp, s.large_box_pos, large_box_goals)
    if small_at_goal > 0 || small_at_top > 0 || large_at_goal > 0 || large_at_top > 0
        return rew
    end
    
    # Add step penalty for each agent
    rew += pomdp.step_penalty * 2  # Both agents

    if a[1] != BOX_PUSH_ACTIONS[:move_forward] && a[2] != BOX_PUSH_ACTIONS[:move_forward]
        # No chance for penalty if agents aren't moving forward
        return rew
    end
    
    grids_south_of_small = Set{Int}()
    for small_pos in s.small_box_pos
        (i, j) = node_to_grid_mapping[small_pos]
        push!(grids_south_of_small, grid_to_node_mapping[(i+1, j)])
    end
    
    grids_south_of_large = Set{Int}()
    for large_pos in s.large_box_pos
        (i, j) = node_to_grid_mapping[large_pos]
        push!(grids_south_of_large, grid_to_node_mapping[i+1, j])
        push!(grids_south_of_large, grid_to_node_mapping[i+1, j+1])
    end
    
    pushing_large = [false, false]
    intended_poss = [0, 0]
    for ii in 1:2
        other_idx = ii == 1 ? 2 : 1
        if a[ii] == BOX_PUSH_ACTIONS[:move_forward]
            intended_pos = move_direction(pomdp, s.agent_pos[ii], s.agent_orientation[ii])
            other_agent_intended_pos = move_direction(pomdp, s.agent_pos[other_idx], 
                s.agent_orientation[other_idx])
            box_idx = findfirst(x -> x == intended_pos, s.small_box_pos)
            intended_poss[ii] = intended_pos
            if intended_pos == s.agent_pos[other_idx]
                # Can't move to a spot currently occupied
                rew += pomdp.wall_penalty
                continue
            elseif intended_pos == other_agent_intended_pos
                rew += pomdp.wall_penalty
                continue
            elseif intended_pos == s.agent_pos[ii]
                # Hit a wall
                rew += pomdp.wall_penalty
                continue
            end
            if s.agent_orientation[ii] == AGENT_ORIENTATIONS[:north]
                # Then we need to check if pushing a box
                if s.agent_pos[ii] in grids_south_of_small
                    @assert !isnothing(box_idx) "Box index is nothing"
                    intended_box_pos = move_direction(pomdp, s.small_box_pos[box_idx], s.agent_orientation[ii], false)
                    if intended_box_pos in small_box_goals
                        rew += pomdp.small_box_goal_reward
                    end
                elseif s.agent_pos[ii] in grids_south_of_large
                    pushing_large[ii] = true
                end
            else
                # We are not facing north
                # Intended position is currently a large box and not facing north, so it can't move
                if intended_pos in s.large_box_pos || intended_pos in s.large_box_pos .+ 1
                    rew += pomdp.wall_penalty
                    continue
                end
                
                if box_idx !== nothing
                    # Our intended position intersects with a small box
                    if s.agent_orientation[ii] == AGENT_ORIENTATIONS[:south]
                        # Only push boxes north, east, West
                        rew += pomdp.wall_penalty
                        continue
                    end
                    intended_box_pos = move_direction(pomdp, s.small_box_pos[box_idx], s.agent_orientation[ii], false)
                    if intended_box_pos == s.small_box_pos[box_idx]
                        # Can't move the box into a wall
                        rew += pomdp.wall_penalty
                        continue
                    end
                    if box_idx == 1 && s.agent_orientation[ii] == AGENT_ORIENTATIONS[:east]
                        # Can't move the first small box east
                        rew += pomdp.wall_penalty
                        continue
                    elseif box_idx == 2 && s.agent_orientation[ii] == AGENT_ORIENTATIONS[:west]
                        # Can't move the second small box west
                        rew += pomdp.wall_penalty
                        continue
                    end
                end
            end
        end
    end
    
    if (pushing_large[1] && !pushing_large[2]) || (!pushing_large[1] && pushing_large[2])
        # One is pushing and the other is not
        rew += pomdp.wall_penalty
    end
    
    if pushing_large[1] && pushing_large[2]
        # Need to check if large is going to reach goal
        # If we made it here, then the intended_poss is the large box position
        left_most_pos = min(intended_poss[1], intended_poss[2])
        left_most_intended_pos = move_direction(pomdp, left_most_pos, AGENT_ORIENTATIONS[:north], false)
        if left_most_intended_pos in large_box_goals
            rew += pomdp.large_box_goal_reward
        end
        # Otherwise, we assume the large box is moving north, but not to the goal yet
    end

    return rew
end
