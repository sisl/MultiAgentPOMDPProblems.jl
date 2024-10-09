

const BOX_COLORS = [
    RGB(1.0, 0.502, 1.0),
    # RGB(0.502, 1.0, 0.502),
    RGB(1.0, 0.843, 0.0),
    # RGB(0.0, 0.502, 0.502)
]

const GOAL_COLOR = RGB(0.4, 0.8, 0.4)

const PUSH_OFFSET_MAGNITUDE = 0.15  # Offset for overlapping agents

"""
    POMDPTools.render(pomdp::BoxPushPOMDP, step::NamedTuple)

Render function for BoxPushPOMDP that plots the environment, agents, and boxes.
"""
function POMDPTools.render(pomdp::BoxPushPOMDP, step::NamedTuple; title_str="", title_font_size=30)
    
    plt = plot(; 
        title=title_str, titlefont=font(title_font_size),
        legend=false, ticks=false, showaxis=false, grid=false, aspectratio=:equal, 
        size=(1200, 1200)
    )
    
    plt = plot_boxpush!(plt, pomdp)
    
    px_p_tick = px_per_tick(plt)
    
    plot_belief_flag = true
    
    # Plot boxes and agents if the state is provided
    if !isnothing(get(step, :s, nothing))
        s = step.s
        small_box_pos = s.small_box_pos
        large_box_pos = s.large_box_pos
        agent_pos = s.agent_pos
        if s in values(pomdp.box_goal_states)
            # Only plot the box(es) that made it to the goal
            node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)
            new_small_box_pos = Vector{Int}()
            for pos in s.small_box_pos
                (i, _) = node_pos_mapping[pos]
                if i == 1
                    push!(new_small_box_pos, pos)
                end
            end
            small_box_pos = new_small_box_pos
            new_large_box_pos = Vector{Int}()
            for pos in s.large_box_pos
                (i, _) = node_pos_mapping[pos]
                if i == 1
                    push!(new_large_box_pos, pos)
                end
            end
            large_box_pos = new_large_box_pos
            plot_belief_flag = false
        end
        if !isempty(small_box_pos)
            plt = plot_boxes!(plt, pomdp, small_box_pos, box_type=:small)
        end
        if !isempty(large_box_pos)
            plt = plot_boxes!(plt, pomdp, large_box_pos, box_type=:large)
        end
        if plot_belief_flag
            plt = plot_agents!(plt, pomdp, agent_pos, s.agent_orientation)
        end
    end

    # Annotate the action if provided
    if !isnothing(get(step, :a, nothing))
        # Determine font size based on plot size
        
        fnt_size = Int(floor(px_p_tick / 7))
        num_cols = get_prop(pomdp.mg, :ncols)
        num_rows = get_prop(pomdp.mg, :nrows)
        xc = (num_cols + 1) / 2
        yc = -(num_rows + 1.0)
        a = step.a
        action_names = [get_action_name(ai) for ai in a]
        act_text = "$(action_names[1]) - $(action_names[2])"
        action_text = latexstring("\$\\textrm{$(replace(act_text, " " => "~"))}\$")
        plt = annotate!(plt, xc, yc, (text(action_text, :black, :center, fnt_size)))
    end
    
    # Plot the belief if provided
    if !isnothing(get(step, :b, nothing)) && plot_belief_flag
        b = step.b
        plt = plot_belief!(plt, pomdp, b; belief_threshold=0.01)
    else
        
        # Still plot orientation belief markers for consistency in sizing
        for i in 1:2
            mult = i == 1 ? -1.0 : 1.0
            if i == 1
                x_p = 0.3
            else
                n_cols = get_prop(pomdp.mg, :ncols)
                x_p = n_cols + 1 - 0.3
            end
            for orient_idx in values(AGENT_ORIENTATIONS)
                belief = 0.0
                alpha_scale = max(belief, 0.1)
                y_p_tmp = -0.5 * orient_idx - 1.3
                plt = plot_agent!(plt, x_p, y_p_tmp, orient_idx; color=ROBOT_COLORS[i], 
                    arrow_scale=0.5, body_size=0.1, alpha=alpha_scale)
            end
        end
        
        
    end
    
    plt = dynamic_plot_resize!(plt; max_size=(1200, 1200))
    return plt
end

"""
    plot_belief!(plt, pomdp::BoxPushPOMDP, b::Any; belief_threshold::Float64)

Function to plot the belief over agent and box positions onto the existing plot.
"""
function plot_belief!(plt, pomdp::BoxPushPOMDP, b::Any; belief_threshold::Float64=0.01)
    # Convert belief to a vector of probabilities and corresponding states
    if b isa DiscreteBelief
        probs = b.b
        state_list = b.state_list
    elseif b isa SparseCat
        probs = b.probs
        state_list = b.vals
    elseif b isa Deterministic
        probs = [1.0]
        state_list = [b.val]
    elseif b isa ParticleCollection
        state_num_part = Dict{BoxPushState, Int}()
        for part in b.particles
            state_num_part[part] = get(state_num_part, part, 0) + 1
        end
        state_list = Vector{BoxPushState}()
        probs = Vector{Float64}()
        for (k, v) in state_num_part
            push!(state_list, k)
            push!(probs, v / length(b.particles))
        end
    else
        error("Unsupported belief type: $(typeof(b))")
    end

    num_agents = 2
    num_cells = get_prop(pomdp.mg, :num_grid_pos)
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)

    num_small_boxes = length(get_prop(pomdp.mg, :small_box_goals))
    num_large_boxes = length(get_prop(pomdp.mg, :large_box_goals))
    
    # Initialize belief grids for agents and boxes
    agent_beliefs = [zeros(num_cells) for _ in 1:num_agents]
    small_box_beliefs = [zeros(num_cells) for _ in 1:num_small_boxes]
    large_box_beliefs = [zeros(num_cells) for _ in 1:num_large_boxes]

    agent_orient_beliefs = [zeros(length(AGENT_ORIENTATIONS)) for _ in 1:num_agents]
    
    # Aggregate belief over the state space
    for (prob, s) in zip(probs, state_list)
        for i in 1:num_agents
            agent_beliefs[i][s.agent_pos[i]] += prob
            agent_orient_beliefs[i][s.agent_orientation[i]] += prob
        end
        for (ii, pos) in enumerate(s.small_box_pos)
            small_box_beliefs[ii][pos] += prob
        end
        # For large boxes, we assume it occupies two positions
        for (ii, pos) in enumerate(s.large_box_pos)
            large_box_beliefs[ii][pos] += prob
        end
    end

    # Plot agent position beliefs
    for i in 1:num_agents
        for pos in 1:num_cells
            belief = agent_beliefs[i][pos]
            if belief >= belief_threshold
                (xi, yj) = node_pos_mapping[pos]
                sign_mult = i == 1 ? 1.0 : -1.0
                x_agent = yj - 0.38 * sign_mult
                y_agent = -xi - 0.38 

                # Plot agent icon with opacity proportional to belief
                c = ROBOT_COLORS[i]
                alpha_scale = max(belief, 0.1)
                plt = plot!(plt, circ(x_agent, y_agent, 0.11); color=c, alpha=alpha_scale)

                # Plot belief percentage as text
                px_p_tick = px_per_tick(plt)
                fnt_size = Int(floor(px_p_tick / 20))
                belief_text = "$(round(belief, digits=2))"
                plt = annotate!(plt, x_agent, y_agent, (text(belief_text, :black, :center, fnt_size)))
            end
        end
    end

    # Plot agent orientation beliefs (on the left side of plot for agent 1 and on the right
    # side for agent 2)
    for i in 1:2
        mult = i == 1 ? -1.0 : 1.0
        if i == 1
            x_p = 0.3
        else
            n_cols = get_prop(pomdp.mg, :ncols)
            x_p = n_cols + 1 - 0.3
        end
        for orient_idx in values(AGENT_ORIENTATIONS)
            belief = agent_orient_beliefs[i][orient_idx]
            alpha_scale = max(belief, 0.1)
            y_p_tmp = -0.5 * orient_idx - 1.3
            plt = plot_agent!(plt, x_p, y_p_tmp, orient_idx; color=ROBOT_COLORS[i], 
                arrow_scale=0.5, body_size=0.1, alpha=alpha_scale)
            px_p_tick = px_per_tick(plt)
            fnt_size = Int(floor(px_p_tick / 20))
            belief_text = "$(round(belief, digits=2))"
            plt = annotate!(plt, x_p + mult * 0.25, y_p_tmp, (text(belief_text, :black, :center, fnt_size)))
        end
    end
    
    # Plot small box beliefs
    for box_idx in 1:num_small_boxes
        for pos in 1:num_cells
            belief = small_box_beliefs[box_idx][pos]
            if belief >= belief_threshold
                (xi, yj) = node_pos_mapping[pos]
                x_box = yj
                y_box = -xi
                color = BOX_COLORS[1]

                # Determine position offset based on box index
                x_offset = box_idx == 1 ? -0.4 : 0.4
                y_offset = 0.4

                # Plot box with opacity proportional to belief
                plt = plot_box!(plt, x_box + x_offset, y_box + y_offset; color=color, box_size=0.1, alpha=max(belief, 0.1))

                # Plot belief percentage as text
                px_p_tick = px_per_tick(plt)
                fnt_size = Int(floor(px_p_tick / 20))
                belief_text = "$(round(belief, digits=2))"
                plt = annotate!(plt, x_box + x_offset, y_box + y_offset, (text(belief_text, :black, :center, fnt_size)))
            end
        end
    end

    # Plot large box beliefs
    @assert num_large_boxes == 1 "Currently only supported for a single large box"
    box_idx = 1
    x_offset = 0.0
    y_offset = 0.39
    for pos in 1:num_cells
        belief = large_box_beliefs[box_idx][pos]
        if belief >= belief_threshold
            (xi, yj) = node_pos_mapping[pos]
            (xi1, yj1) = node_pos_mapping[pos+1]
            x_box = (yj + yj1) / 2 + x_offset
            y_box = -(xi + xi1) / 2 + y_offset
            color = BOX_COLORS[2]

            # Plot large box with opacity proportional to belief
            plt = plot_large_box!(plt, pomdp, pos; 
                color=color, alpha=max(belief, 0.1), body_scale=0.4, 
                x_offset=x_offset, y_offset=y_offset
            )

            # Plot belief percentage as text
            px_p_tick = px_per_tick(plt)
            fnt_size = Int(floor(px_p_tick / 20))
            belief_text = "$(round(belief, digits=2))"
            plt = annotate!(plt, x_box, y_box, (text(belief_text, :black, :center, fnt_size)))
        end
    end

    return plt
end


"""
    plot_boxpush(pomdp::BoxPushPOMDP)

Function to plot the BoxPush environment grid based on the map.
"""
function plot_boxpush!(plt, pomdp::BoxPushPOMDP)
    # Get the map string and dimensions
    map_str = map_str_from_metagraph(pomdp)
    lines = split(map_str, '\n')
    nrows = length(lines)
    ncols = length(lines[1])


    # Plot a blank section at the bottom for action text
    num_cols = get_prop(pomdp.mg, :ncols)
    num_rows = get_prop(pomdp.mg, :nrows)
    plt = plot!(plt, rect(0.0, 0.4, (num_cols+1)/2, -(num_rows+1.1)); linecolor=:white, color=:white)
    
    # Plot a blank section at top for title
    plt = plot!(plt, rect(0.0, 0.1, (num_cols+1)/2, -0.4); linecolor=:white, color=:white)
    
    # Plot the grid cells
    for xi in 1:nrows
        for yj in 1:ncols
            char = lines[xi][yj]
            color = :white  # Default color for open space

            if char == 'x'      # Wall
                color = :black
            elseif xi == 1 # Top row, make it gray for the goal region
                color = RGB(0.9, 0.9, 0.9)
            end
            # Draw the cell
            plt = plot!(plt, rect(0.5, 0.5, yj, -xi); color=color)
        end
    end
    
    small_goals = get_prop(pomdp.mg, :small_box_goals)
    large_goals = get_prop(pomdp.mg, :large_box_goals)
    for small_goal in small_goals
        (xi, yj) = get_prop(pomdp.mg, :node_pos_mapping)[small_goal]
        plt = plot_box!(plt, yj, -xi; color=GOAL_COLOR, alpha=0.3)
    end
    for large_goal in large_goals
        plt = plot_large_box!(plt, pomdp, large_goal; color=GOAL_COLOR, alpha=0.3)
    end
    
    return plt
end

"""
    plot_boxes!(plt, pomdp::BoxPushPOMDP, box_positions::Vector{Int}; box_type::Symbol)

Function to plot small or large boxes on the grid, or goal regions.
"""
function plot_boxes!(plt, pomdp::BoxPushPOMDP, box_positions::Vector{Int}; box_type::Symbol)
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)
    for pos in box_positions
        (xi, yj) = node_pos_mapping[pos]
        if box_type == :small
            color = BOX_COLORS[1]  # Color for small boxes
            plt = plot_box!(plt, yj, -xi; color=color)
        elseif box_type == :large
            color = BOX_COLORS[2]  # Color for large boxes
            plt = plot_large_box!(plt, pomdp, pos; color=color)
        end
    end
    return plt
end

"""
    plot_box!(plt, x, y; color, box_size=0.3)

Helper function to plot a small box or goal region at a given position.
"""
function plot_box!(plt, x, y; color, box_size=0.25, alpha=1.0)
    rect_shape = rect(box_size, box_size, x, y)
    plt = plot!(plt, rect_shape; color=color, alpha=alpha)
    return plt
end

"""
    plot_large_box!(plt, pomdp::BoxPushPOMDP, pos::Int; color)

Helper function to plot a large box, assuming it occupies two adjacent positions.
"""
function plot_large_box!(plt, pomdp::BoxPushPOMDP, pos::Int; color, alpha=1.0,
    body_scale=1.0, x_offset=0.0, y_offset=0.0)
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)
    (xi1, yj1) = node_pos_mapping[pos]
    (xi2, yj2) = node_pos_mapping[pos+1]  # Assuming horizontal orientation

    x_center = (yj1 + yj2) / 2 + x_offset
    y_center = -(xi1 + xi2) / 2 + y_offset

    box_width = 1.25 * body_scale
    box_height = 0.5 * body_scale

    rect_shape = rect(box_width / 2, box_height / 2, x_center, y_center)
    plt = plot!(plt, rect_shape; color=color, alpha=alpha)
    return plt
end

"""
    plot_agents!(plt, pomdp::BoxPushPOMDP, agent_positions::Tuple{Int, Int}, agent_orientations::Tuple{Int, Int})

Function to plot agents at their positions with their orientations.
"""
function plot_agents!(plt, pomdp::BoxPushPOMDP, agent_positions::Tuple{Int, Int}, agent_orientations::Tuple{Int, Int})
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)
    for i in 1:2
        pos = agent_positions[i]
        orientation = agent_orientations[i]
        (xi, yj) = node_pos_mapping[pos]

        # Offset if agents are in the same position
        offset = (0.0, 0.0)
        c = ROBOT_COLORS[i]

        if count(agent_positions .== pos) > 1
            angle = π/2 + (i-1)*π
            offset = PUSH_OFFSET_MAGNITUDE .* (cos(angle), sin(angle))
        end

        x_agent = yj + offset[1]
        y_agent = -xi + offset[2]

        plt = plot_agent!(plt, x_agent, y_agent, orientation; color=c)
    end
    return plt
end

"""
    plot_agent!(plt, x::Float64, y::Float64, orientation::Int; color)

Helper function to plot an agent at a given position with orientation.
"""
function plot_agent!(plt, x::Real, y::Real, orientation::Int; kwargs...)
    return plot_agent!(plt, Float64(x), Float64(y), orientation; kwargs...)
end

function plot_agent!(plt, x::Float64, y::Float64, orientation::Int; color, alpha=1.0, body_size=0.25, arrow_scale=1.0)
    plt = plot!(plt, circ(x, y, body_size); color=color, alpha=alpha)

    # Draw a white arrow on top of the agent to indicate orientation
    arrow_length = 0.25 * arrow_scale
    offset_from_middle = 0.07 * arrow_scale
    line_offset = 0.05 * arrow_scale
    θ = orientation_to_angle(orientation)
    x_beg = x - offset_from_middle * cos(θ)
    y_beg = y - offset_from_middle * sin(θ)
    x_end = x + arrow_length * cos(θ) - offset_from_middle * cos(θ)
    y_end = y + arrow_length * sin(θ) - offset_from_middle * sin(θ)
    x_end_line = x + arrow_length * cos(θ) - (offset_from_middle + line_offset) * cos(θ)
    y_end_line = y + arrow_length * sin(θ) - (offset_from_middle + line_offset) * sin(θ)
    
    # Create a white triangle for the arrow
    arrow_width = 0.15 * arrow_scale
    arrow_points = [
        (x_end, y_end),
        (x_end - arrow_width*cos(θ+π/6), y_end - arrow_width*sin(θ+π/6)),
        (x_end - arrow_width*cos(θ-π/6), y_end - arrow_width*sin(θ-π/6))
    ]
    plt = plot!(plt, Shape(first.(arrow_points), last.(arrow_points)); color=:white, linecolor=:white)
    
    # Draw the arrow stem
    plt = plot!(plt, [x_beg, x_end_line], [y_beg, y_end_line]; color=:white, linewidth=4*arrow_scale)

    return plt
end


"""
    orientation_to_angle(orientation::Int)

Function to convert agent orientation to an angle in radians.
"""
function orientation_to_angle(orientation::Int)
    # Map orientation to angle in radians
    if orientation == AGENT_ORIENTATIONS[:north]
        θ = π/2
    elseif orientation == AGENT_ORIENTATIONS[:east]
        θ = 0.0
    elseif orientation == AGENT_ORIENTATIONS[:south]
        θ = -π/2
    elseif orientation == AGENT_ORIENTATIONS[:west]
        θ = π
    else
        θ = 0.0  # Default
    end
    return θ
end

"""
    get_action_name(action::Int)

Helper function to get the action name from the action index.
"""
function get_action_name(action::Int)
    return BOX_PUSH_ACTION_NAMES[action]
end
function get_action_name(action::Symbol)
    return BOX_PUSH_ACTION_NAMES[BOX_PUSH_ACTIONS[action]]
end
