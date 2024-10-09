
const JOINT_OFFSET_MAGNITUDE = 0.1

"""
    POMDPTools.render(pomdp::JointMeetPOMDP, step::NamedTuple)

Render function for `JointMeetPOMDP` that plots the environment, agents, and beliefs.
"""
function POMDPTools.render(pomdp::JointMeetPOMDP, step::NamedTuple; 
    legend_location::Tuple{Float64, Float64}=(0.0, 0.0), title_str::String="", 
    title_font_size::Int=30, plot_legend::Bool=true
)

    # Default legen locations for simple grids
    if legend_location == (0.0, 0.0)
        nrows = get_prop(pomdp.mg, :nrows)
        legend_location = (-1.0, -(nrows/2 - 0.4))
    end


    plt = plot(; 
        legend=false, ticks=false, showaxis=false, grid=false, aspectratio=:equal, 
        size=(1000, 1000), title=title_str, titlefontsize=title_font_size
    )

    plt = plot_jointmeet!(plt, pomdp)
    plot_belief_flag = true

    if plot_legend
        plt = add_legend!(plt, pomdp, legend_location)
    end

    px_p_tick = px_per_tick(plt)
    fnt_size_a = Int(floor(px_p_tick / 12))
    
    # Plot agents if the state is provided
    if !isnothing(get(step, :s, nothing))
        s = step.s
        agent_positions = s.r_positions
        plt = plot_agents!(plt, pomdp, agent_positions)
    end

    # Annotate the action if provided
    if !isnothing(get(step, :a, nothing))
        # Determine font size based on plot size
        num_cols = get_prop(pomdp.mg, :ncols)
        num_rows = get_prop(pomdp.mg, :nrows)
        xc = (num_cols + 1) / 2
        yc = -(num_rows + 1.0)
        a = step.a
        action_names = action_name(pomdp, a)
        # Capitalize the first letter of each action
        action_names = [uppercase(first(name)) * lowercase(name[2:end]) for name in action_names]
        action_text = join(action_names, "-")
        action_text = replace(action_text, "[" => "", "]" => "")
        action_text = latexstring("\$\\textrm{$(replace(action_text, " " => "~"))}\$")
        plt = annotate!(plt, xc, yc, (text(action_text, :black, :center, fnt_size_a)))
    end

    # Plot the belief if provided
    if !isnothing(get(step, :b, nothing)) && plot_belief_flag
        b = step.b
        plt = plot_belief!(plt, pomdp, b; belief_threshold=0.01)
    end

    dynamic_plot_resize!(plt; max_size=(1000, 1000))
    return plt
end

"""
    plot_agents!(plt, pomdp::JointMeetPOMDP, agent_positions::Vector{Int})

Function to plot agents at their positions.
"""
function plot_agents!(plt, pomdp::JointMeetPOMDP, agent_positions::Vector{Int})
    return plot_agents!(plt, pomdp, Tuple.(agent_positions))
end
function plot_agents!(plt, pomdp::JointMeetPOMDP, agent_positions::Tuple{Vararg{Int}})
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)
    num_agents = length(agent_positions)
    offset_angle = 2*π / num_agents
    for i in 1:num_agents
        pos = agent_positions[i]
        if pos == 0 # Terminal state, agents have met
            continue
        end
        (xi, yj) = node_pos_mapping[pos]

        # Offset if agents are in the same position
        offset = (0.0, 0.0)
        c = ROBOT_COLORS[i]

        if sum(agent_positions .== pos) > 1
            angle = π/2 + (i-1)*offset_angle
            offset = JOINT_OFFSET_MAGNITUDE .* (cos(angle), sin(angle))
        end

        x_agent = yj + offset[1]
        y_agent = -xi + offset[2]

        plt = plot_agent!(plt, x_agent, y_agent; color=c, body_size=0.2)
    end
    return plt
end

"""
    plot_belief!(plt, pomdp::JointMeetPOMDP, b::Any; belief_threshold::Float64=0.01)

Function to plot the belief over agent positions onto the existing plot.
"""
function plot_belief!(plt, pomdp::JointMeetPOMDP, b::Any; belief_threshold::Float64=0.01)
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
    else
        error("Unsupported belief type")
    end

    num_agents = pomdp.num_agents
    num_cells = get_prop(pomdp.mg, :num_grid_pos)
    node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)

    # Initialize belief grids for agents
    agent_beliefs = [zeros(num_cells) for _ in 1:num_agents]

    # Aggregate belief over the state space
    for (prob, s) in zip(probs, state_list)
        for i in 1:num_agents
            if s.r_positions[i] == 0
                continue
            end
            agent_beliefs[i][s.r_positions[i]] += prob
        end
    end

    px_p_tick = px_per_tick(plt)
    fnt_size = Int(floor(px_p_tick / 20))
    
    # Plot agent position beliefs
    for i in 1:num_agents
        for pos in 1:num_cells
            belief = agent_beliefs[i][pos]
            if belief >= belief_threshold
                (xi, yj) = node_pos_mapping[pos]
                x_agent = yj - 0.38 + 0.25 * (i - 1)
                y_agent = -xi - 0.38
                c = ROBOT_COLORS[i]
                alpha_scale = max(belief, 0.1)
                plt = plot_agent!(plt, x_agent, y_agent; color=c, alpha=alpha_scale, body_size=0.11)
                # Plot belief percentage as text
                belief_text = "$(round(belief, digits=2))"
                plt = annotate!(plt, x_agent, y_agent, (text(belief_text, :black, :center, fnt_size)))
            end
        end
    end

    return plt
end

"""
    plot_jointmeet(pomdp::JointMeetPOMDP)

Function to plot the JointMeet environment grid based on the map.
"""
function plot_jointmeet!(plt, pomdp::JointMeetPOMDP)
    map_str = map_str_from_metagraph(pomdp)
    lines = split(map_str, '\n')
    nrows = length(lines)
    ncols = length(lines[1])

    # Plot a blank section at the bottom for action text
    num_cols = get_prop(pomdp.mg, :ncols)
    num_rows = get_prop(pomdp.mg, :nrows)
    plt = plot!(plt, rect(0.0, 0.5, (num_cols+1)/2, -(num_rows+1.1)); 
        linecolor=:white, color=:white)
    
    # Blank section at the top for title
    plt = plot!(plt, rect(0.0, 0.2, (num_cols+1)/2, -0.5); 
        linecolor=:white, color=:white)

    # Plot the grid cells
    for xi in 1:nrows
        for yj in 1:ncols
            char = lines[xi][yj]
            color = :white  # Default color for open space

            if char == 'x'      # Wall
                color = :black
                line_color = :black
            else
                color = :white
                line_color = :black
            end 
            # Draw the cell
            plt = plot!(plt, rect(0.5, 0.5, yj, -xi); color=color, linecolor=line_color)
        end
    end

    return plt
end


"""
    add_legend!(plt::Plot, pomdp::JointMeetPOMDP)

Function to add a legend to the plot.
"""
function add_legend!(plt, pomdp::JointMeetPOMDP, legend_location::Tuple{Float64, Float64})
    # Coordinates for the legend
    legend_x, legend_y = legend_location
    legend_x_center = (legend_x + legend_x + 1.0) / 2.0
    legend_y_start = legend_y
    y_spacing = 0.15  # Spacing between entries

    px_p_tick = px_per_tick(plt)
    fnt_size = Int(floor(px_p_tick / 12))

    plt = annotate!(plt, legend_x_center, legend_y_start, text("Legend", :black, :bold, :center, fnt_size))

    legend_y_start = legend_y - y_spacing 
    curr_y = legend_y_start

    fnt_size = Int(floor(px_p_tick / 25))

    # Add agent legend
    num_agents = pomdp.num_agents
    for i in 1:num_agents
        y_space_mult = floor(Int, (i - 1) / 2)
        curr_y = legend_y_start - (1 + y_space_mult) * y_spacing

        x_space = i % 2 == 1 ? 0.0 : 0.6

        # Plot agent symbol
        # plt = plot_robot!(plt, legend_x + x_space, curr_y; color=ROBOT_COLORS[i])
        plt = plot!(plt, circ(legend_x + x_space, curr_y, 0.07); color=ROBOT_COLORS[i])
        plt = annotate!(plt, legend_x + x_space + 0.12, curr_y, text("Agent $i", :black, :left, fnt_size))
    end

    # Add belief explanations
    y_pos = curr_y - y_spacing - 0.5
    x_pos = legend_x_center

    plt = plot!(plt, rect(0.5, 0.5, x_pos, y_pos); color=:white)

    plt = plot_agent!(plt, x_pos, y_pos; color=:white, body_size=0.2)
    plt = annotate!(plt, x_pos, y_pos, text("True\nAgent\nState", :black, :center, fnt_size))

    x_pos_p = x_pos - 0.38
    y_pos_p = y_pos - 0.38
    for i in 1:num_agents
        x_agent = x_pos_p + 0.25 * (i - 1)
        y_agent = y_pos_p
        plt = plot!(plt, circ(x_agent, y_agent, 0.11); color=ROBOT_COLORS[i], alpha=0.3)
    end

    plt = annotate!(plt, x_pos, y_pos_p, text("Belief of Agent States", :black, :center, fnt_size))

    return plt
end


"""
    action_name(pomdp::JointMeetPOMDP, a::Vector{Int})

Helper function to get the action names from the action index.
"""
action_name(pomdp::JointMeetPOMDP, a::Tuple{Vararg}) = action_name(pomdp, collect(a))
action_name(::JointMeetPOMDP, a::Vector{Symbol}) = string.(a)
function action_name(pomdp::JointMeetPOMDP, a::Vector{Int})
    return string.([JOINTMEET_ACTION_NAMES[ai] for ai in a])
end

# function POMDPTools.render(pomdp::JointMeetPOMDP, step::NamedTuple; pre_act_text::String="")
#     plt = nothing

#     offset_angle = 2*π / pomdp.num_agents
    
#     plt = plot_jointmeet(pomdp)

#     if !isnothing(get(step, :s, nothing))
#         for (ii, r_pos) in enumerate(step.s.r_positions)
#             offset = (0.0, 0.0)
#             c = ROBOT_COLORS[ii % length(ROBOT_COLORS) + 1]
            
#             robot_in_same_position = sum(r_pos == r_pos2 for r_pos2 in step.s.r_positions) > 1
#             if robot_in_same_position
#                 angle_to_offset = π/2 + offset_angle * (ii - 1)
#                 offset = JOINT_OFFSET_MAGNITUDE .* (cos(angle_to_offset), sin(angle_to_offset))
#             end
            
#             if r_pos == 0
#                 continue
#             end
#             r_x, r_y = get_prop(pomdp.mg, :node_pos_mapping)[r_pos]
#             plt = plot_robot!(plt, (r_y, -r_x) .+ offset; color=c)
#         end
#     end

#     if !isnothing(get(step, :a, nothing))
#         # Determine appropriate font size based on plot size
#         px_p_tick = px_per_tick(plt)
#         fnt_size = Int(floor(px_p_tick / 2 / 1.3333333))
#         num_cols = get_prop(pomdp.mg, :ncols)
#         num_rows = get_prop(pomdp.mg, :nrows)
#         xc = (num_cols + 1) / 2
#         yc = -(num_rows + 1.0)
#         a1, a2 = action_from_index(pomdp, step.a)
#         act_text = "$(JOINTMEET_ACTION_NAMES[a1]) - $(JOINTMEET_ACTION_NAMES[a2])"
#         action_text = latexstring("\$a = \\textrm{$(act_text)}\$")
#         plt = annotate!(plt, xc, yc, (text(action_text, :black, :center, fnt_size)))
#     end

#     return plt
# end

# function plot_jointmeet_w_belief(pomdp::JointMeetPOMDP, b::Vector{Float64}, agent_id::Int, own_id::Int; kwargs...)
#     @assert length(b) == length(pomdp) "Belief must be the same length as the state list"
#     return plot_jointmeet_w_belief(pomdp, b, ordered_states(pomdp), agent_id, own_id; kwargs...)
# end
# function plot_jointmeet_w_belief(pomdp::JointMeetPOMDP, b::DiscreteBelief, agent_id::Int, own_id::Int; kwargs...)
#     return plot_jointmeet_w_belief(pomdp, b.b, b.state_list, agent_id, own_id; kwargs...)
# end
# function plot_jointmeet_w_belief(pomdp::JointMeetPOMDP, b::SparseCat, agent_id::Int, own_id::Int; kwargs...)
#     return plot_jointmeet_w_belief(pomdp, b.probs, b.vals, agent_id, own_id; kwargs...)
# end

# function plot_jointmeet_w_belief(
#     pomdp::JointMeetPOMDP, b::Vector, state_list::Vector{JointMeetState}, 
#     agent_id::Int, own_id::Int;
#     color_grad=cgrad(:Greens_9), prob_color_scale=1.0,
#     a=nothing,
#     o=nothing
# )

#     @assert length(b) == length(state_list) "Belief and state list must be the same length"
#     num_cells = get_prop(pomdp.mg, :num_grid_pos)
#     node_pos_mapping = get_prop(pomdp.mg, :node_pos_mapping)

#     map_str = map_str_from_metagraph(pomdp)
#     map_str_mat = Matrix{Char}(undef, get_prop(pomdp.mg, :nrows), get_prop(pomdp.mg, :ncols))
#     for (i, line) in enumerate(split(map_str, '\n'))
#         map_str_mat[i, :] .= collect(line)
#     end

#     # Get the belief of the robot and the target in each cell
#     grid_r_bs = [zeros(num_cells) for _ in 1:pomdp.num_agents]
#     for (ii, sᵢ) in enumerate(state_list)
#         if isterminal(pomdp, sᵢ)
#             continue
#         end
#         for (jj, r_pos) in enumerate(sᵢ.r_positions)
#             grid_r_bs[jj][r_pos] += b[ii]
#         end
#     end
    
#     plt = plot(; legend=false, ticks=false, showaxis=false, grid=false, aspectratio=:equal)

#     # Plot blank section at bottom for action text
#     nc = get_prop(pomdp.mg, :ncols)
#     nr = get_prop(pomdp.mg, :nrows)
#     plt = plot!(plt, rect(0.0, 0.5, (nc+1)/2, -(nr+1.1)); linecolor=RGB(1.0, 1.0, 1.0), color=:white)

#     node_mapping = get_prop(pomdp.mg, :node_mapping)
#     # Plot the grid
#     for xi in 1:get_prop(pomdp.mg, :nrows)
#         for yj in 1:get_prop(pomdp.mg, :ncols)

#             if map_str_mat[xi, yj] == 'x'
#                 color = :black
#             else
#                 cell_i = node_mapping[(xi, yj)]
#                 color_scale = grid_r_bs[agent_id][cell_i] * prob_color_scale
#                 if color_scale < 0.05
#                     color = :white
#                 else
#                     color = get(color_grad, color_scale)
#                 end
#             end

#             plt = plot!(plt, rect(0.5, 0.5, yj, -xi); color=color)
#         end
#     end
#     # Determine scale of font based on plot size
#     px_p_tick = px_per_tick(plt)
#     fnt_size = Int(floor(px_p_tick / 4 / 1.3333333))

#     # Plot the robot (tranparancy based on belief) and annotate the target belief as well
#     for cell_i in 1:num_cells
#         xi, yi = node_pos_mapping[cell_i]
#         prob_text = round(grid_r_bs[agent_id][cell_i]; digits=2)
#         if prob_text < 0.01
#             prob_text = ""
#         end
#         plt = annotate!(yi, -xi, (text(prob_text, :black, :center, fnt_size)))
#         if own_id != 0
#             c = ROBOT_COLORS[own_id % length(ROBOT_COLORS) + 1]
#             if grid_r_bs[own_id][cell_i] >= 1/num_cells - 1e-5
#                 plt = plot_robot!(plt, (yi, -xi); fillalpha=grid_r_bs[own_id][cell_i], color=c)
#             end
#         end
#     end
    
#     # Determine appropriate font size based on plot size
#     px_p_tick = px_per_tick(plt)
#     fnt_size = Int(floor(px_p_tick / 2 / 1.3333333)) 
#     num_cols = get_prop(pomdp.mg, :ncols)
#     num_rows = get_prop(pomdp.mg, :nrows)
#     xc = (num_cols + 1) / 2
#     yc = -(num_rows + 1.0)
    
#     if !isnothing(a)
#         a1, a2 = action_from_index(pomdp, a)
#         act_text = "$(JOINTMEET_ACTION_NAMES[a1]) - $(JOINTMEET_ACTION_NAMES[a2])"
#         action_text = latexstring("\$\\pi(b) = \\textrm{$(act_text)}\$")
#         plt = annotate!(plt, xc, yc, (text(action_text, :black, :center, fnt_size)))
#         yc -= 0.5
#     end
    
#     if !isnothing(o)
#         obs_text = latexstring("\$o = \\textrm{$(o)}\$")
#         plt = annotate!(plt, xc, yc, (text(obs_text, :black, :center, fnt_size)))
#     end
    
#     return plt

# end

# function plot_jointmeet(pomdp::JointMeetPOMDP; size::Tuple{Int, Int}=(1200, 1200))
#     map_str = map_str_from_metagraph(pomdp)
#     map_str_mat = Matrix{Char}(undef, get_prop(pomdp.mg, :nrows), get_prop(pomdp.mg, :ncols))
#     for (i, line) in enumerate(split(map_str, '\n'))
#         map_str_mat[i, :] .= collect(line)
#     end

#     plt = plot(; 
#         legend=false, ticks=false, showaxis=false, grid=false, aspectratio=:equal, 
#         size=size
#     )

#     # Plot blank section at bottom for action text
#     nc = get_prop(pomdp.mg, :ncols)
#     nr = get_prop(pomdp.mg, :nrows)
#     plt = plot!(plt, rect(0.0, 0.5, (nc+1)/2, -(nr+1.0)); linecolor=RGB(1.0, 1.0, 1.0), color=:white)

#     # Plot the grid
#     for xi in 1:get_prop(pomdp.mg, :nrows)
#         for yj in 1:get_prop(pomdp.mg, :ncols)
#             color = :white
#             if map_str_mat[xi, yj] == 'x'
#                 color = :black
#             end
#             plt = plot!(plt, rect(0.5, 0.5, yj, -xi); color=color)
#         end
#     end

#     return plt
# end
