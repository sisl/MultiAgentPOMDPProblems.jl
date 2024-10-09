
const ROBOT_COLORS = [
    RGB(1.0, 0.627, 0.0),
    RGB(0.678, 0.847, 0.902),
    RGB(1.0, 0.502, 1.0),
    RGB(0.502, 1.0, 0.502),
    RGB(1.0, 0.843, 0.0),
    RGB(0.0, 0.502, 0.502)
]

function rect(w, h, x, y)
    return Shape(x .+ [w, -w, -w, w, w], y .+ [h, h, -h, -h, h])
end

function circ(x, y, r; kwargs...)
    return ellip(x, y, r, r; kwargs...)
end

function ellip(x, y, a, b; num_pts=25)
    angles = [range(0; stop=2π, length=num_pts); 0]
    xs = a .* sin.(angles) .+ x
    ys = b .* cos.(angles) .+ y
    return Shape(xs, ys)
end

function px_per_tick(plt)
    (x_size, y_size) = plt[:size]
    xlim = xlims(plt)
    ylim = ylims(plt)
    xlim_s = xlim[2] - xlim[1]
    ylim_s = ylim[2] - ylim[1]
    if xlim_s >= ylim_s
        px_p_tick = x_size / xlim_s
    else
        px_p_tick = y_size / ylim_s
    end
    return px_p_tick
end

function plot_agent!(plt, x::Real, y::Real; kwargs...)
    return plot_agent!(plt, Float64(x), Float64(y); kwargs...)
end

function plot_agent!(plt, x::Float64, y::Float64; color, alpha=1.0, body_size=0.25)
    plt = plot!(plt, circ(x, y, body_size); color=color, alpha=alpha)
    return plt
end

# Helper function to draw a belief bar at position (x, y) with height proportional to belief
function draw_belief_bar!(plt, x::Real, y::Real, belief::Real; height::Real=0.15, width::Real=0.5)

    # Outline of the bar
    outer_rec = rect(width/2, height/2, x, y)

    # Inner rectangle representing the belief
    inner_width = width * belief
    x_inner = x - width/2 + inner_width/2
    
    inner_rec = rect(inner_width/2, height/2, x_inner, y)
    # Colors
    bar_color = RGB(0.6, 0.8, 0.6)  # Light green

    # Plot the outer rectangle
    plot!(plt, outer_rec, fillcolor=:white, linecolor=:black)
    # Plot the inner rectangle
    plot!(plt, inner_rec, fillcolor=bar_color, alpha=0.9, linecolor=:green)
end

# Helper function to draw the broadcast symbol at position (x, y)
function draw_broadcast_symbol!(plt, x::Real, y::Real; radius::Real=0.3)
    # Draw lines at certain angles to represent broadcasting
    angles = [π/4, π/2, 3π/4]
    for angle in angles
        x_end = x + radius * cos(angle)
        y_end = y + radius * sin(angle)
        plot!(plt, [x, x_end], [y, y_end], linecolor=:red, linewidth=4)
    end
end

function estimate_plot_extent(p::Plots.Plot)
    # Get the current axis limits
    xlims = Plots.xlims(p)
    ylims = Plots.ylims(p)
    
    # Initialize variables to track the full extent
    min_x, max_x = xlims
    min_y, max_y = ylims
    
    # Function to update extent based on a point
    function update_extent!(x, y)
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    end
    
    # Check annotations
    # We'll use a try-catch block to handle potential differences in Plot structure
    try
        for series in p.series_list
            if hasfield(typeof(series), :annotations)
                for ann in series.annotations
                    update_extent!(ann.x, ann.y)
                end
            end
        end
    catch e
        @warn "Unable to process annotations: $e"
    end
    
    # Check other plot elements (like titles, labels, etc.)
    # This is a simplified approach and might need adjustment
    padding = 0.1  # 10% padding
    range_x = max_x - min_x
    range_y = max_y - min_y
    min_x -= range_x * padding
    max_x += range_x * padding
    min_y -= range_y * padding
    max_y += range_y * padding
    
    return (min_x, max_x, min_y, max_y)
end

function dynamic_plot_resize!(p; max_size=(1000, 1000), aspect_ratio_range=(0.5, 2.0))
    # Get the estimated full extent of the plot
    min_x, max_x, min_y, max_y = estimate_plot_extent(p)
    
    # Calculate the data ranges
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Calculate the aspect ratio of the data
    data_aspect_ratio = x_range / y_range
        
    # Clamp the aspect ratio within the specified range
    clamped_aspect_ratio = clamp(data_aspect_ratio, aspect_ratio_range[1], aspect_ratio_range[2])
    
    # Calculate new dimensions
    if clamped_aspect_ratio > 1
        new_width = max_size[1]
        new_height = new_width / clamped_aspect_ratio
    else
        new_height = max_size[2]
        new_width = new_height * clamped_aspect_ratio
    end
    
    # Ensure dimensions are integers
    new_size = (ceil(Int, new_width), ceil(Int, new_height))
    
    # Resize the plot
    plot!(p, size=new_size)
    
    return p
end
