
@enum TigerState TigerLeft TigerRight
@enum TigerAction OpenLeft OpenRight Listen
@enum TigerObservation HearLeft HearRight

const MULTITIGER_ACTION_NAMES = Dict(
    OpenLeft => "Open Left",
    OpenRight => "Open Right",
    Listen => "Listen"
)

"""
    MultiTigerPOMDP <: POMDP{TigerState, Vector{TigerAction}, Vector{TigerObservation}}

Multi-agent Tiger POMDP.

# Fields
- `num_agents::Int`: Number of agents
- `listen_cost::Float64`: Cost for listening
- `open_correct_reward::Float64`: Reward for opening the correct door
- `open_wrong_penalty::Float64`: Penalty for opening the wrong door
- `discount_factor::Float64`: Discount factor for future rewards
- `correct_obs_prob::Float64`: Probability of receiving the correct observation
- `observation_agent::Int`: 0 = joint, >0 = individual
"""
struct MultiTigerPOMDP <: POMDP{TigerState, Vector{TigerAction}, Vector{TigerObservation}}
    num_agents::Int
    listen_cost::Float64
    open_correct_reward::Float64
    open_wrong_penalty::Float64
    discount_factor::Float64
    correct_obs_prob::Float64
    observation_agent::Int
end

"""
    MultiTigerPOMDP(; kwargs...)

Construct a MultiTigerPOMDP.

# Keywords
- `num_agents::Int = 2`: Number of agents
- `listen_cost::Float64 = -1.0`: Cost for listening
- `open_correct_reward::Float64 = 10.0`: Reward for opening the correct door
- `open_wrong_penalty::Float64 = -100.0`: Penalty for opening the wrong door
- `discount_factor::Float64 = 0.9`: Discount factor for future rewards
- `correct_obs_prob::Float64 = 0.85`: Probability of receiving the correct observation
- `observation_agent::Int = 0`: 0 = joint, >0 = individual
"""
function MultiTigerPOMDP(;
    num_agents::Int = 2,
    listen_cost::Float64 = -1.0,
    open_correct_reward::Float64 = 10.0,
    open_wrong_penalty::Float64 = -100.0,
    discount_factor::Float64 = 0.9,
    correct_obs_prob::Float64 = 0.85,
    observation_agent::Int = 0
)
    return MultiTigerPOMDP(
        num_agents, listen_cost, open_correct_reward, open_wrong_penalty,
        discount_factor, correct_obs_prob, observation_agent
    )
end

POMDPs.states(::MultiTigerPOMDP) = [TigerLeft, TigerRight]

POMDPs.stateindex(::MultiTigerPOMDP, s::TigerState) = s == TigerLeft ? 1 : 2

POMDPs.actions(pomdp::MultiTigerPOMDP) = 
    vec([collect(a) for a in Base.product(fill([OpenLeft, OpenRight, Listen], pomdp.num_agents)...)])
    
POMDPs.actionindex(pomdp::MultiTigerPOMDP, a::Vector{TigerAction}) = findfirst(==(a), actions(pomdp))
    
function POMDPs.observations(pomdp::MultiTigerPOMDP)
    if pomdp.observation_agent == 0
        return vec([collect(o) for o in Base.product(fill([HearLeft, HearRight], pomdp.num_agents)...)])
    else
        return [[HearLeft], [HearRight]]
    end
end

POMDPs.obsindex(pomdp::MultiTigerPOMDP, o::Vector{TigerObservation}) = findfirst(==(o), observations(pomdp))

POMDPs.discount(pomdp::MultiTigerPOMDP) = pomdp.discount_factor

function POMDPs.transition(pomdp::MultiTigerPOMDP, s::TigerState, a::Vector{Int})
    return POMDPs.transition(pomdp, s, map(x -> TigerAction(x), a))
end
function POMDPs.transition(pomdp::MultiTigerPOMDP, s::TigerState, a::Vector{Symbol})
    return POMDPs.transition(pomdp, s, map(x -> eval(x), a))
end
function POMDPs.transition(pomdp::MultiTigerPOMDP, s::TigerState, a::Vector{TigerAction})
    if all(action == Listen for action in a)
        return SparseCat([s], [1.0])
    else
        return SparseCat([TigerLeft, TigerRight], [0.5, 0.5])
    end
end

function POMDPs.observation(pomdp::MultiTigerPOMDP, a::Vector{Int}, sp::TigerState)
    return POMDPs.observation(pomdp, map(x -> TigerAction(x), a), sp)
end
function POMDPs.observation(pomdp::MultiTigerPOMDP, a::Vector{Symbol}, sp::TigerState)
    return POMDPs.observation(pomdp, map(x -> eval(x), a), sp)
end
function POMDPs.observation(pomdp::MultiTigerPOMDP, a::Vector{TigerAction}, sp::TigerState)
    if any(ai != Listen for ai in a)
        return POMDPTools.Uniform(observations(pomdp))
    end

    correct_obs = sp == TigerLeft ? HearLeft : HearRight
    incorrect_obs = sp == TigerLeft ? HearRight : HearLeft

    if pomdp.observation_agent == 0
        obs_space = observations(pomdp)
        probs = zeros(length(obs_space))
        for (i, o) in enumerate(obs_space)
            p = 1.0
            for oi in o
                p *= (oi == correct_obs) ? pomdp.correct_obs_prob : (1 - pomdp.correct_obs_prob)
            end
            probs[i] = p
        end
        return SparseCat(obs_space, probs)
    else
        return SparseCat([[correct_obs], [incorrect_obs]], [pomdp.correct_obs_prob, 1 - pomdp.correct_obs_prob])
    end
end

function POMDPs.reward(pomdp::MultiTigerPOMDP, s::TigerState, a::Vector{Int})
    return POMDPs.reward(pomdp, s, map(x -> TigerAction(x), a))
end
function POMDPs.reward(pomdp::MultiTigerPOMDP, s::TigerState, a::Vector{Symbol})
    return POMDPs.reward(pomdp, s, map(x -> eval(x), a))
end
function POMDPs.reward(pomdp::MultiTigerPOMDP, s::TigerState, a::Vector{TigerAction})
    if all(action == Listen for action in a)
        return pomdp.listen_cost * pomdp.num_agents
    end

    # Made it here, so at least one agent is opening a door
    total_reward = 0.0
    num_open_left = count(action -> action == OpenLeft, a)
    num_open_right = count(action -> action == OpenRight, a)
    num_listen = count(action -> action == Listen, a)
    num_correct_door = s == TigerLeft ? num_open_right : num_open_left
    num_incorrect_door = s == TigerLeft ? num_open_left : num_open_right
    
    # Listen actions still get listen_cost for each agent listening
    total_reward += pomdp.listen_cost * num_listen
    
    if num_incorrect_door > 0
        # Penalize all agents for opening the wrong door
        total_reward += pomdp.open_wrong_penalty / num_incorrect_door
    else
        # We get a reward based on the number of agents opening the correct door
        total_reward += pomdp.open_correct_reward * num_correct_door
    end

    return total_reward
end

POMDPs.initialstate(pomdp::MultiTigerPOMDP) = POMDPTools.Uniform(states(pomdp))

"""
    POMDPTools.render(pomdp::MultiTigerPOMDP, step::NamedTuple)

Render function for `MultiTigerPOMDP` that visualizes the doors, tiger position, belief, and actions.

# Arguments
- `pomdp::MultiTigerPOMDP`: The MultiTiger POMDP problem instance.
- `step::NamedTuple`: A named tuple containing the step information. Fields can include:
    - `s`: The current state (`TigerLeft` or `TigerRight`).
    - `b`: The current belief (a probability distribution over states).
    - `a`: The action taken by each agent (vector of `TigerAction`).
    - `step_number`: The current step number (optional).

# Returns
- A `Plots.Plot` object displaying the visualization.
"""
function POMDPTools.render(pomdp::MultiTigerPOMDP, step::NamedTuple; title_str="", title_font_size=30)
    # Extract information from the step
    s = get(step, :s, nothing)      # Current state
    b = get(step, :b, nothing)      # Current belief
    a = get(step, :a, nothing)      # Action taken

    # Create a new plot
    plt = plot(; 
        title=title_str, titlefont=font(title_font_size), 
        legend=false, ticks=false, showaxis=false, grid=false, aspectratio=:equal, 
        size=(800, 800)
    )
    
    px_p_tick = px_per_tick(plt)

    fnt_size = Int(floor(px_p_tick / 35))
    fnt_size_b = Int(floor(px_p_tick / 50))
    fnt_size_a = Int(floor(px_p_tick / 60))
    
    # Draw blank section at the top for title
    plt = plot!(plt, rect(0.0, 0.3, 5.0, 7.1); 
        linecolor=:white, color=:white)
        
    # Draw the doors
    draw_door!(plt, 3, 5)  # Left door at (3, 5)
    draw_door!(plt, 7, 5)  # Right door at (7, 5)
    # Annotate Door #'s at top
    annotate!(plt, 3, 7.25, text("Left Door", :black, :bold, :center, fnt_size))
    annotate!(plt, 7, 7.25, text("Right Door", :black, :bold, :center, fnt_size))
    # Display the tiger behind the correct door if the state is known
    if !isnothing(s)
        if s == TigerLeft
            draw_tiger!(plt, 3, 5)
        elseif s == TigerRight
            draw_tiger!(plt, 7, 5)
        end
    end

    # Display the belief as text near each door
    if !isnothing(b)
        belief_left = pdf(b, TigerLeft)
        belief_right = pdf(b, TigerRight)
        draw_belief_bar!(plt, 3, 2.3, belief_left; width=2.0, height=0.5)
        annotate!(plt, 3, 2.3, text("$(round(belief_left, digits=2))", :black, :center, fnt_size_b))
        annotate!(plt, 3, 2.3 + 0.5/2 + 0.2, text("Belief of Tiger Left", :black, :center, fnt_size_b))
        draw_belief_bar!(plt, 7, 2.3, belief_right; width=2.0, height=0.5)
        annotate!(plt, 7, 2.3, text("$(round(belief_right, digits=2))", :black, :center, fnt_size_b))
        annotate!(plt, 7, 2.3 + 0.5/2 + 0.2, text("Belief of Tiger Right", :black, :center, fnt_size_b))
    end

    # Display the actions of each agent
    if !isnothing(a)
        num_agents = length(a)
        action_names = action_name(pomdp, a)
        for i in 1:num_agents
            annotate!(plt, 4.1, 6.75 - 0.3*(i-1), text("Agent $i: $(action_names[i])", :black, :left, fnt_size_a))
        end
    end

    plt = dynamic_plot_resize!(plt; max_size=(1000, 800))
    return plt
end

# Helper function to draw a door at position (x, y)
function draw_door!(plt, x::Real, y::Real)
    door_width = 2.0
    door_height = 4.0
    rectangle = rect(door_width/2, door_height/2, x, y)
    plot!(plt, rectangle, fillcolor=RGB(0.6, 0.6, 0.6), linecolor=:black)
    # Draw door knob
    door_knob_x = x + door_width/2 - door_width/10
    plot!(plt, circ(door_knob_x, y, door_width/20), fillcolor=:black)
end

# Helper function to draw a tiger at position (x, y)
function draw_tiger!(plt, x::Real, y::Real; circle_radius=0.6, tiger_alpha=0.5)
    # Represent the tiger as an orange circle with stripes
    circle = circ(x, y, circle_radius)
    plot!(plt, circle, fillcolor=RGB(1.0, 0.6, 0.0), linecolor=:black, alpha=tiger_alpha)
    
    # Parameters for the stripes
    stripe_space = 0.2
    
    # Calculate the starting angle for the first stripe
    start_angle = 3 * π / 4
    stripe_angle = π / 2.6
    perp_angle = stripe_angle - π / 2
    circle_radius = 0.6
    
    start_point = (x + cos(start_angle) * circle_radius, y + sin(start_angle) * circle_radius)
    # Point inthe angle of perp_angle from start_point by stripe_space
    perp_point = (start_point[1] + cos(perp_angle) * stripe_space/2, start_point[2] + sin(perp_angle) * stripe_space/2)
    
    dist_to_perp = sqrt((start_point[1] - perp_point[1])^2 + (start_point[2] - perp_point[2])^2)
    
    while is_point_within_circle(perp_point, (x, y), circle_radius)
        # Second point along the stripe_angle from perp_point
        stripe_point = (perp_point[1] + cos(stripe_angle) * 0.3, perp_point[2] + sin(stripe_angle) * 0.3)
        
        # Intersection of perp_point at angle stripe_angle with the circle
        (x1_int, y1_int), (x2_int, y2_int) = find_intersection(perp_point, stripe_point, (x, y), circle_radius)
        
        # Draw a line between these two intersection points
        plot!(plt, [x1_int, x2_int], [y1_int, y2_int], linecolor=:black, linewidth=6, alpha=tiger_alpha)
        
        start_point = perp_point
        perp_point = (start_point[1] + cos(perp_angle) * stripe_space, start_point[2] + sin(perp_angle) * stripe_space)
    end
end

function is_point_within_circle(pt, circle_center, r)
    x, y = pt
    cx, cy = circle_center
    return (x - cx)^2 + (y - cy)^2 <= r^2
end

function find_intersection(pt1, pt2, circle_center, r)
    x1, y1 = pt1
    x2, y2 = pt2
    cx, cy = circle_center

    # Calculate the coefficients of the quadratic equation
    a = (x2 - x1)^2 + (y2 - y1)^2
    b = 2 * ((x2 - x1) * (x1 - cx) + (y2 - y1) * (y1 - cy))
    c = cx^2 + cy^2 + x1^2 + y1^2 - 2 * cx * x1 - 2 * cy * y1 - r^2
    
    # Calculate the discriminant
    discriminant = b^2 - 4 * a * c
    
    # Check if the discriminant is non-negative
    @assert discriminant >= 0 "Discriminant is negative"
    
    # Calculate the two intersection points
    t1 = (-b + sqrt(discriminant)) / (2 * a)
    t2 = (-b - sqrt(discriminant)) / (2 * a)
    
    # Calculate the intersection points
    x1_int = x1 + t1 * (x2 - x1)
    y1_int = y1 + t1 * (y2 - y1)
    
    x2_int = x1 + t2 * (x2 - x1)
    y2_int = y1 + t2 * (y2 - y1)
    
    return (x1_int, y1_int), (x2_int, y2_int)
end

action_name(pomdp::MultiTigerPOMDP, a::Tuple{Vararg}) = action_name(pomdp, collect(a))
action_name(pomdp::MultiTigerPOMDP, a::Vector{Symbol}) = action_name(pomdp, map(x -> eval(x), a))
action_name(pomdp::MultiTigerPOMDP, a::Vector{Int}) = action_name(pomdp, map(x -> TigerAction(x), a))
function action_name(::MultiTigerPOMDP, a::Vector{TigerAction})
    return string.([MULTITIGER_ACTION_NAMES[ai] for ai in a])
end
