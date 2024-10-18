@enum WirelessAction WirelessListen=1 WirelessSend=2
@enum WirelessSourceState WirelessIdle=1 WirelessPacket=2
@enum ChannelObservation ChannelIdle=1 ChannelCollision=2 ChannelSingleTx=3

const WIRELESS_ACTION_NAMES = Dict(
    WirelessListen => "Listen",
    WirelessSend => "Send"
)

struct WirelessState
    source_state::Vector{WirelessSourceState}
    queue_length::Vector{Int}
end

function WirelessState(source_states::Vector{Int}, queue_lengths::Vector{Int})
    return WirelessState(WirelessSourceState.(source_states), queue_lengths)
end
function WirelessState(source_states::Tuple, queue_lengths::Vector{Int})
    return WirelessState(collect(source_states), queue_lengths)
end

struct WirelessObservation
    channel_observation::ChannelObservation
    queue_empty::Bool
end

function Base.show(io::IO, o::WirelessObservation)
    print(io, "$(o.channel_observation)--$(o.queue_empty)")
end

import Base: ==

function ==(s1::WirelessState, s2::WirelessState)
    source_states_equal = all(s1.source_state .== s2.source_state)
    queue_lengths_equal = all(s1.queue_length .== s2.queue_length)
    return source_states_equal && queue_lengths_equal
end

function ==(o1::WirelessObservation, o2::WirelessObservation)
    channel_observations_equal = o1.channel_observation == o2.channel_observation
    queue_empties_equal = o1.queue_empty == o2.queue_empty
    return channel_observations_equal && queue_empties_equal
end

import Base: hash

function hash(s::WirelessState, h::UInt)
    h = hash(s.source_state, h)
    h = hash(s.queue_length, h)
    return h
end

function hash(o::WirelessObservation, h::UInt)
    h = hash(o.channel_observation, h)
    h = hash(o.queue_empty, h)
    return h
end

struct WirelessPOMDP <: POMDP{WirelessState, Vector{WirelessAction}, Vector{WirelessObservation}}
    num_agents::Int
    discount_factor::Float64
    g::Graph
    idle_to_packet_prob::Float64
    packet_to_idle_prob::Float64
    observation_prob::Float64
    observation_agent::Int
    send_penalty::Float64
end

function Base.show(io::IO, pomdp::WirelessPOMDP)
    println(io, "WirelessPOMDP")
    for name in fieldnames(typeof(pomdp))
        println(io, "\t", name, ": ", getfield(pomdp, name))
    end
end

"""
    WirelessPOMDP()
Constructor for a WirelessPOMDP type.
    
Keyword arguments:
- `num_agents::Int=2`: Number of agents.
- `discount_factor::Float64=0.99`: Discount factor.
- `g::Graph=cycle_graph(num_agents)`: Graph representing the network topology (defines the channels as neighbors)
- `idle_to_packet_prob::Float64=0.0470`: Probability of transitioning from idle to packet state. This is 0.01 in the MASPlan.org implementation.
- `packet_to_idle_prob::Float64=0.0741`: Probability of transitioning from packet to idle state. This is 0.02 in the MASPlan.org implementation.
- `observation_prob::Float64=0.9`: Probability of observing the channel correctly. 
- `observation_agent::Int=0`: Agent that makes the observation. 0 = all agents observe (joint observation)
"""
function WirelessPOMDP(;
    num_agents::Int=2,
    discount_factor::Float64=0.99,
    g::Graph=cycle_graph(num_agents),
    idle_to_packet_prob::Float64=0.0470,
    packet_to_idle_prob::Float64=0.0741,
    observation_prob::Float64=0.9,
    observation_agent::Int=0,
    send_penalty::Float64=0.0
)
    if !(0.0 <= idle_to_packet_prob <= 1.0)
        throw(ArgumentError("Invalid idle_to_packet_prob. Must be between 0 and 1."))
    end
    if !(0.0 <= packet_to_idle_prob <= 1.0)
        throw(ArgumentError("Invalid packet_to_idle_prob. Must be between 0 and 1."))
    end
    if !(0.0 <= observation_prob <= 1.0)
        throw(ArgumentError("Invalid observation_prob. Must be between 0 and 1."))
    end
    if !(0 <= observation_agent <= num_agents)
        throw(ArgumentError("Invalid observation_agent. Must be between 0 and $(num_agents)."))
    end
    if !(0.0 < discount_factor < 1.0)
        throw(ArgumentError("Invalid discount_factor. Must be between 0 and 1."))
    end

    return WirelessPOMDP(num_agents, discount_factor, g, idle_to_packet_prob, 
        packet_to_idle_prob, observation_prob, observation_agent, send_penalty)
end

function Base.length(pomdp::WirelessPOMDP)
    num_source_states = length(instances(WirelessSourceState))
    num_queue_lengths = 4
    return num_source_states^pomdp.num_agents * num_queue_lengths^pomdp.num_agents
end

POMDPs.isterminal(::WirelessPOMDP, ::WirelessState) = false
POMDPs.discount(pomdp::WirelessPOMDP) = pomdp.discount_factor

POMDPs.states(pomdp::WirelessPOMDP) = pomdp

function Base.iterate(pomdp::WirelessPOMDP, ii::Int=1)
    if ii > length(pomdp)
        return nothing
    end
    s = getindex(pomdp, ii)
    return (s, ii + 1)
end

function Base.getindex(pomdp::WirelessPOMDP, si::Int)
    @assert si <= length(pomdp) "Index out of bounds"
    @assert si > 0 "Index out of bounds"
    source_state_vec = [length(instances(WirelessSourceState)) for _ in 1:pomdp.num_agents]
    queue_length_vec = [4 for _ in 1:pomdp.num_agents]
    state_vec = vcat(source_state_vec, queue_length_vec)
    indices = ind2sub_dims(si, state_vec)
    source_states = collect(indices[1:pomdp.num_agents])
    queue_lengths = collect(indices[pomdp.num_agents+1:end])
    return WirelessState(source_states, queue_lengths .- 1)
end

function Base.getindex(pomdp::WirelessPOMDP, si_range::UnitRange{Int})
    return [getindex(pomdp, si) for si in si_range]
end

function Base.firstindex(::WirelessPOMDP)
    return 1
end

function Base.lastindex(pomdp::WirelessPOMDP)
    return length(pomdp)
end

function POMDPs.stateindex(pomdp::WirelessPOMDP, s::WirelessState)
    num_source_states = length(instances(WirelessSourceState))
    num_queue_lengths = 4
    state_vec = vcat([num_source_states for _ in 1:pomdp.num_agents], [num_queue_lengths for _ in 1:pomdp.num_agents])
    source_states_int = Int.(s.source_state)
    queue_lengths = s.queue_length .+ 1
    return LinearIndices(Tuple(state_vec))[source_states_int..., queue_lengths...]
end

# Uniform over source states with queue lengths of 0
function POMDPs.initialstate(pomdp::WirelessPOMDP)
    poss_source_states = [instances(WirelessSourceState) for _ in 1:pomdp.num_agents]
    queue_lengths = zeros(Int, pomdp.num_agents)
    poss_states = Vector{WirelessState}(undef, length(instances(WirelessSourceState))^pomdp.num_agents)
    ii = 1
    for source_state in Iterators.product(poss_source_states...)
        poss_states[ii] = WirelessState(source_state, queue_lengths)
        ii += 1
    end
    probs = normalize(ones(length(poss_states)), 1)
    return SparseCat(poss_states, probs)
end

function POMDPs.actions(pomdp::WirelessPOMDP)
    return vec([collect(a) for a in Base.product(fill(instances(WirelessAction), pomdp.num_agents)...)])
end

function POMDPs.actionindex(pomdp::WirelessPOMDP, a::Vector{WirelessAction})
    @assert length(a) == pomdp.num_agents "Invalid action tuple length"
    @assert all(ai in instances(WirelessAction) for ai in a) "Invalid action"
    return LinearIndices(Tuple(fill(length(instances(WirelessAction)), pomdp.num_agents)))[Int.(a)...]
end
POMDPs.actionindex(::WirelessPOMDP, ::Tuple{}) = throw(ArgumentError("Invalid action tuple (empty)"))
function POMDPs.actionindex(pomdp::WirelessPOMDP, a::Tuple{Vararg{Int}})
    return POMDPs.actionindex(pomdp, WirelessAction.(collect(a)))
end
function POMDPs.actionindex(pomdp::WirelessPOMDP, a::Tuple{Vararg{WirelessAction}})
    return POMDPs.actionindex(pomdp, collect(a))
end
function POMDPs.actionindex(pomdp::WirelessPOMDP, a::Vector{Int})
    return POMDP.actionindex(pomdp, WirelessACtion.(a))
end

function action_from_index(pomdp::WirelessPOMDP, ai::Int)
    @assert ai >= 1 "Invalid action index"
    @assert ai <= length(instances(WirelessAction))^pomdp.num_agents "Invalid action index"
    action_tuple = Tuple(CartesianIndices(Tuple(length(instances(WirelessAction)) for _ in 1:pomdp.num_agents))[ai])
    action_tuple = WirelessAction.(action_tuple)
    return action_tuple
end

POMDPs.transition(pomdp::WirelessPOMDP, s::WirelessState, a::Tuple{Vararg}) = POMDPs.transition(pomdp, s, collect(a))
function POMDPs.transition(pomdp::WirelessPOMDP, s::WirelessState, a::Vector{Int})
    return POMDPs.transition(pomdp, s, map(x -> WirelessAction(x), a))
end
function POMDPs.transition(pomdp::WirelessPOMDP, s::WirelessState, a::Vector{Symbol})
    return POMDPs.transition(pomdp, s, map(x -> eval(x), a))
end
function POMDPs.transition(pomdp::WirelessPOMDP, s::WirelessState, a::Vector{WirelessAction})
    
    queue_lengths = deepcopy(s.queue_length)
    
    new_source_states = [WirelessSourceState[] for _ in 1:pomdp.num_agents]
    new_source_state_probs = [Float64[] for _ in 1:pomdp.num_agents]
    for ii in 1:pomdp.num_agents
        if s.source_state[ii] == WirelessIdle
            push!(new_source_states[ii], WirelessPacket)
            push!(new_source_state_probs[ii], pomdp.idle_to_packet_prob)
            push!(new_source_states[ii], WirelessIdle)
            push!(new_source_state_probs[ii], 1.0 - pomdp.idle_to_packet_prob)
        else # WirelessPacket
            push!(new_source_states[ii], WirelessIdle)
            push!(new_source_state_probs[ii], pomdp.packet_to_idle_prob)
            push!(new_source_states[ii], WirelessPacket)
            push!(new_source_state_probs[ii], 1.0 - pomdp.packet_to_idle_prob)
        end
        
        if a[ii] == WirelessSend
            neighs = neighbors(pomdp.g, ii)
            all_neighs_idle = true
            for neigh in neighs
                if s.source_state[neigh] == WirelessPacket && a[neigh] == WirelessSend
                    all_neighs_idle = false
                    break
                end
            end
            if all_neighs_idle
                new_queue = queue_lengths[ii] - 1
                queue_lengths[ii] = max(new_queue, 0)
            end
        end
        
        if s.source_state[ii] == WirelessPacket
            new_queue = queue_lengths[ii] + 1
            queue_lengths[ii] = min(new_queue, 3)
        end
    end

    num_new_states = prod(length.(new_source_states))
    new_states = Vector{WirelessState}(undef, num_new_states)
    
    new_source_states = collect(Iterators.product(new_source_states...))
    new_probs = prod.(Iterators.product(new_source_state_probs...))
    
    for ii in 1:num_new_states
        new_states[ii] = WirelessState(new_source_states[ii], queue_lengths)
    end
    
    if !(isapprox(sum(new_probs), 1.0, atol=1e-6))
        @warn "New state probabilities sum to $(sum(new_probs)) instead of 1.0"
    end
    normalize!(new_probs, 1)
    
    return SparseCat(new_states, new_probs)
end

struct WirelessObservationIterator
    pomdp::WirelessPOMDP
end

function Base.length(oi::WirelessObservationIterator)
    pomdp = oi.pomdp
    num_queue_obs = 2  # true or false
    num_channel_obs = length(instances(ChannelObservation))
    
    if pomdp.observation_agent == 0
        # Joint observations
        num_agents = pomdp.num_agents
    else
        num_agents = 1
    end
    return (num_queue_obs * num_channel_obs) ^ num_agents
end

function Base.iterate(oi::WirelessObservationIterator, ii::Int=1)
    if ii > length(oi)
        return nothing
    else
        obs = getindex(oi, ii)
        return (obs, ii + 1)
    end
end

function Base.getindex(oi::WirelessObservationIterator, idx::Int)
    @assert 1 <= idx <= length(oi) "Observation index out of bounds"
    return observation_from_index(oi.pomdp, idx)
end

function Base.getindex(oi::WirelessObservationIterator, idx_range::UnitRange{Int})
    return [getindex(oi, idx) for idx in idx_range]
end

Base.firstindex(oi::WirelessObservationIterator) = 1
Base.lastindex(oi::WirelessObservationIterator) = length(oi)

function observation_from_index(pomdp::WirelessPOMDP, oi::Int)
    @assert 1 <= oi <= length(observations(pomdp)) "Observation index out of bounds"
    
    num_queue_obs = 2
    queue_empty_values = [false, true]
    channel_obs_values = collect(instances(ChannelObservation))
    num_channel_obs = length(channel_obs_values)

    if pomdp.observation_agent == 0
        num_agents = pomdp.num_agents
    else
        num_agents = 1
    end

    total_obs_per_agent = num_queue_obs * num_channel_obs
    agent_dims = [total_obs_per_agent for _ in 1:num_agents]

    indices = ind2sub_dims(oi, agent_dims)
    obs = Vector{WirelessObservation}(undef, num_agents)

    for (i, agent_obs_idx) in enumerate(indices)
        queue_idx = ((agent_obs_idx - 1) ÷ num_channel_obs) + 1
        channel_idx = ((agent_obs_idx - 1) % num_channel_obs) + 1

        queue_empty = queue_empty_values[queue_idx]
        co = channel_obs_values[channel_idx]

        obs[i] = WirelessObservation(co, queue_empty)
    end

    return obs
end

function POMDPs.observations(pomdp::WirelessPOMDP)
    return WirelessObservationIterator(pomdp)
end

function POMDPs.obsindex(pomdp::WirelessPOMDP, o::Vector{WirelessObservation})
    num_queue_obs = 2
    queue_empty_values = [false, true]
    channel_obs_values = collect(instances(ChannelObservation))
    num_channel_obs = length(channel_obs_values)
    channel_obs_map = Dict(co => idx for (idx, co) in enumerate(channel_obs_values))

    if pomdp.observation_agent == 0
        num_agents = pomdp.num_agents
    else
        num_agents = 1
    end

    total_obs_per_agent = num_queue_obs * num_channel_obs
    agent_dims = [total_obs_per_agent for _ in 1:num_agents]
    agent_indices = Int[]

    for obs in o
        queue_idx = findfirst(==(obs.queue_empty), queue_empty_values)
        channel_idx = channel_obs_map[obs.channel_observation]
        agent_obs_idx = (queue_idx - 1) * num_channel_obs + channel_idx
        push!(agent_indices, agent_obs_idx)
    end

    oi = sub2ind_dims(agent_indices, agent_dims)
    return oi
end


POMDPs.observation(pomdp::WirelessPOMDP, a::Tuple{Vararg}, sp::WirelessState) = POMDPs.observation(pomdp, collect(a), sp)
function POMDPs.observation(pomdp::WirelessPOMDP, a::Vector{Int}, sp::WirelessState)
    return POMDPs.observation(pomdp, map(x -> WirelessAction(x), a), sp)
end
function POMDPs.observation(pomdp::WirelessPOMDP, a::Vector{Symbol}, sp::WirelessState)
    return POMDPs.observation(pomdp, map(x -> eval(x), a), sp)
end
function POMDPs.observation(pomdp::WirelessPOMDP, a::Vector{WirelessAction}, sp::WirelessState)
    if pomdp.observation_agent == 0
        return joint_observation(pomdp, a, sp)
    else
        return single_observation(pomdp, a, sp, pomdp.observation_agent)
    end
end

function single_observation(pomdp::WirelessPOMDP, a::Vector{WirelessAction}, sp::WirelessState, obs_agent::Int)
    channel_obs_values = collect(instances(ChannelObservation))
    num_channel_obs = length(channel_obs_values)

    # Determine the true channel state for the observing agent
    cnt_send = 0
    neighs = neighbors(pomdp.g, obs_agent)
    for neigh in neighs
        if a[neigh] == WirelessSend
            cnt_send += 1
        end
    end
    if a[obs_agent] == WirelessSend
        cnt_send += 1
    end

    # Aggregate the channel state
    if cnt_send == 0
        true_channel_state = ChannelIdle
    elseif cnt_send == 1
        true_channel_state = ChannelSingleTx
    else
        true_channel_state = ChannelCollision
    end

    # Agent's own queue status (observed perfectly)
    queue_empty = sp.queue_length[obs_agent] == 0

    # Generate possible observations and their probabilities
    obs_vec = Vector{Vector{WirelessObservation}}()
    prob_vec = Vector{Float64}()

    for co in channel_obs_values
        if co == true_channel_state
            prob = pomdp.observation_prob
        else
            prob = (1.0 - pomdp.observation_prob) / (num_channel_obs - 1)
        end
        obs = [WirelessObservation(co, queue_empty)]
        push!(obs_vec, obs)
        push!(prob_vec, prob)
    end

    return SparseCat(obs_vec, prob_vec)
end

function joint_observation(pomdp::WirelessPOMDP, a::Vector{WirelessAction}, sp::WirelessState)
    num_agents = pomdp.num_agents

    # Prepare to store possible observations and probabilities for each agent
    agent_obs_probs = Vector{Vector{Tuple{WirelessObservation, Float64}}}(undef, num_agents)

    for agent in 1:num_agents
        obs_dist_i = single_observation(pomdp, a, sp, agent)
        obs_list = [(obs[1], prob) for (obs, prob) in weighted_iterator(obs_dist_i)]
        agent_obs_probs[agent] = obs_list
    end

    # Compute joint observations and their probabilities
    joint_obs_list = Vector{Tuple{Vector{WirelessObservation}, Float64}}()

    for obs_combination in Iterators.product(agent_obs_probs...)
        obs_vector = Vector{WirelessObservation}(undef, num_agents)
        prob = 1.0
        for i in 1:num_agents
            (obs_i, prob_i) = obs_combination[i]
            obs_vector[i] = obs_i
            prob *= prob_i
        end
        push!(joint_obs_list, (obs_vector, prob))
    end

    # Extract observations and probabilities
    obs_vec = [obs for (obs, _) in joint_obs_list]
    prob_vec = [prob for (_, prob) in joint_obs_list]

    return SparseCat(obs_vec, prob_vec)
end

# Reward is the negative sum of the queue lengths
function POMDPs.reward(pomdp::WirelessPOMDP, s::WirelessState, a::Tuple)
    return POMDPs.reward(pomdp, s, collect(a))
end
function POMDPs.reward(pomdp::WirelessPOMDP, s::WirelessState, a::Vector{Int})
    return POMDPs.reward(pomdp, s, map(x -> WirelessAction(x), a))
end
function POMDPs.reward(pomdp::WirelessPOMDP, s::WirelessState, a::Vector{Symbol})
    return POMDPs.reward(pomdp, s, map(x -> eval(x), a))
end
function POMDPs.reward(pomdp::WirelessPOMDP, s::WirelessState, a::Vector{WirelessAction})
    rew = 0.0
    rew += pomdp.send_penalty * sum(a .== WirelessSend)
    return rew - sum(s.queue_length)
end

function POMDPTools.render(pomdp::WirelessPOMDP, step::NamedTuple; title_str="", title_font_size=30)
    # Extract information from the step
    s = get(step, :s, nothing)      # Current state
    b = get(step, :b, nothing)      # Current belief
    a = get(step, :a, nothing)      # Action taken

    num_agents = pomdp.num_agents

    # Create a new plot
    plt = plot(; 
        title=title_str, titlefont=font(title_font_size),
        legend=false, ticks=false, showaxis=false, grid=false, aspectratio=:equal, 
        size=(1000, 1000)
    )

    # Arrange agents in a circle
    agent_angles = [2π * (i-1)/num_agents for i in 1:num_agents]
    radius = 1.0
    agent_coords = [(radius * cos(θ), radius * sin(θ)) for θ in agent_angles]

    # Plot connections based on neighbor relationships
    for i in 1:num_agents
        xi, yi = agent_coords[i]
        for j in neighbors(pomdp.g, i)
            # To avoid duplicate lines, only draw when j > i
            if j > i
                xj, yj = agent_coords[j]
                plot!(plt, [xi, xj], [yi, yj], linecolor=:gray, linewidth=1)
            end
        end
    end
    
    px_p_tick = px_per_tick(plt)
    # Determine font sizes
    fnt_size_text = Int(floor(px_p_tick / 15))
    fnt_size_small = Int(floor(px_p_tick / 30))
    fnt_size_smaller = Int(floor(px_p_tick / 40))
    fnt_size_tiny = Int(floor(px_p_tick / 50))

    # Plot agents
    for (i, (x, y)) in enumerate(agent_coords)
        color = ROBOT_COLORS[(i - 1) % length(ROBOT_COLORS) + 1]
        agent_shape = circ(x, y, 0.13)
        plot!(plt, agent_shape, color=color, alpha=1.0, linecolor=:black)
        annotate!(plt, x, y + 0.025, text("Agent $(i)", :black, :center, fnt_size_tiny))
        
        if !isnothing(s)    
            state_str_source = s.source_state[i] == WirelessPacket ? "Pack" : "Idle"
            state_str = state_str_source * " - " * string(s.queue_length[i])
            annotate!(plt, x, y - 0.025, text(state_str, :black, :center, fnt_size_tiny))
        end

        # If buffer size belief is available, display it
        if !isnothing(b)
            # Compute belief over buffer state and queue size for agent i
            belief_source, belief_buffer_size = compute_buffer_belief_for_agent(pomdp, b, i)

            bar_height = 0.06
            bar_width = 0.3
            
            queue_sizes = collect(0:3)
            num_bars = length(queue_sizes)
            dist_between_bars = bar_height + 0.000
            bar_offsets = [-bar_height*num_bars/2 + dist_between_bars * (i-1) for i in 1:num_bars]
            bar_x = x - 0.33
            bar_y = y .+ bar_offsets
            
            annotate!(plt, bar_x, bar_y[end] + bar_height/2 + 0.07, text("Belief of\nQueue Size", :black, :center, fnt_size_smaller))

            for (by, q_size) in zip(bar_y, queue_sizes)
                prob = get(belief_buffer_size, q_size, 0.0)
                draw_belief_bar!(plt, bar_x, by, prob; height=bar_height, width=bar_width)
                annotate!(plt, bar_x - 0.19, by, text("$(q_size)", :black, :left, fnt_size_tiny))
                annotate!(plt, bar_x, by, text("$(round(prob, digits=2))", :black, :center, fnt_size_tiny))
            end
            
            buffer_states = collect(instances(WirelessSourceState))
            num_bars = length(buffer_states)
            dist_between_bars = bar_height + 0.000
            bar_offsets = [-bar_height*num_bars/2 + dist_between_bars * (i-1) for i in 1:num_bars]
            bar_x = x + 0.36
            bar_y = y .+ bar_offsets

            annotate!(plt, bar_x, bar_y[end] + bar_height/2 + 0.07, text("Belief of\nBuffer State", :black, :center, fnt_size_smaller))
            for (by, bs) in zip(bar_y, buffer_states)
                ss_str = bs == WirelessPacket ? "P" : "I"
                prob = get(belief_source, bs, 0.0)
                draw_belief_bar!(plt, bar_x, by, prob; height=bar_height, width=bar_width)
                annotate!(plt, bar_x - 0.19, by, text("$(ss_str)", :black, :left, fnt_size_tiny))
                annotate!(plt, bar_x, by, text("$(round(prob, digits=2))", :black, :center, fnt_size_tiny))
            end
            
        end
    end

    # Optionally, display action taken
    if !isnothing(a)
        action_names = action_name(pomdp, a)
        # Display action names near agents
        for (i, (x, y)) in enumerate(agent_coords)
            action_text = "$(action_names[i])"
            annotate!(plt, x, y - 0.18, text(action_text, :black, :center, fnt_size_tiny))
            
            if is_action_send(a[i])
                draw_broadcast_symbol!(plt, x, y + 0.1; radius=0.15)
            end            
        end
    end

    plt = dynamic_plot_resize!(plt; max_size=(1000, 1000))
    return plt
end

is_action_send(a::WirelessAction) = a == WirelessSend
is_action_send(a::Int) = is_action_send(WirelessAction(a))
is_action_send(a::Symbol) = is_action_send(eval(a))

# Helper function to compute the belief over buffer sizes for an agent
function compute_buffer_belief_for_agent(pomdp::WirelessPOMDP, b::Any, agent_index::Int)
    belief_source = Dict{WirelessSourceState, Float64}()
    belief_buffer_size = Dict{Int, Float64}()
    if b isa DiscreteBelief
        for (state, prob) in zip(b.state_list, b.b)
            buffer_state = state.source_state[agent_index]
            queue_length = state.queue_length[agent_index]
            belief_source[buffer_state] = get(belief_source, buffer_state, 0.0) + prob
            belief_buffer_size[queue_length] = get(belief_buffer_size, queue_length, 0.0) + prob
        end
    elseif b isa SparseCat
        for (state, prob) in zip(b.vals, b.probs)
            buffer_state = state.source_state[agent_index]
            queue_length = state.queue_length[agent_index]
            belief_source[buffer_state] = get(belief_source, buffer_state, 0.0) + prob
            belief_buffer_size[queue_length] = get(belief_buffer_size, queue_length, 0.0) + prob
        end
    elseif b isa Deterministic
        state = b.val
        buffer_state = state.source_state[agent_index]
        queue_length = state.queue_length[agent_index]
        belief_source[buffer_state] = 1.0
        belief_buffer_size[queue_length] = 1.0
    elseif b isa ParticleCollection
        for part in b.particles
            buffer_state = part.source_state[agent_index]
            queue_length = part.queue_length[agent_index]
            belief_source[buffer_state] = get(belief_source, buffer_state, 0.0) + 1.0
            belief_buffer_size[queue_length] = get(belief_buffer_size, queue_length, 0.0) + 1.0
        end

        for (k, v) in belief_source
            belief_source[k] = v / length(b.particles)
        end
        for (k, v) in belief_buffer_size
            belief_buffer_size[k] = v / length(b.particles)
        end
    else
        error("Unsupported belief type: $(typeof(b))")
    end
    return belief_source, belief_buffer_size
end

action_name(pomdp::WirelessPOMDP, a::Tuple{Vararg}) = action_name(pomdp, collect(a))
action_name(pomdp::WirelessPOMDP, a::Vector{Symbol}) = action_name(pomdp, map(x -> eval(x), a))
action_name(pomdp::WirelessPOMDP, a::Vector{Int}) = action_name(pomdp, map(x -> WirelessAction(x), a))
function action_name(::WirelessPOMDP, a::Vector{WirelessAction})
    return [WIRELESS_ACTION_NAMES[ai] for ai in a]
end
