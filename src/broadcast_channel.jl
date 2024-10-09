@enum BroadcastAction BroadcastSend BroadcastListen
@enum BroadcastState BufferEmpty BufferFull
@enum BroadcastObservation BroadcastCollision BroadcastSuccess BroadcastNothing

const BROADCAST_ACTION_NAMES = Dict(BroadcastSend => "Send", BroadcastListen => "Listen")

struct BroadcastChannelPOMDP <: POMDP{Vector{BroadcastState},Vector{BroadcastAction},Vector{Tuple{BroadcastObservation,BroadcastState}}}
    num_agents::Int
    discount_factor::Float64
    buffer_fill_prob::Vector{Float64}
    correct_obs_prob::Vector{Float64}
    success_reward::Float64
    observation_agent::Int
    send_penalty::Float64
end

function BroadcastChannelPOMDP(;
    num_agents::Int=2,
    discount_factor::Float64=0.9,
    buffer_fill_prob::Vector{Float64}=[0.9, 0.1],
    correct_obs_prob::Vector{Float64}=[1.0],
    success_reward::Float64=1.0,
    observation_agent::Int=0,
    send_penalty::Float64=0.0
)
    if length(buffer_fill_prob) == 1
        buffer_fill_prob = fill(buffer_fill_prob[1], num_agents)
    end
    if length(correct_obs_prob) == 1
        correct_obs_prob = fill(correct_obs_prob[1], num_agents)
    end

    0 <= observation_agent <= num_agents || throw(ArgumentError("Observation agent $observation_agent must be between 0 and $num_agents"))
    num_agents == length(buffer_fill_prob) || throw(ArgumentError("Number of agents $num_agents must be equal to the length of buffer_fill_prob $(length(buffer_fill_prob))"))
    num_agents == length(correct_obs_prob) || throw(ArgumentError("Number of agents $num_agents must be equal to the length of correct_obs_prob $(length(correct_obs_prob))"))

    return BroadcastChannelPOMDP(num_agents, discount_factor, buffer_fill_prob,
        correct_obs_prob, success_reward, observation_agent, send_penalty)
end

POMDPs.states(pomdp::BroadcastChannelPOMDP) = vec([collect(s) for s in Base.product(fill([BufferEmpty, BufferFull], pomdp.num_agents)...)])

POMDPs.stateindex(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}) = findfirst(==(s), states(pomdp))

POMDPs.actions(pomdp::BroadcastChannelPOMDP) = vec([collect(a) for a in Base.product(fill([BroadcastSend, BroadcastListen], pomdp.num_agents)...)])

POMDPs.actionindex(pomdp::BroadcastChannelPOMDP, a::Vector{BroadcastAction}) = findfirst(==(a), actions(pomdp))

"""
    observations(pomdp::BroadcastChannelPOMDP)
   
Return the observations for the broadcast channel pomdp. Each agent observes whether there 
    was a collision or success with the given `correct_obs_prob`. Each agent also observes
    wehther its own buffer is full or empty with 1.0 probability.
"""
function POMDPs.observations(pomdp::BroadcastChannelPOMDP)
    if pomdp.observation_agent == 0
        # joint observations is the cartesian product of the individual observations
        indiv_obs_flat = reduce(vcat, individual_observations(pomdp))
        return vec([collect(o) for o in Base.product(fill(indiv_obs_flat, pomdp.num_agents)...)])
    else
        return individual_observations(pomdp)
    end
end

function individual_observations(::BroadcastChannelPOMDP)
    # all possible observations for a single agent
    channel_obs = [BroadcastCollision, BroadcastSuccess, BroadcastNothing]
    buffer_obs = [BufferEmpty, BufferFull]
    obs_opts = Vector{Vector{Tuple{BroadcastObservation,BroadcastState}}}(undef, 5)
    cnt = 0
    for channel_ob in channel_obs
        for buffer_ob in buffer_obs
            if channel_ob == BroadcastCollision && buffer_ob == BufferEmpty
                continue # If we had a collision, then we had a buffer and tried to send
            end
            cnt += 1
            obs_opts[cnt] = [(channel_ob, buffer_ob)]
        end
    end
    return obs_opts
end

function POMDPs.obsindex(pomdp::BroadcastChannelPOMDP, o::Vector{Tuple{BroadcastObservation,BroadcastState}})
    obs_opts = observations(pomdp)
    return findfirst(==(o), obs_opts)
end

POMDPs.discount(pomdp::BroadcastChannelPOMDP) = pomdp.discount_factor

POMDPs.transition(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Tuple{Vararg}) = POMDPs.transition(pomdp, s, collect(a))
function POMDPs.transition(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Vector{Int})
    return POMDPs.transition(pomdp, s, map(x -> BroadcastAction(x), a))
end
function POMDPs.transition(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Vector{Symbol})
    return POMDPs.transition(pomdp, s, map(x -> eval(x), a))
end
function POMDPs.transition(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Vector{BroadcastAction})
    send_and_has_message = (a .== BroadcastSend) .& (s .== BufferFull)
    new_probs_states = Vector{Vector{Tuple{Float64,BroadcastState}}}(undef, pomdp.num_agents)
    for ii in 1:pomdp.num_agents
        new_probs_states_ii = Vector{Tuple{Float64,BroadcastState}}()
        if s[ii] == BufferEmpty
            push!(new_probs_states_ii, (pomdp.buffer_fill_prob[ii], BufferFull))
            push!(new_probs_states_ii, (1 - pomdp.buffer_fill_prob[ii], BufferEmpty))
        else # BufferFull
            if a[ii] == BroadcastListen || (sum(send_and_has_message) > 1)
                push!(new_probs_states_ii, (1.0, BufferFull))
            else
                push!(new_probs_states_ii, (pomdp.buffer_fill_prob[ii], BufferFull))
                push!(new_probs_states_ii, (1 - pomdp.buffer_fill_prob[ii], BufferEmpty))
            end
        end
        new_probs_states[ii] = new_probs_states_ii
    end

    new_states = Vector{Vector{BroadcastState}}()
    new_probs = Vector{Float64}()

    for ii in 1:pomdp.num_agents
        new_states_ii = Vector{Vector{BroadcastState}}()
        new_probs_ii = Vector{Float64}()
        if ii == 1
            for (p, s) in new_probs_states[ii]
                push!(new_states_ii, [s])
                push!(new_probs_ii, p)
            end
        else
            for jj in 1:length(new_states)
                for (p, s) in new_probs_states[ii]
                    push!(new_states_ii, [new_states[jj]..., s])
                    push!(new_probs_ii, p * new_probs[jj])
                end
            end
        end
        new_states = new_states_ii
        new_probs = new_probs_ii
    end
    return SparseCat(new_states, new_probs)
end

function POMDPs.observation(pomdp::BroadcastChannelPOMDP, a::Vector{BroadcastAction}, sp::Vector{BroadcastState})
    possible_states = POMDPs.states(pomdp)
    
    obs_probs = Dict{Vector{Tuple{BroadcastObservation,BroadcastState}}, Float64}()
    
    for s in possible_states
        trans_dist = POMDPs.transition(pomdp, s, a)
        trans_prob = pdf(trans_dist, sp)
        
        if trans_prob > 0
            obs_dist = POMDPs.observation(pomdp, s, a, sp)
            for (obs, obs_prob) in zip(obs_dist.vals, obs_dist.probs)
                obs_probs[obs] = get(obs_probs, obs, 0.0) + trans_prob * obs_prob
            end
        end
    end
    
    obs_prob_vec = Vector{Float64}()
    obs_vec = Vector{Vector{Tuple{BroadcastObservation,BroadcastState}}}()
    for (k, v) in obs_probs
        if v > 0
            push!(obs_vec, k)
            push!(obs_prob_vec, v)
        end
    end
    
    normalize!(obs_prob_vec, 1.0)

    return SparseCat(obs_vec, obs_prob_vec)
end

POMDPs.observation(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Tuple{Vararg}, sp::Vector{BroadcastState}) = POMDPs.observation(pomdp, s, collect(a), sp)
function POMDPs.observation(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Vector{Int}, sp::Vector{BroadcastState})
    return POMDPs.observation(pomdp, s, map(x -> BroadcastAction(x), a), sp)
end
function POMDPs.observation(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Vector{Symbol}, sp::Vector{BroadcastState})
    return POMDPs.observation(pomdp, s, map(x -> eval(x), a), sp)
end
function POMDPs.observation(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState},
    a::Vector{BroadcastAction}, sp::Vector{BroadcastState}
)
    if pomdp.observation_agent == 0
        indvidual_obs_dists = [individual_observation(pomdp, s, a, sp, ii) for ii in 1:pomdp.num_agents]
        inidividal_obs = [reduce(vcat, individual_obs_dist.vals) for individual_obs_dist in indvidual_obs_dists]
        inidividal_obs_probs = [individual_obs_dist.probs for individual_obs_dist in indvidual_obs_dists]

        joint_obs_vals = vec([collect(o) for o in Base.product(inidividal_obs...)])
        joint_obs_probs = [prod(probs) for probs in Base.product(inidividal_obs_probs...)]

        return SparseCat(joint_obs_vals, joint_obs_probs)

    else
        return individual_observation(pomdp, s, a, sp, pomdp.observation_agent)
    end
end

function individual_observation(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState},
    a::Vector{BroadcastAction}, sp::Vector{BroadcastState}, ii::Int
)
    send_and_has_message = (a .== BroadcastSend) .& (s .== BufferFull)
    collision_exist = sum(send_and_has_message) > 1
    correct_obs = collision_exist ? BroadcastCollision : BroadcastSuccess
    incorrect_obs = collision_exist ? BroadcastSuccess : BroadcastCollision

    if (a[ii] == BroadcastListen) || (a[ii] == BroadcastSend && s[ii] == BufferEmpty)
        return SparseCat([[(BroadcastNothing, sp[ii])]], [1.0])
    elseif sp[ii] == BufferFull
        return SparseCat([[(correct_obs, sp[ii])], [(incorrect_obs, sp[ii])]], [pomdp.correct_obs_prob[ii], 1 - pomdp.correct_obs_prob[ii]])
    elseif sp[ii] == BufferEmpty
        # s[ii] == BufferFull and sp[ii] == BufferEmpty, therefore we must of successfully sent
        return SparseCat([[(BroadcastSuccess, sp[ii])]], [1.0])
    else
        error("Invalid state or action")
    end
end

POMDPs.reward(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Tuple{Vararg}) = POMDPs.reward(pomdp, s, collect(a))
function POMDPs.reward(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Vector{Int})
    return POMDPs.reward(pomdp, s, map(x -> BroadcastAction(x), a))
end
function POMDPs.reward(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Vector{Symbol})
    return POMDPs.reward(pomdp, s, map(x -> eval(x), a))
end
function POMDPs.reward(pomdp::BroadcastChannelPOMDP, s::Vector{BroadcastState}, a::Vector{BroadcastAction})
    rew = 0.0
    rew += pomdp.send_penalty * sum(a .== BroadcastSend)
    if all(a .== BroadcastListen)
        return rew
    elseif sum(a .== BroadcastSend) > 1
        return rew
    end
    # Made it here, so only one agent is sending
    ii = findfirst(a -> a == BroadcastSend, a)
    if s[ii] == BufferFull
        return rew + pomdp.success_reward
    else
        return rew
    end
end

function POMDPs.initialstate(pomdp::BroadcastChannelPOMDP)
    # Start with highest probability of fill being full and others being empty
    s = fill(BufferEmpty, pomdp.num_agents)
    idx_highest_fill = argmax(pomdp.buffer_fill_prob)
    s[idx_highest_fill] = BufferFull
    return SparseCat([s], [1.0])
end

# Helper function to compute the belief that an agent's buffer is full
function compute_belief_for_agent(::BroadcastChannelPOMDP, b::DiscreteBelief, agent_index::Int, state_value::BroadcastState)
    belief = 0.0
    for (state, prob) in zip(b.state_list, b.b)
        if prob > 0 && state[agent_index] == state_value
            belief += prob
        end
    end
    return belief
end
function compute_belief_for_agent(::BroadcastChannelPOMDP, b::SparseCat, agent_index::Int, state_value::BroadcastState)
    belief = 0.0
    for (state, prob) in zip(b.vals, b.probs)
        if state[agent_index] == state_value
            belief += prob
        end
    end
    return belief
end

# Main render function
function POMDPTools.render(pomdp::BroadcastChannelPOMDP, step::NamedTuple; title_str="", title_font_size=30)
    # Extract information from the step
    s = get(step, :s, nothing)      # Current state
    b = get(step, :b, nothing)      # Current belief
    a = get(step, :a, nothing)      # Action taken
    num_agents = pomdp.num_agents

    # Create a new plot
    plt = plot(; 
        title=title_str, titlefont=font(title_font_size),
        legend=false, ticks=false, showaxis=false, 
        grid=false, aspectratio=:equal,
        size=(800, 800)
    )

    # Set up positions for agents
    x_positions = collect(1:num_agents)
    y_position = 0.0
    
    x_mid = sum(x_positions) / num_agents
    # Blank spot at top for title
    plot!(plt, rect(0.0, 0.2, x_mid, 1.3); 
        linecolor=:white, color=:white)
    # Blank spot at bottom
    plot!(plt, rect(0.0, 0.1, x_mid, 0.05); 
        linecolor=:white, color=:white)

    px_p_tick = px_per_tick(plt)

    # Determine font sizes
    fnt_size_text = Int(floor(px_p_tick / 20))
    fnt_size_smaller = Int(floor(px_p_tick / 25))
    fnt_size_belief = Int(floor(px_p_tick / 35))
        
    if !isnothing(b)
        # Annotate "Belief of Buffer Full" above the belief bars
        x_center = sum(x_positions) / num_agents
        annotate!(plt, x_center, y_position + 0.01 , text("Belief of Buffer Full", :black, :center, fnt_size_smaller))
    end
    
    # Draw agents
    for i in 1:num_agents
        agent_color = :white
        buffer_text = ""
        x_pos = x_positions[i]

        if !isnothing(s)
            buff_state = s[i]
            agent_color = buff_state == BufferFull ? :gray : :white
            buffer_text = buff_state == BufferFull ? "Full" : "Empty"
        end
        
        # Draw the agent as a circle
        
        plot!(plt, circ(x_pos, y_position + 1.0, 0.3; num_pts=50); color=agent_color, linewidth=1)
        annotate!(plt, x_pos, y_position + 1.0, text(buffer_text, :black, :center, fnt_size_smaller))
        annotate!(plt, x_pos, y_position + 0.6, text("Agent $i", :black, :center, fnt_size_text))

        # Draw the belief bar below the agent
        
        if !isnothing(b)
            # Compute the belief that the agent's buffer is full
            belief_full = compute_belief_for_agent(pomdp, b, i, BufferFull)
            draw_belief_bar!(plt, x_pos, y_position + 0.2, belief_full; width=0.6, height=0.15)
            annotate!(plt, x_pos, y_position + 0.2, text("$(round(belief_full, digits=2))", :black, :center, fnt_size_belief))
        end

        # If action is provided, check if the agent is sending
        if !isnothing(a)
            action_i = a[i]
            if action_i == BroadcastSend
                # Draw broadcast symbol
                draw_broadcast_symbol!(plt, x_pos, y_position + 1.2)
            end
            # Optionally, display the action
            action_text = action_name(pomdp, a)[i]
            annotate!(plt, x_pos, y_position + 0.4, text("$(action_text)", :black, :center, fnt_size_smaller))
        end
    end

    plt = dynamic_plot_resize!(plt; max_size=(800, 800))
    return plt
end

action_name(pomdp::BroadcastChannelPOMDP, a::Tuple{Vararg}) = action_name(pomdp, collect(a))
action_name(pomdp::BroadcastChannelPOMDP, a::Vector{Symbol}) = action_name(pomdp, map(x -> eval(x), a))
action_name(pomdp::BroadcastChannelPOMDP, a::Vector{Int}) = action_name(pomdp, map(x -> BroadcastAction(x), a))
function action_name(::BroadcastChannelPOMDP, a::Vector{BroadcastAction})
    return string.([BROADCAST_ACTION_NAMES[ai] for ai in a])
end
