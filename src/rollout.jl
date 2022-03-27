function estimate_value(estimator::Union{BasicPOMCP.SolvedPORollout,BasicPOMCP.SolvedFORollout}, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int)
    rollout(estimator, pomdp, start_state, h, steps)
end

function rollout(est::BasicPOMCP.SolvedPORollout, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int)
    b = extract_belief(est.updater, h)
    sim = BasicPOMCP.RolloutSimulator(est.rng, steps)
    return POMDPs.simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

function extract_belief end

# some defaults are provided
extract_belief(::BasicPOMCP.NothingUpdater, node::BeliefNode) = nothing

function simulate(sim::BasicPOMCP.RolloutSimulator, pomdp::POMDP, policy::Policy, updater::Updater, initial_belief, s)
    if sim.eps == nothing
        eps = 0.0
    else
        eps = sim.eps
    end

    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    disc = 1.0
    r_total = 0.0
    b = initialize_belief(updater, initial_belief)
    step = 1
    action_list = []

    while disc > eps && !isterminal(pomdp, s) && step <= max_steps
        a = action(policy, b)
        push!(action_list, a)
        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a, sim.rng)
        r_total += disc*r
        s = sp
        bp = update(updater, b, a, o)
        b = bp
        disc *= discount(pomdp)
        step += 1
    end
    
    return r_total, action_list
end