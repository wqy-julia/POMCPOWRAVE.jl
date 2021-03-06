function simulate(pomcp::POMCPOWRAVEPlanner, h_node::POWRAVETreeObsNode{B,A,O}, s::S, d) where {B,S,A,O}
    tree = h_node.tree
    h = h_node.node
    sol = pomcp.solver
    dep = 1.0
    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return 0.0, dep, []
    end

    if sol.enable_action_pw
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action * total_n ^ sol.alpha_action
        # if length(tree.tried[h]) < 1
            if h == 1
                a = next_action(pomcp.next_action, pomcp.problem, tree.root_belief, POWRAVETreeObsNode(tree, h))
            else
                a = next_action(pomcp.next_action, pomcp.problem, StateBelief(tree.sr_beliefs[h]), POWRAVETreeObsNode(tree, h))
            end
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWRAVETreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWRAVETreeObsNode(tree, h), a),
                            sol.check_repeat_act)
            end
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            if h == 1
                action_space_iter = POMDPs.actions(pomcp.problem, tree.root_belief)
            else
                action_space_iter = POMDPs.actions(pomcp.problem, StateBelief(tree.sr_beliefs[h]))
            end
            anode = length(tree.n)
            for a in action_space_iter
                # update_lookup false to true
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWRAVETreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWRAVETreeObsNode(tree, h), a),
                            true)
            end
        end
    end
    total_n = tree.total_n[h]

    best_node = select_best(pomcp.criterion, h_node, pomcp.solver.rng)
    a = tree.a_labels[best_node]

    new_node = false
    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)
    # if tree.n_a_children[best_node] < 1

        sp, o, r = @gen(:sp, :o, :r)(pomcp.problem, s, a, sol.rng)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            new_node = true
            hao = length(tree.sr_beliefs) + 1
            push!(tree.sr_beliefs,
                  init_node_sr_belief(pomcp.node_sr_belief_updater,
                                      pomcp.problem, s, a, sp, o, r))
            push!(tree.total_n, 0)
            push!(tree.tried, Int[])
            push!(tree.o_labels, o)

            if sol.check_repeat_obs
                tree.a_child_lookup[(best_node, o)] = hao
            end
            tree.n_a_children[best_node] += 1
        end
        push!(tree.generated[best_node], o=>hao)
    else
        sp, r = @gen(:sp, :r)(pomcp.problem, s, a, sol.rng)

    end

    if r == Inf
        @warn("POMCPOWRAVE: +Inf reward. This is not recommended and may cause future errors.")
    end

    if new_node
        dep = dep + 1.0
        est_value, action_list = estimate_value_action(pomcp.solved_estimate, pomcp.problem, sp, POWRAVETreeObsNode(tree, hao), d-1)
        R = r + POMDPs.discount(pomcp.problem) * est_value
    else
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)
        sp, r = rand(sol.rng, tree.sr_beliefs[hao])
        reward, depth, action_list = simulate(pomcp, POWRAVETreeObsNode(tree, hao), sp, d-1)
        dep = depth + 1
        # dep = max(depth + 1, dep)
        R = r + POMDPs.discount(pomcp.problem) * reward
        # R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWRAVETreeObsNode(tree, hao), sp, d-1)
    end

    tree.n[best_node] += 1
    tree.total_n[h] += 1
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
    end

    current_action_list = []
    push!(current_action_list, a)
    for act in eachindex(action_list)
        push!(current_action_list, action_list[act])
        if haskey(tree.o_child_lookup, (h,action_list[act]))
            anode = tree.o_child_lookup[(h,action_list[act])]
            tree.n_hat[anode] += 1
            if tree.v_hat[anode] != -Inf
                tree.v_hat[anode] += (R-tree.v_hat[anode])/tree.n_hat[anode]
            end
        end
    end
    return R, dep, current_action_list
end

