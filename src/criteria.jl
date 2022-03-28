struct MaxUCB
    c::Float64
    b::Float64
end

function select_best(crit::MaxUCB, h_node::POWRAVETreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_criterion_val = -Inf
    local best_node::Int
    istied = false
    local tied::Vector{Int}
    ltn = log(tree.total_n[h])
    for node in tree.tried[h]
        n = tree.n[node]
        n_hat = tree.n_hat[node]
        if n + n_hat + 4 * n * n_hat * crit.b * crit.b == 0
            β = 0
        else
            β = n_hat / (n + n_hat + 4 * n * n_hat * crit.b * crit.b)
        end
        if n == 0 && ltn <= 0.0
            criterion_value = (1 - β) * tree.v[node] + β * tree.v_hat[node]
        elseif n == 0 && (1 - β) * tree.v[node] + β * tree.v_hat[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = ((1 - β) * tree.v[node] + β * tree.v_hat[node]) + crit.c*sqrt(ltn/n)
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            best_node = node
            istied = false
        elseif criterion_value == best_criterion_val
            if istied
                push!(tied, node)
            else
                istied = true
                tied = [best_node, node]
            end
        end
    end
    if istied
        return rand(rng, tied)
    else
        return best_node
    end
end

struct MaxQ 
    b::Float64
end

function select_best(crit::MaxQ, h_node::POWRAVETreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_v = tree.v[best_node]
    @assert !isnan(best_v)
    for node in tree.tried[h][2:end]
        n = tree.n[node]
        n_hat = tree.n_hat[node]
        if n + n_hat + 4 * n * n_hat * crit.b * crit.b == 0
            β = 0
        else
            β = n_hat / (n + n_hat + 4 * n * n_hat * crit.b * crit.b)
        end
        if (1 - β) * tree.v[node] + β * tree.v_hat[node] >= best_v
            best_v = (1 - β) * tree.v[node] + β * tree.v_hat[node]
            best_node = node
        end
    end
    return best_node
end

struct MaxTries end

function select_best(crit::MaxTries, h_node::POWRAVETreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_n = tree.n[best_node]
    @assert !isnan(best_n)
    for node in tree.tried[h][2:end]
        if tree.n[node] >= best_n
            best_n = tree.n[node]
            best_node = node
        end
    end
    return best_node
end
