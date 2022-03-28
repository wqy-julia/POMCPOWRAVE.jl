using POMDPs
using POMCPOWRAVE
using ProfileView
using POMDPModels

#=
using Gallium
breakpoint(Pkg.dir("POMCPOWRAVE", "src", "solver.jl"), 40)
=#

solver = POMCPOWRAVESolver(tree_queries=50_000,
                     eps=0.01,
                     c=10.0,
                     enable_action_pw=false,
                     alpha_observation=1/8,
                     rng=MersenneTwister(2))

problem = LightDark1D()
policy = solve(solver, problem)
ib = initial_state_distribution(problem)
a = action(policy, ib)

@time a = action(policy, ib)

Profile.clear()
@profile a = action(policy, ib)
ProfileView.view()
