using POMDPs
using POMCPOWRAVE
using ProfileView
using POMDPModels

#=
using Gallium
breakpoint(Pkg.dir("POMCPOWRAVE", "src", "solver2.jl"), 100)
=#

solver = POMCPOWRAVESolver(tree_queries=250_000,
                     eps=0.01,
                     criterion=MaxUCB(10.0),
                     enable_action_pw=true,
                     check_repeat_obs=false,
                     alpha_observation=1/8,
                     rng=MersenneTwister(2))

problem = LightDark1D()
policy = POMCPOWRAVEPlanner(solver, problem)
ib = initial_state_distribution(problem)

a = action(policy, ib)

@time a = action(policy, ib)

Profile.clear()
@profile a = action(policy, ib)
ProfileView.view()

#=
P = typeof(problem)
S = state_type(P)
A = action_type(P)
O = obs_type(P)
tree = POMCPOWRAVETree{POWRAVENodeBelief{S,A,O,P},A,O,typeof(ib)}(ib, 500_000)

@code_warntype POMCPOWRAVE.simulate(policy, POMCPOWRAVE.POWRAVETreeObsNode(tree, 1), rand(Base.GLOBAL_RNG, ib), 10)
=#

#=
solver = POMCPOWRAVESolver(tree_queries=30,
                     eps=0.01,
                     criterion=MaxUCB(10.0),
                     enable_action_pw=true,
                     check_repeat_obs=false,
                     alpha_observation=1/8,
                     rng=MersenneTwister(2))

policy = POMCPOWRAVEPlanner(solver, problem)
a = action(policy, ib)
blink(policy)
=#
