using POMCPOWRAVE
using Test

using POMDPs
using POMDPModels
using ParticleFilters
using POMDPTesting
using D3Trees
using BeliefUpdaters
using POMDPModelTools
using POMDPPolicies

@testset "all" begin

    @testset "POMDPTesting" begin
        solver = POMCPOWRAVESolver()
        pomdp = BabyPOMDP()
        test_solver(solver, pomdp, updater=DiscreteUpdater(pomdp))
        test_solver(solver, pomdp)

        solver = POMCPOWRAVESolver(max_time=0.1, tree_queries=typemax(Int))
        test_solver(solver, pomdp, updater=DiscreteUpdater(pomdp))
    end

    @testset "type stability" begin
        # make sure internal function is type stable
        pomdp = BabyPOMDP()
        solver = POMCPOWRAVESolver()
        planner = solve(solver, pomdp)
        b = initialstate_distribution(pomdp)
        B = POMCPOWRAVE.belief_type(POMCPOWRAVE.POWRAVENodeFilter, typeof(pomdp))
        tree = POMCPOWRAVETree{B,Bool,Bool,typeof(b)}(b, 2*planner.solver.tree_queries)
        @inferred POMCPOWRAVE.simulate(planner, POMCPOWRAVE.POWRAVETreeObsNode(tree, 1), true, 10)

        pomdp = LightDark1D()
        solver = POMCPOWRAVESolver(default_action=485)
        planner = solve(solver, pomdp)

        b = ParticleCollection([LightDark1DState(-1, 0)])
        @test @test_logs (:warn,) @inferred(action(planner, b)) == 485

        b = initialstate_distribution(pomdp)
        @inferred action(planner, b)
    end;

    @testset "currentobs and history" begin
        pomdp = BabyPOMDP()
        solver = POMCPOWRAVESolver()
        planner = solve(solver, pomdp)
        b = initialstate_distribution(pomdp)
        B = POMCPOWRAVE.belief_type(POMCPOWRAVE.POWRAVENodeFilter, typeof(pomdp))
        tree = POMCPOWRAVETree{B,Bool,Bool,typeof(b)}(b, 2*planner.solver.tree_queries)

        n = POMCPOWRAVE.POWRAVETreeObsNode(tree, 1)
        nb = belief(n)
        # we can't call current obs on the root node
        @test_throws MethodError currentobs(nb)
        # simulate the tree to expand one step
        POMCPOWRAVE.simulate(planner, n, true, 1)
        n = POMCPOWRAVE.POWRAVETreeObsNode(tree, 2)
        nb = belief(n)
        # but at a non-root node, this should work
        @test currentobs(nb) isa Bool
        @test currentobs(nb) == history(nb)[end].o
        @test history(nb)[end].a isa Bool
    end;

    @testset "D3tree" begin
        # make sure internal function is type stable
        pomdp = BabyPOMDP()
        solver = POMCPOWRAVESolver()
        planner = solve(solver, pomdp)
        b = initialstate_distribution(pomdp)
        a, info = action_info(planner, b)
        # d3t = D3Tree(planner)
        @test_throws KeyError d3t = D3Tree(info[:tree])

        a, info = action_info(planner, b, tree_in_info=true)
        # d3t = D3Tree(planner)
        d3t = D3Tree(info[:tree])
        # inchrome(d3t)
    end;

    @testset "actionvalues" begin
        pomdp = BabyPOMDP()
        solver = POMCPOWRAVESolver()
        planner = solve(solver, pomdp)
        b = initialstate_distribution(pomdp)
        @test actionvalues(planner, b) isa AbstractVector
    end

    @testset "init_node_sr_belief_error" begin
        include("init_node_sr_belief_error.jl")
    end;
end;
