using Distributed
@show nprocs()
@everywhere using Revise
@everywhere using Opaac
@everywhere using Distributions
@everywhere using Random
@everywhere using JuMP


@everywhere const air_cap = 10_000
@everywhere const air_cost = 100_000
@everywhere const cap_range = Truncated(Exponential(), 1, 1.05)
@everywhere const cost_range = 100:200
@everywhere const upload_cost = 1
@everywhere const download_cost = 0
@everywhere const vol_range = Truncated(Poisson(10), 1, Inf)
@everywhere function f(x)
    n, m = x.n, x.m
    
    fn = "instances/inst-$n-$m.txt"
    
    # isfile(fn) && return
    
    lg = mk_land_graph(n, n; cap_range=cap_range, cost_range=cost_range)
    Random.seed!(1923)
    
    now = time()
    lg, D = gen_instance(lg, m, 5, vol_range, cap_range)
    old_time = time() - now
    inst = Instance(lg, D, upload_cost, download_cost, air_cap)
    write_instance(fn, inst)
    @assert load_instance(fn) == inst
    
    now = time()
    state = init_solve_land(D, lg)
    @assert termination_status(state.model) == MOI.OPTIMAL
    solve_time = time() - now

    println("n: ", n, " m: ", m, " |D|: ", length(D), " t: ", old_time, " ", solve_time)
    nothing
end
function main()
    mkpath("instances")
    mkpath("partials")
    data = [(n=n, m=m) 
        for m in [10, 100, 500, 1000] for n in [6, 11, 21, 51, 101]]
    pmap(f, data)
end
main()