using Distributed
addprocs(16)
@everywhere using Opaac

using Random


@everywhere const air_cap = 1000
@everywhere const air_cost = 1000
@everywhere const cap_range = 10:100
@everywhere const cost_range = 100:200
@everywhere const upload_cost = 1
@everywhere const download_cost = 0
@everywhere const vol_range = 1:10

function main()
    mkpath("instances")
    mkpath("partials")
    data = [(n=n, m=m) 
        for n in 11:10:201,
        m in [1, 10, 100, 1000]]
    pmap(data) do x
        n, m = x.n, x.m
        @show (n, m)
        fn = "instances/inst-$n-$m.txt"
        mps_fn = "partials/inst-$n-$m"
        isfile(fn) && return

        lg = mk_land_graph(n, n; cap_range=cap_range, cost_range=vol_range)
        D = @time gen_instance(lg, m; vol_range=vol_range, feas_only=false, fn=mps_fn)
        inst = Instance(lg, D, upload_cost, download_cost, air_cap)
        write_instance(fn, inst)
        nothing
    end
end
main()