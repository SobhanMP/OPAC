using Infiltrator
using Revise
using Opaac
using Graphs
using Printf
using MetaGraphs
using JuMP
using Cthulhu
Revise.errors()

const inst_list = filter(map(readdir("instances")) do x
    n,m = parse.(Int, split(x[1:end-4], "-")[2:end])
    (n=n, m=m, fn="instances/"*x)
end) do x
    x.n < 20
end

function fun_colgen(n, inst, m)
    ag, a2l, l2a = mk_air_graph(n, n, n-2)
    mg = combgraph(inst.lg, ag, a2l, 0)
    l0 = []

    colgen_state = init_colgen(inst.D, mg, inst.lg, ag, [], nothing, inst.upload_cost, inst.download_cost;  air_cap=inst.air_cap)
    colgen(colgen_state)
    
    objective_value(colgen_state.model)
end
function show_table(cols...)
    d = []
    for cc in cols,
        c in cc
        push!(d, c.i)
    end
    d = sort(unique(d); by=x->(x.n, x.m))
    join([string(e.n) * " & " * string(e.m) * " & " *join([begin
        l = [i for i in c if i.i == e]
        @assert length(l) <= 1
        if length(l) == 1
            @sprintf "%.1f" (l[1].time) # * "/" * string(l[1].val)
        else
            ""
        end
    end for c in cols], " & ")
    for e in d], " \\\\\n") * " \\\\"
end

function main()
    # warmup
    our = []
    for m in [multiflow_gurobi, multiflow_cplex]
        @time fun_colgen(inst_list[1].n, load_instance(inst_list[1].fn), m())
        push!(our, []) 
        for i in inst_list
            inst = load_instance(i.fn)
            t = @elapsed val = fun_colgen(i.n, inst, m())
            push!(our[end], (i=i, time=t, val=val))
        end
    end
    solver = []
    for m in [multiflow_gurobi, multiflow_cplex]
        @time let l =  load_instance(inst_list[1].fn)
            init_solve_land(l.D, l.lg; model=m())
        end;
        push!(solver, [])
        for i in inst_list
            inst = load_instance(i.fn)
            t = @elapsed state = init_solve_land(inst.D, inst.lg; model=m());
            push!(solver[end], (i=i, time=t, val=objective_value(state.model)))
        end
    end
    
    x = show_table(our..., solver...)
    println(x)
    open("col-table.txt", "w") do fd
        println(fd, x)
    end
end
main()
