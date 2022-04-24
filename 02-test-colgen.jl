# using Infiltrator
using Serialization
# using Revise
using OPAC
using Graphs
using Printf
using MetaGraphs
using JuMP
# using Cthulhu


const inst_ind = 1
const inst_list = sort(filter(map(readdir("instances")) do x
    n,m = parse.(Int, split(x[1:end-4], "-")[2:end])
    (n=n, m=m, fn="instances/"*x)
end) do x
    x.n <= 41 && x.m <= 1000
end, by=x->(x.n, x.m))
function fun_colgen(n, inst, m)
    ag, a2l, l2a = mk_air_graph(n, n, n-2)
    mg = combgraph(inst.lg, ag, a2l, 0, download_cost=0, upload_cost=0)

    colgen_state = init_colgen(inst.D, mg, inst.lg, ag, [], nothing, 
    inst.upload_cost, inst.download_cost;  air_cap=inst.air_cap, model=m)
    colgen(colgen_state)
    
    objective_value(colgen_state.model), termination_status(colgen_state.model)
end

# warmup
for m in [multiflow_gurobi, multiflow_cplex]
    fun_colgen(inst_list[1].n, load_instance(inst_list[1].fn), m())
    let inst = load_instance(inst_list[1].fn)
        init_solve_land(inst.D, inst.lg; model=m())
    end
end

show_table(res; show_loss=false) = join([string(hk.n) * " \t & \t  " * string(hk.m) * " \t & \t " * join([
    let j = (mode=i, hk...)
        if j ∈ keys(res)
            l = res[j]
            if show_loss
                @sprintf " %.2e " l.val[1]
            else
                if l.val[2] == MOI.OPTIMAL # optimal
                    @sprintf " %.2f " l.time
                else
                    ""
                end
            end
        else
            ""
        end
    end
    for i in ["CG", "CC", "G", "C"]], " \t & \t ") * " \\\\ \n"
    for hk in sort(unique([(n=k.n, m=k.m) for k in keys(res)]))])



function main(fn, res)
    
    for i in inst_list
        for (m, mn) in zip([multiflow_gurobi, multiflow_cplex], ["G", "C"])
            key = (mode="C" * mn, n=i.n, m=i.m)
            if key ∉ keys(res)
                inst = load_instance(i.fn)
                t = @elapsed val = fun_colgen(i.n, inst, m())
                res[key] = (i=i, time=t, val=val)
                fn !== nothing && serialize(fn, res)
            end
            key = (mode=mn, n=i.n, m=i.m)
            if keys ∉ keys(res)
                inst = load_instance(i.fn)
                t = @elapsed state = begin
                    state = init_solve_land(inst.D, inst.lg; model=m());
                    objective_value(state.model), termination_status(state.model)
                end
                res[key] = (i=i, time=t, val=state)
                fn !== nothing && serialize(fn, res)
            end
        end        
        x = show_table(res)
        println(x)
        if fn !== nothing
            open(fn * ".txt", "w") do fd
                println(fd, x)
            end
        end
    end
end

main("col.jld", try
    deserialize("col.jld")
catch e
    println("ignoring error ", e)
    Dict()
end)
