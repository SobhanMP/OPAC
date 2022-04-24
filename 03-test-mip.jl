# using Infiltrator
# using Revise
using OPAC
using Graphs
using Serialization
using Printf
using MetaGraphs
using JuMP
# using Cthulhu
Revise.errors()
const air_cost=1_000
const air_cap=10_000
const L=4000
const inst_ind = 1
const inst_list = sort(filter(map(readdir("instances")) do x
    n,m = parse.(Int, split(x[1:end-4], "-")[2:end])
    (n=n, m=m, fn="instances/"*x)
end) do x
    x.n <= 41 && x.m <= 1000
end, by=x->(x.n, x.m))

function load_inst(i, s, P)
    inst = load_instance(i.fn)
    ag, a2l, l2a = mk_air_graph(i.n, i.n, s)
    mg = combgraph(inst.lg, ag, a2l, P; download_cost=inst.download_cost, upload_cost=inst.upload_cost)
    inst, P, mg, ag, l2a
end

obj_if(model) =  if termination_status(model) == MOI.OPTIMAL
    objective_value(model), 0.0
else
    if has_values(model)
        objective_value(model), relative_gap(model)
    else
        -1, relative_gap(model)
    end
end

function fun_benders_gen(inst, P, mg, ag, l2a, m)
    model = master_colgen(inst.D, P, mg, inst.lg, ag, l2a; 
        air_cost=air_cost, air_cap=air_cap, L=L, 
        model=m, download_cost=inst.download_cost, upload_cost=inst.upload_cost)

    obj_if(model)
end

function fun_benders(inst, P, mg, ag, l2a, m, im)
    model = master_problem(inst.D, P, inst.lg, ag, l2a; 
        air_cost=air_cost, air_cap=air_cap, L=L, 
        model=m, download_cost=inst.download_cost, upload_cost=inst.upload_cost, inner_model=im)
    obj_if(model)
end



function fun_native(inst, P, mg, ag, l2a, m)
    model = solve(inst.D, P, inst.lg, ag, l2a, air_cap=air_cap, L=L, 
        air_cost=air_cost, model=m(), upload_cost=inst.upload_cost, download_cost=inst.download_cost)
   obj_if(model)    
end




function show_table(res; show_loss=false)
    k = ["BD", "BGG", "BGC", "Gurobi", "CPLEX"]
    join([string(hk.n) * " & " * string(hk.m) * " & " * string(hk.s) * " & " * string(hk.p) * " & " * join([
    begin
        j = (mode=i, hk...)
        if j ∈ keys(res)
            l = res[j]
            if show_loss
                if l.val[2] == -1.0 # optimal
                    ""
                else
                    @sprintf " %.3e " l.val[1]
                end
            else
                if l.val[2] == 0.0 # optimal
                    @sprintf " %.1f " l.time
                elseif l.val[1] == -1.0 || l.val[2] >= 1# infeasible
                    ""
                else
                    
                    @sprintf " %.1f\\,\\si{\\percent} " (l.val[2] * 100)
                    
                end
            end
        else
            ""
        end
    end
    for i in k
    ], " & ") * " \\\\ \n"
    for hk in sort(unique([(n=k.n, m=k.m, s=k.s, p=k.p) for k in keys(res)]), by=x->(x.n, x.m, -x.s, x.p))])
end
println(show_table(deserialize("runs.jld"); show_loss=false))
println(show_table(deserialize("runs.jld"); show_loss=true))


 # warmup
@time let (inst, P, mg, ag, l2a) = load_inst(inst_list[inst_ind], 5, 3)
    fun_benders_gen(inst, 3, mg, ag, l2a, multiflow_gurobi())
    for im in [multiflow_gurobi, multiflow_cplex]
        fun_benders(inst, 3, mg, ag, l2a, multiflow_gurobi(), im) 
        fun_native(inst, 3, mg, ag, l2a, im)
    end
end

function main(inst_list, fn, res)
    for i in inst_list,
        s in [5, 10],
        p in [1, 5, 10]
        

        let kn = (mode="BD", n=i.n, m=i.m, s=s, p=p)
            if kn ∉ keys(res)

                try
                    t = @elapsed val = let (inst, P, mg, ag, l2a) = load_inst(i, 5, p)
                        fun_benders_gen(inst, P, mg, ag, l2a, multiflow_gurobi())
                    end
                    @show kn, t
                    res[kn] = (time=t, val=val)
                    serialize(fn, res)
                catch e
                    @show e
                end
            end
        end
        for (m, ms) in [(multiflow_gurobi, "G")],
            (im, ims) in zip([multiflow_gurobi, multiflow_cplex], ["G", "C"])
            
           
            
            let kn = (mode="B"*ms*ims, n=i.n, m=i.m, s=s, p=p)
                if kn ∉ keys(res)
                
                    try
                        inst = load_instance(i.fn)
                        t = @elapsed val = let (inst, P, mg, ag, l2a) = load_inst(i, 5, p)
                            fun_benders(inst, P, mg, ag, l2a, m(), im) 
                        end
                        @show kn, t
                        res[kn] = (time=t, val=val)
                        serialize(fn, res)
                    catch e
                        @show e
                    end
                end
            end
        end
        
        for (m, ms) in zip([multiflow_gurobi, multiflow_cplex], ["Gurobi", "CPLEX"])
            
            
            let kn = (mode=ms, n=i.n, m=i.m, s=s, p=p)
                if kn ∉ keys(res)

                    try
                        inst = load_instance(i.fn)
                        t = @elapsed val = let (inst, P, mg, ag, l2a) = load_inst(i, 5, p)
                                fun_native(inst, P, mg, ag, l2a, m)
                        end
                        @show kn, t
                        res[kn] = (time=t, val=val)
                        serialize(fn, res)
                    catch e
                        @show e

                    end
                end
            end
        end

        w1 = show_table(deepcopy(res); show_loss=false)
        
        open(fn *"-t.txt", "w") do fd
            println(fd, w1)
        end
        println(w1)
        w2 = show_table(deepcopy(res); show_loss=true)
        open(fn *"-l.txt", "w") do fd
            println(fd, w2)
        end
        println(w2)
    end
    res
end

main(inst_list[[1]], "dump", Dict())
main(inst_list, "runs.jld", Dict())


