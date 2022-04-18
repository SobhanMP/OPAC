using Infiltrator
using Revise
using Opaac
using Graphs
using Printf
using MetaGraphs
using JuMP
using Cthulhu
Revise.errors()
const air_cost=1_000
const air_cap=10_000
const L=4000
const inst_ind = 1
const inst_list = filter(map(readdir("instances")) do x
    n,m = parse.(Int, split(x[1:end-4], "-")[2:end])
    (n=n, m=m, fn="instances/"*x)
end) do x
    x.n <= 21
end

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
        air_cost=air_cost, model=multiflow_gurobi(), upload_cost=inst.upload_cost, download_cost=inst.download_cost)
   obj_if(model)    
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
            if l[1].val[2] == 0.0 # optimal
                @sprintf "%.1f" (l[1].time) # * "/" * string(l[1].val)
            elseif l[1].val[1] == -1.0 # infeasible
                @sprintf "-" (l[1].time)
            else
                @sprintf "%.1f/%.1f" (l[1].time, l[1].val[2] * 100)
            end
        else
            ""
        end
    end for c in cols], " & ")
    for e in d], " \\\\\n") * " \\\\"
end


function main()
    # warmup
    res = []
    for p in [1]
        our = []

        push!(our, [])
        let (inst, P, mg, ag, l2a) = load_inst(inst_list[inst_ind], 5, p)
            fun_benders_gen(inst, P, mg, ag, l2a, multiflow_gurobi())
        end
        
        for i in inst_list
            inst = load_instance(i.fn)
            t = @elapsed val = let (inst, P, mg, ag, l2a) = load_inst(i, 5, p)
                fun_benders_gen(inst, P, mg, ag, l2a, multiflow_gurobi())
            end
            push!(our[end], (i=i, time=t, val=val))
        end

        for m in [multiflow_gurobi],
            im in [multiflow_gurobi, multiflow_cplex]
            let (inst, P, mg, ag, l2a) = load_inst(inst_list[inst_ind], 5, p)
                fun_benders(inst, P, mg, ag, l2a, m(), im) 
            end
            push!(our, []) 
            for i in inst_list
                inst = load_instance(i.fn)
                t = @elapsed val = let (inst, P, mg, ag, l2a) = load_inst(i, 5, p)
                    fun_benders(inst, P, mg, ag, l2a, m(), im) 
                end
                push!(our[end], (i=i, time=t, val=val))
            end
        end
        solver = []
        for m in [multiflow_gurobi, multiflow_cplex]
            let (inst, P, mg, ag, l2a) = load_inst(inst_list[inst_ind], 5, p)
                fun_native(inst, P, mg, ag, l2a, m)
            end
            push!(solver, [])
            for i in inst_list
                inst = load_instance(i.fn)
                t = @elapsed val = let (inst, P, mg, ag, l2a) = load_inst(i, 5, p)
                        fun_native(inst, P, mg, ag, l2a, m)
                end

                push!(solver[end], (i=i, time=t, val=val))
            end
        end
        x = show_table(our..., solver...)
        println(x)
        open("col-table-$p.txt", "w") do fd
            println(fd, x)
        end
        push!(res, (our, solver))
    end
end
ret = main()
