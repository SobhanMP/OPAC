module Opaac

using Graphs
using LinearAlgebra
using StaticGraphs
using CPLEX
using SparseArrays

using MetaGraphs
using JuMP
using Gurobi
using DataStructures
using Luxor


const IN = inneighbors
const ON = outneighbors
const V = vertices

const E = edges

const GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    global GRB_ENV[] = Gurobi.Env()
end

struct Demand
    o :: Int
    d :: Int
    v :: Int
end

@inline besparse(adj::Union{Graph,MetaDiGraph}, x) = besparse(make_adj_matrix(adj), x)
@inline besparse(adj::SparseMatrixCSC, x::T) where {T<:Real} =
    besparse(adj, fill(x, length(adj.nzval)))
@inline besparse(adj::SparseMatrixCSC, x::Vector{T}) where {T} =
    SparseMatrixCSC{T,Int}(adj.m, adj.n, adj.colptr, adj.rowval, x)
@inline besparse(adj::SparseMatrixCSC, x::Matrix{T}) where {T} = [
    SparseMatrixCSC{T,Int}(adj.m, adj.n, adj.colptr, adj.rowval, x[:, i]) for
    i = 1:size(x, 2)
]
@inline besparse(adj::SparseMatrixCSC, x::Array{T,3}) where {T} = [
    SparseMatrixCSC{T,Int}(adj.m, adj.n, adj.colptr, adj.rowval, x[:, i, j]) for
    i = 1:size(x, 2), j = 1:size(x, 3)
]

@inline Base.getindex(g::AbstractMetaGraph, i::Int) = props(g, i)
@inline Base.getindex(g::AbstractMetaGraph, i::Int, j::Int) = props(g, i, j)
@inline Base.getindex(g::AbstractMetaGraph, e::Edge) = props(g, e)


@inline Base.getindex(g::AbstractMetaGraph, i::Int, s::Symbol) = get_prop(g, i, s)
@inline Base.getindex(g::AbstractMetaGraph, i::Int, j::Int, s::Symbol) = get_prop(g, i, j, s)
@inline Base.getindex(g::AbstractMetaGraph, e::Edge, s::Symbol) = get_prop(g, e, s)

@inline Base.setindex!(g::AbstractMetaGraph, v, i::Int, s::Symbol) = set_prop!(g, i, s, v)
@inline Base.setindex!(g::AbstractMetaGraph, v, i::Int, j::Int, s::Symbol) = set_prop!(g, i, j, s, v)
@inline Base.setindex!(g::AbstractMetaGraph, v, e::Edge, s::Symbol) = set_prop!(g, e, s, v)

@inline to_ind(x, y, i, j) = begin
    @assert i >= 1 && i <= x
    @assert j >= 1 && j <= y
    (i - 1) * y + j 
end


struct Instance{I, V}
    lg::MetaDiGraph{I, V}
    D::Vector{Demand}
    upload_cost::Int
    download_cost::Int
    air_cap::Int
end
Base.:(==)(x::Instance{I, V}, y::Instance{I, V}) where {I, V} = x.lg == y.lg && 
    x.D == y.D && 
    x.upload_cost == y.upload_cost && 
    x.download_cost == y.download_cost && 
    x.air_cap == y.air_cap

function write_instance(fn, inst::Instance)
    
    D, lg = inst.D, inst.lg

    open(fn, "w") do fd
        println(fd, length(D), " ", nv(lg), " ", ne(lg))
        println(fd, inst.upload_cost, " ", inst.download_cost, " ", inst.air_cap)
        
        for v in V(lg)
            println(fd, v, " ", lg[v, :x], " ", lg[v, :y])
        end
        for e in E(lg)
            println(fd, src(e), " ", dst(e), " ", lg[e, :length], " ", lg[e, :cap], " ",  lg[e, :cost])
        end
        for d in D
            println(fd, d.o, " ", d.d, " ", d.v)
        end
    end
    nothing
end

readL(fd, type=Int) = parse.(type, split(readline(fd)))

function load_instance(fn)
    open(fn, "r") do fd
        nD, nvlg, nelg = readL(fd) 
        upload_cost, download_cost, air_cap = readL(fd)

        lg = MetaDiGraph(nvlg)
        for _ in 1:nvlg
            v, x, y = readL(fd)
            lg[v, :x] = x
            lg[v, :y] = y
        end
        for _ in 1:nelg
            src, dst, length, cap, cost = readL(fd, [Int, Int, Float64, Int, Int])
            add_edge!(lg, src, dst)
            lg[src, dst, :length] = length
            lg[src, dst, :cap] = cap
            lg[src, dst, :cost] = cost
        end
        
        D = Vector{Demand}()
        sizehint!(D, nD)
        for _ in 1:nD
            o, d, v = readL(fd)
            push!(D, Demand(o, d, v))
        end
        Instance(lg, D, upload_cost, download_cost, air_cap)
    end
end

function add_land_edge(graph, a, b, cap_range, cost_range) 
    add_edge!(graph, a, b)
    set_prop!(graph, a, b, :cap, rand(cap_range))
    set_prop!(graph, a, b, :cost, rand(cost_range))
    set_prop!(graph, a, b, :length,
        norm(
            (graph[a][:x] - graph[b][:x], 
                graph[a][:y] - graph[b][:y])))
    nothing
end

function add_land_edge(graph, a, b) 
    add_edge!(graph, a, b)
    set_prop!(graph, a, b, :length,
        norm(
            (graph[a][:x] - graph[b][:x], 
                graph[a][:y] - graph[b][:y])))
    nothing
end


struct LazyDijkstraState{T<:Real,U<:Integer}
    src::U
    parent::Vector{Vector{U}}
    distance::Vector{T}
    visited::Vector{Bool}
    q::PriorityQueue{U,T, Base.Order.ForwardOrdering}
end

function graph_sparse_weight(g; key=:time, n=nv(g))
    w = spzeros(n, n)
    for i in vertices(g), j in inneighbors(g, i)
        w[j, i] = get_prop(g, j, i, key)
    end
    w
end


function gen_instance(lg, max_instance_comodity; retry0=10, vol_range, feas_only=true, 
    state = @time init_solve_land(max_instance_comodity, lg; feas_only=feas_only),
    fn = nothing)
    @show lg
    
    demands = Dict{Tuple{Int, Int}, Tuple{Demand, Int}}()
    retry = retry0
    iter = 0

    while retry > 0 && length(demands) < max_instance_comodity
        iter += 1
        o, d = rand(1:nv(lg)), rand(1:nv(lg) - 1)
        if d >= o
            d += 1
        end
        v = rand(vol_range)
        
        if (o, d) ∈ keys(demands)
            old_dem, k = demands[(o, d)]
            dem = Demand(o, d, v + old_dem.v)
        else
            dem = Demand(o, d, v)
            k = length(demands) + 1
        end
        
        set_comodity_demand(state, o, d, dem.v, k)

        @time optimize!(state.model)
        fn !== nothing && GRBwrite(backend(model), "$(fn)-$(iter).mps")
        
        if termination_status(state.model) == MOI.OPTIMAL    
            demands[(o, d)] = (dem, k)
            @show length(demands)
        else
            retry -= 1
            @show retry
            if (o, d) ∈ keys(demands)
                set_comodity_demand(state, o, d, old_dem.v, k)
            else
                @assert k == length(demands) + 1
                set_comodity_demand(state, o, d, 0, k)
            end
        end
        
    end
    
    optimize!(state.model)
    
    
    [i for (i, _) in values(demands)], model
end

function new_state(g, src::U, ::AbstractMatrix{T}) where {T,U}
    state = LazyDijkstraState{T,U}(
        src,
        fill([], nv(g)),
        fill(typemax(T), nv(g)),
        zeros(Bool, nv(g)),
        PriorityQueue{U,T}()
    )
    state.distance[src] = 0
    state.q[src] = zero(T)
    state
end

function lazy_dijkstra(
    g::AbstractGraph,
    src::I,
    dst::I;
    w::AbstractMatrix{T} = weights(g),
    state::LazyDijkstraState{T, I} = new_state(g, src, w),
    lazy::Bool=true, 
    filter::F=(_, _) -> true
) where {T <: Real, I <: Integer, F}
    @assert state.src == src
    visited = state.visited
    q = state.q
    parent = state.parent
    distance = state.distance
    
    if visited[dst] && lazy
        return distance[dst], state
    end

    while !isempty(q)
        u = dequeue!(q)
        d = distance[u]

        visited[u] = true

        for v in outneighbors(g, u)
            if !filter(u, v)
                continue
            end

            alt = d + w[u, v]

            if !visited[v] && distance[v] > alt
                # if visited[v]
                #     error("negative cycle detected")
                # end
                parent[v] = [u]
                distance[v] = alt
                q[v] = alt
            elseif distance[v] == alt
                if u ∉ parent[v]
                    push!(parent[v], u)
                end
            end
        end

        if lazy && u == dst
            return d, state
        end
    end
    return distance[dst], state
end
function ford_fulkerson(lg_adj, lg_cap, lg, ag_adj, ag, l, o, d, v, w, P)
    paths = Vector{Int}[]
    flows = Int[]
    lx = SparseMatrixCSC{Float64, Int}(
        lg_adj.m, lg_adj.n,
        lg_adj.colptr, lg_adj.rowval,
        zeros(Float64, length(lg_adj.nzval))
    )
    ax = [
        SparseMatrixCSC{Float64, Int}(
        ag_adj.m, ag_adj.n,
        ag_adj.colptr, ag_adj.rowval,
        zeros(Float64, length(ag_adj.nzval)))
        for _ in 1:P
    ]
    cap(i, j) = if max(i, j) ≤ nv(lg)
        lg_cap[i, j] - lx[i, j]  
    elseif min(i, j) ≤ nv(lg)
        typemax(Float64)
    else
        idp, irp, jrp = split_air(ag, lg, i, j)
        air_cap * l[idp][irp, jrp] - ax[idp][irp, jrp]
    end
    djf(i, j) = cap(i, j) > 0
    cost = 0
    while v > 1e-10
        c, state = lazy_dijkstra(mg, o, d; w=w, filter=djf)
        p = dijkstra_to_path(state, o, d)
        Δv = min(minimum(cap(i, j) for (i, j) in make_pair(p)), v)
        # @show Δv, c?
        cost += Δv * c
        v = max(0, v - Δv)
        
        for (i, j) in make_pair(p)
            if max(i, j) ≤ nv(lg)
                lx[i, j] += Δv
            elseif min(i, j) ≤ nv(lg)
                nothing
            else
                idp, irp, jrp = split_air(ag, lg, i, j)
                ax[idp][irp, jrp] += Δv
            end
        end
        push!(paths, p)
        push!(flows, Δv)
    end
    paths, (flow=flows, lx=lx, ax=ax, cost=cost)
end

function dijkstra_to_path(
    pred::AbstractVector{V}, 
    s::U, 
    t::U, 
    ignore_paths::AbstractVector{V}=Vector{U}[], 
    visited::Set{U}=Set{U}(t)) where {U, V <:AbstractVector{U},}


    
    # @show s, t
    c = t
    res = [t]
    while c != s
        cc = pred[c][1]
        if length(cc) == 1
            c = cc[1]
            
            if c ∉ visited 
                push!(res, c)
                push!(visited, c)
            end
            
        else
            reverse!(res)
            for i in cc
                p = dijkstra_to_path_(pred, s, i, [@view i[1:end-length(res)]
                    for i in ingore_paths if (@view i[end-length(res) + 1:end]) == res], deepcopy(visited))
                if p !== nothing
                    
                    append!(p, res)
                    return p
                end
            end
            
            return nothing
        end
    end
    reverse!(res)
    # @show res
    if res ∈ ignore_paths
        return nothing
    else
        return res
    end
end  


@inline dijkstra_to_path(state::LazyDijkstraState, 
    s::U, t, 
    ignore_paths=Vector{U}[], 
    visited=Set{U}(t)) where U = dijkstra_to_path(
        state.parent, s, t, ignore_paths, visited)



function combgraph(lg, ag, a2l, P)
    mg =  MetaDiGraph(P * nv(ag) + nv(lg))
    for e in E(lg)
        s = src(e)
        d = dst(e)
        
        add_edge!(mg, s, d)
        
        mg[s, d, :cost] = lg[e][:cost]
        mg[s, d, :cap] = lg[e][:cap]
        mg[s, d, :air] = false
    end

    for p in 1:P,
        e in E(ag)
        @show p    
        s = src(e) + nv(lg) + (p - 1) * nv(ag)
        d = dst(e) + nv(lg) + (p - 1) * nv(ag)
        
        add_edge!(mg, s, d)
    end

    for p in 1:P,
        (s, d) in a2l
        @show p
        s += nv(lg) + (p - 1) * nv(ag)
        
        add_edge!(mg, s, d)
        
        mg[s, d, :cost] = download_cost
        mg[s, d, :cap] = -1
        
        add_edge!(mg, d, s)
        
        mg[d, s, :cost] = upload_cost
        mg[d, s, :cap] = -1
    end

    mg
end

function mk_land_graph(x, y; cap_range, cost_range)
    graph = MetaDiGraph(x * y)
    for i in 1:x,
        j in 1:y
        
        ind = to_ind(x, y, i , j)
        set_prop!(graph, ind, :x, i * 10)
        set_prop!(graph, ind, :y, j * 10)
    end
    @assert all(has_prop(graph, i, :x) for i in 1:(x * y))
    lg = deepcopy(graph)
    for i in 1:x,
        j in 1:y
        ind = to_ind(x, y, i , j)
        j > 1 && add_land_edge(lg, ind, to_ind(x, y, i, j - 1), cap_range, cost_range)
        j < y && add_land_edge(lg, ind, to_ind(x, y, i, j + 1), cap_range, cost_range)
        i > 1 && add_land_edge(lg, ind, to_ind(x, y, i - 1, j), cap_range, cost_range)
        i < x && add_land_edge(lg, ind, to_ind(x, y, i + 1, j), cap_range, cost_range)
    end
    lg
end
function mk_air_graph(x, y, skip)
    ag = deepcopy(graph)
    for i in 1:skip:x,
        j in 1:skip:y
        ind = to_ind(x, y, i , j)

        j > skip && add_land_edge(ag, ind, to_ind(x, y, i, j - skip))
        j <= y - skip && add_land_edge(ag, ind, to_ind(x, y, i, j + skip))
        i > skip && add_land_edge(ag, ind, to_ind(x, y, i - skip, j))
        i <= x - skip && add_land_edge(ag, ind, to_ind(x, y, i + skip, j))
    end
    ag, vp = induced_subgraph(ag, connected_components(ag)[1])
    a2l = Dict((i, vp[i]) for i in eachindex(vp))
    
    l2a = Dict(map(reverse, collect(a2l)));

   
    ag, a2l, l2a
end


function sum_affine(x, acc = AffExpr(0.0))
    for i in x
        add_to_expression!(acc, i)
    end
    acc
end
multiflow_gurobi() = direct_model(
    optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV[]), 
        "OutputFlag" => 0, 
        "Method" => 1,
        "NormAdjust" => 2,
        "Presolve" => 0
        ))

function init_solve_land(D::Vector{Demand},
    lg; 
    feas_only=false, 
    w_cap::SparseMatrixCSC=graph_sparse_weight(lg, key=:cap),
    w_cost::SparseMatrixCSC=graph_sparse_weight(lg, key=:cost),
    adj=make_adj_matrix(lg),
    model = multiflow_gurobi())
    K = length(D)

    x_ = @variable(model, [1:nnz(adj), 1:K], lower_bound = 0)
    x = besparse(adj, x_)
    
    cape = [sum_affine(x[k].nzval[i] for k in 1:K) for i in 1:nnz(adj)]

    
    deme = [sum_affine(
                (-x[k][i, j] for j in ON(lg, i)),
                sum_affine(x[k][j, i] for j in IN(lg, i)))
                for k in 1:K, i=1:nv(lg)
    ]

    @constraints(model, begin 
        cap_[i=1:nnz(adj)], 
        cape[i] ≤ w_cap.nzval[i] # land cap 
        dem[k=1:K, i=1:nv(lg)], deme[k, i] == (D[k].o == i ? -D[k].v : (D[k].d == i ? D[k].v : 0))
    end)
    
    if !feas_only
        @objective(model, Min,  sum_affine(x_[i, k] * w_cost.nzval[i] 
            for i in 1:nnz(w_cost) 
            for k in 1:K))
    end
    optimize!(model)
    (model=model, x=x, dem=dem)
end
        
function init_solve_land(K,
    lg; 
    feas_only=false, 
    w_cap::SparseMatrixCSC=graph_sparse_weight(lg, key=:cap),
    w_cost::SparseMatrixCSC=graph_sparse_weight(lg, key=:cost),
    adj=make_adj_matrix(lg),
    model = multiflow_gurobi())
    
    
    x_ = @variable(model, [1:nnz(adj), 1:K], lower_bound = 0)
    x = besparse(adj, x_)
    
    cape = [sum_affine(x[k].nzval[i] for k in 1:K) for i in 1:nnz(adj)]

    
    deme = [sum_affine(
                (-x[k][i, j] for j in ON(lg, i)),
                sum_affine(x[k][j, i] for j in IN(lg, i)))
                for k in 1:K, i=1:nv(lg)
    ]

    @constraints(model, begin 
        cap_[i=1:nnz(adj)], 
        cape[i] ≤ w_cap.nzval[i] # land cap 
        dem[k=1:K, i=1:nv(lg)], deme[k, i] == 0
    end)
    
    if !feas_only
        @objective(model, Min,  sum_affine(x_[i, k] * w_cost.nzval[i] 
            for i in 1:nnz(w_cost) 
            for k in 1:K))
    end

    (model=model, x=x, dem=dem)
end


function set_comodity_demand(state, o, d, v, k)
    set_normalized_rhs(state.dem[k, o], -v)
    set_normalized_rhs(state.dem[k, d], v)
end

function solve(D::Vector{Demand}, P::Int, lg, ag, l2a; air_cost=1_000, air_cap=1000, L=1_000_000)
    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV[]), 
        "TimeLimit" => 600.0))
    K = length(D)
    
    @variables(model, begin
        x[1:K, i=V(lg), j=ON(lg, i)] >= 0 # land flow
        y[1:P, 1:K, i=V(ag), j=ON(ag, i)] >= 0 # air flow
        l[1:P, i=V(ag), j=ON(ag, i)], Bin # pigeon paths
        sp[1:P, V(ag)], Bin # start node of the pigeon path, the cycles of each pigeon need to contain sp
        a[1:P], Bin # pigeon activation
        u[1:P, 1:K, V(ag)] >= 0 # upload data from job k to pigeon p at air node v
        d[1:P, 1:K, V(ag)] >= 0 # download
        U[1:P, V(ag)] # MTZ like potential variable
    end)

    
    @constraints(model, begin
            [p=1:P-1], a[p] >= a[p+1] # give solutions order
            
            [i=V(lg), j=ON(lg, i)], sum(x[k, i, j] for k in 1:K) ≤ lg[i, j][:cap] # land cap

            [p=1:P, i=V(ag), j=ON(ag, i)], l[p, i, j] <= a[p] # cativation  
            [p=1:P, i=V(ag), j=ON(ag, i)], sum(y[p, k, i, j] for k in 1:K) ≤ l[p, i, j] * air_cap # pigeon data cap
            [p=1:P], sum(l[p, i, j] * ag[i, j][:length] for i in V(ag) for j in ON(ag, i)) ≤ L # max travel distance

            [p=1:P, i=V(ag)], 
                sum(l[p, i, j] for j in ON(ag, i))  == 
                sum(l[p, j, i] for j in IN(ag, i))  # the pigeon paths should form loops
            [p=1:P, i=V(ag)], sum(l[p, i, j] for j in ON(ag, i)) <= 1 # loops should be cycles
            [p=1:P, i=V(ag), j=ON(ag, i)], U[p, i] - U[p, j] + nv(ag) * (l[p, i, j] - 1 - sp[i] - sp[j]) ≤ -1 # mtz potential with variable sp

            [p=1:P], sum(sp[p, i] for i in V(ag)) <= 1 # there is one starting point
 
            [k=1:K, i=V(lg)], 
                    source_sink(D[k], i) + 
                    uplink(P, l2a, d, u, k, i) +
                    sum(x[k, j, i] for j in IN(lg, i)) - 
                    sum(x[k, i, j] for j in ON(lg, i)) == 0
            [p=1:P, k=1:K, i=V(ag)],
                u[p, k, i] -
                d[p, k, i] +
                sum(y[p, k, j, i] for j in IN(ag, i)) - 
                sum(y[p, k, i, j] for j in ON(ag, i)) == 0
        end)

    @objective(model, Min,
        sum(lg[i, j][:cost] * x[k, i, j] for i in V(lg) for j in ON(lg, i) for k in 1:K) +
        sum(a) * air_cost + 
        sum(u)
    );
    optimize!(model)
    model
end

rep(((x_min, x_max), (y_min, y_max), (im_x, im_y), (mar_x, mar_y)), x, y) = Point(
    (x - x_min) / (x_max - x_min) * im_x + mar_x,
    (y - y_min) / (y_max - y_min) * im_y + mar_y
)

source_sink(d, i) = ((i == d.o) - (i == d.d)) * d.v

function mkstate(g)
    x = extrema(V(g)) do v
        g[v][:x]
    end
    y = extrema(V(g)) do v
        g[v][:y]
    end
    (x, y, (1000, 1000), (20, 20)), (1040, 1040)
end
function draw_plot(g; vcol=(g, i)->(0, 0, 0, 0.5), ecol=(g, i, j)->(0, 0, 0, 0.5), fil=(g, x, y) -> true, ps=2)
    state, img_s = mkstate(g)
    
    drawing = Drawing(img_s..., :png)
    background("white")
    for i in V(g)
        setcolor(vcol(g, i)...)
        circle(rep(state, g[i][:x], g[i][:y]), ps, :fill)
    end
    for e in E(g)
        s, d = src(e), dst(e)
        !fil(g, s, d) && continue
        Δ = Point(g[d, :y] - g[s, :y], g[s, :x] - g[d, :x])
        Δ /= norm(Δ)
        Δ *= ps
        setcolor(ecol(g, s, d)...)
        line(
            rep(state, g[s, :x], g[s, :y]) - Δ,
            rep(state, g[d, :x], g[d, :y]) - Δ,
        :stroke)
    end
    finish()
    drawing
end

function make_adj_matrix(ag)
     adj_matrix = spzeros(Bool, nv(ag), nv(ag))
    for i in V(ag),
        j in IN(ag, i)
        adj_matrix[j, i] = true
    end
    adj_matrix
end
uplink(P, l2a, d, u, k, v) = if v ∈ keys(l2a)
    let i = l2a[v]
        sum(d[p, k, i] - u[p, k, i] for p in 1:P)
    end
else
    0
end

function subproblem(D::Vector{Demand}, P, lg, ag, l2a, air_cap, l; 
        ag_adj=make_adj_matrix(ag), 
        lg_adj=make_adj_matrix(lg),
        ed=[(j, i) for i=V(ag) for j=IN(ag, i)]
    )
    # model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), 
    #         "OutputFlag" => 0,
    #         "Threads" => 8,
    #         "Method" => 2
    #         ))
    model = direct_model(CPLEX.Optimizer())
    set_silent(model)
    K = length(D)
    @variables(model, begin
        x_[1:ne(lg), 1:K] >= 0 # land flow
        y_[1:ne(ag), 1:P, 1:K] >= 0 # air flow
        u[1:P, 1:K, V(ag)] >= 0 # upload data from job k to pigeon p at air node v
        d[1:P, 1:K, V(ag)] >= 0 # download
        
    end)
    y = besparse(ag_adj, y_)
    x = besparse(lg_adj, x_)
    
    @constraints(model, begin
            [i=V(lg), j=ON(lg, i)], sum(x[k][i, j] for k in 1:K) ≤ lg[i, j][:cap] # land cap
            [k=1:K, i=V(lg)], 
                source_sink(D[k], i) + 
                uplink(P, l2a, d, u, k, i) +
                sum(x[k][j, i] for j in IN(lg, i)) - 
                sum(x[k][i, j] for j in ON(lg, i)) == 0
            [p=1:P, k=1:K, i=V(ag)],
                u[p, k, i] -
                d[p, k, i] +
                sum(y[p, k][j, i] for j in IN(ag, i)) - 
                sum(y[p, k][i, j] for j in ON(ag, i)) == 0
            
            con_[i=1:length(ed), p=1:P], 
                sum(y[p, k][ed[i][1], ed[i][2]] for k in 1:K) ≤ 
                    l[p][ed[i][1], ed[i][2]] * air_cap # pigeon data cap
        end)
     
        
    @objective(model, Min,
        sum(lg[i, j][:cost] * x[k][i, j] for i in V(lg) for j in ON(lg, i) for k in 1:K) +
        
        sum(u)
    );      
    
    optimize!(model)
    @assert termination_status(model) ∈ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    cond = dual.(con_)
    return (
        θ = objective_value(model),
        π = besparse(ad_adj, cond)
    )
end

function master_problem(D::Vector{Demand}, P::Int, lg, ag, l2a; air_cost=1_000, air_cap=1000, L=1_000_000)
    adj_matrix = make_adj_matrix(ag)
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV[]), "Threads" => 8))
    
    # model = direct_model(CPLEX.Optimizer())
    K = length(D)
    
    @variables(model, begin
        
        l_[1:ne(ag), 1:P], Bin # pigeon paths
        sp[1:P, 1:nv(ag)], Bin # start node of the pigeon path, the cycles of each pigeon need to contain sp
        a[1:P], Bin # pigeon activation
        U[1:P, 1:nv(ag)] # MTZ like potential variable
        θ >= 0
    end)
    
    l = besparse(adj_matrix, l_)
    
    @constraints(model, begin
            # [p=1:P-1], a[p] >= a[p+1] # give solutions order
            [p=1:P, i=V(ag), j=ON(ag, i)], l[p][i, j] <= a[p] # cativation  
            
            [p=1:P], sum(l[p][i, j] * ag[i, j][:length] for i in V(ag) for j in ON(ag, i)) ≤ L # max travel distance

            [p=1:P, i=V(ag)], 
                sum(l[p][i, j] for j in ON(ag, i))  == 
                sum(l[p][j, i] for j in IN(ag, i))  # the pigeon paths should form loops
                    
            [p=1:P, i=V(ag)], sum(l[p][i, j] for j in ON(ag, i)) <= 1 # loops should be cycles
            [p=1:P, i=V(ag), j=ON(ag, i)], U[p, i] - U[p, j] + 
            nv(ag) * (l[p][i, j] - 1 - sp[i] - sp[j]) ≤ -1 # mtz potential with variable sp

            [p=1:P], sum(sp[p, i] for i in V(ag)) <= 1 # there is one starting point
            
        end)
    l_i = besparse(adj_matrix, 0)[SparseMatrixCSC{Float64, Int}(
            nv(ag), nv(ag), 
            adj_matrix.colptr, adj_matrix.rowval, 
            zeros(ne(ag)))
        for p=1:P]
    l_k = zeros(ne(ag), P)
    lg_adj = make_adj_matrix(lg)
    function my_callback(cb_data)
        η = 0.5
        begin
            l_k .= callback_value.(cb_data, l_)    :: Matrix{Float64}
            for p in 1:P
                @assert length(l_i[p].nzval) == length(l_k[:, p])
                l_i[p].nzval .= (1 - η) .* l_i[p].nzval .+ η .* l_k[:, p]
            end
            θ_k = callback_value(cb_data, θ) ::Float64
            ret = subproblem(D, P, lg, ag, l2a, air_cap, l_i; ag_adj=adj_matrix, lg_adj=lg_adj)
            
            # @show θ_k, ret.θ, extrema(l_k), dot(ret.π, l_k)
            flush(stdout)
            cut = @build_constraint(θ >= ret.θ +
                sum(
                    air_cap * ret.π[p][i, j] * (l[p][i, j] - l_i[p][i, j])
                    for p=1:P for i=V(ag) for j=ON(ag, i)))
            MOI.submit(model, MOI.LazyConstraint(cb_data), cut)
        end
        return
    end
    MOI.set(model, MOI.LazyConstraintCallback(), my_callback)
    @objective(model, Min, sum(a) * air_cost + θ)
    optimize!(model)
    model
end






function x_to_path(o, d, g, x)
    ps = [Int[]]
    if d == o
        return ps
    end
    
    while o != d
        if 1 == sum(x[o, i] > 0 ? 1 : 0 for i in outneighbors(g, o))
            for i in ON(g, o)
                if x[o, i] > 0
                    push!(ps[1], o)
                    o = i
                    break
                end
            end
        else
            push!.(ps, o)
            qs = Vector{Int}[]
            for i in ON(g, o) 
                if x[o, i] > 0
                    for p in x_to_path(i, d, g, x),
                        q in ps
                        q = deepcopy(q)
                        append!(q, p)
                        push!(qs, q)
                    end
                end
            end
                                 
            return qs
        end
    end
    for i in eachindex(ps)
        push!(ps[i], d)
    end
    ps     
end
x_to_path(d::AbstractArray{Demand}, g, x, k) = x_to_path(d[k].o, d[k].d, g, x[k])
x_to_path(d::Demand, g, x, k) = x_to_path(d.o, d.d, g, x)



@inline make_pair(p) = zip(p, @view p[2:end])
function path_cost(lg::AbstractMatrix{T}, ag, p) where T
    cost = zero(T)
    for (i, j) in make_pair(p)
        if max(i, j) ≤ size(lg, 1)
            cost += lg[i, j]
        elseif i ≤ size(lg, 1)
            cost += upload_cost
        elseif j ≤ size(lg, 1)
            cost += download_cost
        end
    end
    cost 
end
function path_cost(w::AbstractMatrix{T}, p) where T
    cost = zero(T)
    for (i, j) in make_pair(p)
        cost += w[i, j]
    end
    cost 
end
struct RWM <: AbstractMatrix{Float64}
    c::SparseMatrixCSC{Float64, Int}
    wl::SparseMatrixCSC{Float64, Int}
    wa::Vector{SparseMatrixCSC{Float64, Int}}
    vl::Int
    va::Int
end
Base.size(w::RWM) = let l = w.va * length(w.wa) + w.vl
    (l, l)
end
@inline function Base.getindex(r::RWM, i::Int, j::Int)
    if max(i, j) ≤ r.vl
        r.c[i, j] + r.wl[i, j]
    elseif i ≤ r.vl
        upload_cost
    elseif j ≤ r.vl
        download_cost
    else
        idp, irp, jrp = split_air(ag, lg, i, j)
        
        r.wa[idp][irp, jrp]
    end

end
struct RC <: AbstractMatrix{Float64}
    c::SparseMatrixCSC{Float64, Int}
    n::Int
end
Base.size(w::RC) = (w.n, w.n)
@inline function Base.getindex(r::RC, i::Int, j::Int)
    if max(i, j) ≤ r.n
        r.c[i, j]
    elseif i ≤ r.n
        1
    else
        0
    
    end

end

function split_air(ag, lg, i, j)
    idp, irp = divrem(i - nv(lg) - 1, nv(ag)) .+ (1, 1)
    jdp, jrp = divrem(j - nv(lg) - 1, nv(ag)) .+ (1, 1)
    @assert idp == jdp
    idp, irp, jrp
end
function init_colgen(D, mg, lg, ag, l, ub)
    P = length(l)
    model = direct_model(Gurobi.Optimizer())
    lg_cap = graph_sparse_weight(lg; key=:cap)
    lg_adj = make_adj_matrix(lg)
    lg_cost = graph_sparse_weight(lg; key=:cost)

    @variable(model, phase1[i=1:length(D)] >= 0)

    
    
    @constraint(model, lg_w_[i=1:nnz(lg_cap)], 0 <= lg_cap.nzval[i])
    lg_w = besparse(lg_adj, lg_w_)
    
    @constraint(model, ag_w_[i=1:nnz(ag_adj), p=1:P], 0 <= air_cap * l[p].nzval[i])
    ag_w = besparse(ag_adj, ag_w_)
    wc = RC(lg_cost, nv(lg))
    ps = [ford_fulkerson(lg_adj, lg_cap, lg, ag_adj, ag, l, d.o, d.d, d.v, wc, P)[1] for d in D] 
    
    fs = [@variable(model, [i=1:length(pp)], lower_bound=0) for pp in ps]

    @constraint(model, dem[k=1:length(D)], sum(fs[k]) + phase1[k] == D[k].v)
    
    for (pp, ff) in zip(ps, fs),
        (p, f) in zip(pp, ff),
        (i, j) in make_pair(p)
        
        if max(i, j) ≤ nv(lg)
            set_coefficient(lg_w[i, j], f, 1)
        elseif min(i, j) > nv(lg)
            idp, irp, jrp = split_air(ag, lg, i, j)
            set_coefficient(ag_w[idp][irp, jrp], f, 1)
        end
        
    end
    if ub === nothing
        @objective(model, Min, sum_affine(phase1))
        
    else
        @objective(model, Min, 
                    sum_affine(path_cost(lg_cost, ag, p) * f 
                    for (pp, ff) in zip(ps, fs) 
                        for (p, f) in zip(pp, ff)) + (ub + 1) * sum_affine(phase1))
    end
    optimize!(model)

    @assert termination_status(model) == MOI.OPTIMAL
    d_lg_w = besparse(lg_adj, abs.(dual.(lg_w_)))::SparseMatrixCSC{Float64, Int}
    d_ag_w = besparse(ag_adj, abs.(dual.(ag_w_)))::Vector{SparseMatrixCSC{Float64, Int}}
    π = dual.(dem)
    w = RWM(deepcopy(lg_cost), d_lg_w, d_ag_w, nv(lg), nv(ag))
    if ub === nothing
        w.c.nzval .= zero(eltype(w.c.nzval))
    end

    group = Dict{Int, Vector{Tuple{Int, Demand}}}()
    for (i, d) in enumerate(D)
        e = get!(() -> Tuple{Int, Demand}[], group, d.o)
        push!(e, (i, d))
    end

    (model=model, dem=dem, phase1=phase1, w=w, wc=wc, ub=ub, π=π, f=fs, p=ps, P=P, l=l, 
    mg=mg, lg=(
        g=lg,
        cost = lg_cost,
        adj = lg_adj,
        cap = lg_cap,
        w=lg_w,
        w_=lg_w_,
        d = d_lg_w,
        nv=nv(lg)
    ), ag=(
        g=ag,
        adj = make_adj_matrix(ag),
        w=ag_w,
        d=d_ag_w,
        nv=nv(ag),
        w_=[ag_w_[:, i] for i in 1:P]
    ), air_cap=air_cap, group=collect(group))
end

function lag_obj(state)
    a = sum(value(f) * path_cost(state.wc, p) 
        for (pp, ff) in zip(state.p, state.f) 
        for (p, f) in zip(pp, ff)) ::Float64
    b = sum(state.lg.d[src(e), dst(e)] * state.lg.cap[src(e), dst(e)] 
        for e in E(state.lg.g))::Float64
    c = state.air_cap * sum(
        state.l[p][src(e), dst(e)] * state.ag.d[p][src(e), src(e)]
        for p in 1:state.P for e in E(state.ag.g))::Float64
    a - b - c
end
function load_dual(state)
    optimize!(state.model)
    @assert termination_status(state.model) == MOI.OPTIMAL
    
    @show objective_value(state.model)
    state.lg.d.nzval .= abs.(dual.(state.lg.w_))
    for p in 1:state.P
        state.ag.d[p].nzval .= abs.(dual.(state.ag.w_[p]))
    end
    state.π .= dual.(state.dem)
    nothing
end
function register_path(state, dem, p, f)
    set_coefficient(dem, f, 1)
    for (i, j) in make_pair(p)
        if max(i, j) ≤ state.lg.nv
            set_coefficient(state.lg.w[i, j], f, 1)
        elseif min(i, j) > state.lg.nv
            idp, irp, jrp = split_air(state.ag, state.lg, i, j)
            set_coefficient(state.ag.w[idp][irp, jrp], f, 1)
        end
    end
    nothing
end
function set_phase2_obj(state)
    @objective(state.model, Min, 
        sum_affine(path_cost(state.lg.cost, 0, p) * f 
        for (pp, ff) in zip(state.p, state.f) 
            for (p, f) in zip(pp, ff)))
    nothing
end
function add_path(state)
    new_path = 0
    for (orig, es) in state.group
        state_w = new_state(mg, orig, w)
        for (i, d) in es
            # @assert d.o == orig
            e, _ = lazy_dijkstra(mg, d.o, d.d; w=w, state=state_w, 
            filter=(i, j) -> max(i, j) <= nv(lg) || min(i, j) <= nv(lg) || let (idp, irp, jrp) = split_air(ag, lg, i, j)
            l[idp][irp, jrp] > 0
            end)

            p = dijkstra_to_path(state_w, d.o, d.d, ps[i])
            
            if p !== nothing
                # @assert p ∉ ps[i]
                # @assert p[1] == D[i].o
                # @assert p[end] == D[i].d
                
                c = path_cost(state.lg.cost, 0, p)
                new_path += 1

                f = @variable(model, lower_bound=0)
                push!(ps[i], p)
                push!(fs[i], f)

                if phase == 2
                    set_objective_coefficient(model, f, c)
                end
                register_path(state, state.dem[i], p, f)
            end
        end
    end
    new_path
end

function colgen(state)
    phase = 1
    model, phase1 = state.model, state.phase1

    for _ in 1:1000000
        if phase == 1 && maximum(value, phase1)  < 1e-10
            if state.ub === nothing
                state.w.c.nzval .= state.lg.cost.nzval
            end
            set_phase2_obj(state)        
            fix.(phase1, 0; force=true)
            load_dual(state)
            phase = 2
        elseif phase == 2 # termination critirion
            lag_obj(state)  == objective_value(model) && break
        end

        new_path = add_path(state)
        if new_path == 0            
            # @assert state.w.wl == state.lg.d
            # @assert state.w.c == state.lg.cost

            (phase != 1 || maximum(value, phase1) > 1) && break
        else
            load_dual(state)
        end
    end
    # @assert phase == 2
    nothing
end
export colgen, init_colgen
export master_problem, subproblem
export init_solve_land, set_comodity_demand
export solve, make_adj_matrix
export mk_land_graph, combgraph, mk_air_graph, gen_instance, Demand
export write_instance, load_instance, Instance
end