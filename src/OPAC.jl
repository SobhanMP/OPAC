module OPAC
using Infiltrator, Cthulhu, StatsBase, Graphs, LinearAlgebra
using StaticGraphs, CPLEX, SparseArrays, Distributions, MetaGraphs
using JuMP, Gurobi, DataStructures, Luxor

const IN = inneighbors
const ON = outneighbors
const V = vertices

const E = edges

const GRB_ENV = Ref{Gurobi.Env}()

"trick to initalize GRB_ENV at runtime"
function __init__()
    global GRB_ENV[] = Gurobi.Env()
end

"direct model with default gurobi configuration"
multiflow_gurobi(time=600.0) = direct_model(
    optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV[]), 
        "OutputFlag" => 0, 
        "Threads" => 8,
        "Method" => 1,
        "TimeLimit" => time))

"direct model with default cplex configuration"
multiflow_cplex(time=600.0) = direct_model(
    optimizer_with_attributes(
        CPLEX.Optimizer, 
        "CPXPARAM_TimeLimit" => time, 
        "CPX_PARAM_THREADS" => 8,
        "CPXPARAM_ScreenOutput" => 0))

"standard holder for demands"
struct Demand
    o :: Int
    d :: Int
    v :: Int
end


@inline besparse(adj::Union{Graph,MetaDiGraph}, x) = besparse(make_adj_matrix(adj), x)
"make a new sparse matrix with non zero entries filled with `x`"
@inline besparse(adj::SparseMatrixCSC, x::T) where {T<:Real} =
    besparse(adj, fill(x, length(adj.nzval)))
"make a new sparse matrix with the nzval replaced with `x`"
@inline besparse(adj::SparseMatrixCSC, x::Vector{T}) where {T} =
    SparseMatrixCSC{T,Int}(adj.m, adj.n, adj.colptr, adj.rowval, x)
"make a vector of sparse matrix with the nzval replaced with `x[:, i]`"
@inline besparse(adj::SparseMatrixCSC, x::Matrix{T}) where {T} = [
    SparseMatrixCSC{T,Int}(adj.m, adj.n, adj.colptr, adj.rowval, x[:, i]) for
    i = 1:size(x, 2)
]
"make a matrix of sparse matrix with the nzval replaced with `x[:, i, j]`"
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

"convert (x, y) cordinate to lienar ones"
@inline to_ind(x, y, i, j) = begin
    @assert i >= 1 && i <= x
    @assert j >= 1 && j <= y
    (i - 1) * y + j 
end

"problem instance, since the air graph is not fixed, we don't store it"
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
            println(fd, src(e), " ", dst(e), " ", 
            lg[e, :length], " ", lg[e, :cap], " ",  lg[e, :cost])
        end
        for d in D
            println(fd, d.o, " ", d.d, " ", d.v)
        end
    end
    nothing
end

"read one line from the standard output and covnert it to type, use array of types for different types"
readL(fd, type=Int) = parse.(type, split(readline(fd)))

load_instance(fn) = open(fn, "r") do fd
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



load_planar(fn) = open(fn, "r") do fd
        nvlg, = readL(fd)
        nelg, = readL(fd)
        lD, = readL(fd)
        lg = MetaDiGraph(nvlg)
        for i in 1:nvlg
            lg[i, :x] = sin(2 * i / nvlg * π) + 2
            lg[i, :y] = cos(2 * i / nvlg * π) + 2
        end
        for _ in 1:nelg
            x, y, co, ca = readL(fd)
            add_edge!(lg, x, y)
            lg[x, y, :cost] = co
            lg[x, y, :cap] = ca
        end
        D = Vector{Demand}()
        sizehint!(D, lD)
        for _ in 1:lD
            o, d, v = readL(fd)
            push!(D, Demand(o, d, v))
        end
        lg, D
    end
    
"add new edges with random cap/cost"
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
"add new edges w/o random cap/cost"
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
    parent::Vector{U}
    distance::Vector{T}
    visited::Vector{Bool}
    q::PriorityQueue{U,T, Base.Order.ForwardOrdering}
end

"store key on arcs as a sparse graph"
function graph_sparse_weight(g; key=:time, n=nv(g))
    w = spzeros(n, n)
    for i in vertices(g), j in inneighbors(g, i)
        w[j, i] = get_prop(g, j, i, key)
    end
    w
end

"DFS to see if a node t is reachable while not visiting any already visited node(bool vec)"
function can_reach(g, s, t, visited)
    s == t && return true
    reach = fill(false, nv(g))
    reach[t] = true
    q = Queue{Int}()
    enqueue!(q, t)
    while length(q) > 0
        u = dequeue!(q)
        for i in inneighbors(g, u)
            if i == s
                return true
            elseif i ∈ visited
                continue
            elseif !reach[i]
                reach[i] = true
                enqueue!(q, i) 
            end
        end
    end

    return false
   
end
"dijkstra on a grid"
function edge_dist(g, t)
    dist = fill(typemax(Int), nv(g))
    dist[t] = 0
    q = PriorityQueue{Int, Int}()
    q[t] = 0
    while length(q) > 0
        u = dequeue!(q)
        nd = dist[u] + 1
        for i in inneighbors(g, u)
            if dist[i] > nd
                dist[i] = nd + 1
                q[i] = nd
            end
        end
    end
    dist
end

function random_walk(g, s, t)
    w = Int[s]
    visited = Set{Int}([s])
    dist = exp.(-10 ./ nv(g) .* edge_dist(g, t))
    c = s
    while c != t
        choices = [i for i in outneighbors(g, c) if i ∉ visited && can_reach(g, i, t, visited)]
        if length(choices) == 0
            break
        end
        c = sample(choices, Weights(dist[choices]))
        # c = rand(choices)
        push!(w, c)
        push!(visited, c)
    end
    w
end
function gen_instance(g, n, s, v, d)
    @assert n <= nv(g) * (nv(g) - 1)
    D = Dict{Tuple{Int, Int}, Int}()
    
    for e in E(g)
        g[e, :cap] = 0
    end

    while length(D) < n
        i, j = rand(1:nv(g)), rand(1:nv(g) - 1)
        if i <= j
            j += 1
        end
        (i, j) ∈ keys(D) && continue
        ft = 0
        for _ in 1:s
            f = rand(v)
            ft += f
            w = random_walk(g, i, j)
            for (a, b) in make_pair(w)
                g[a, b, :cap] = g[a, b, :cap] + f
            end
        end
        D[(i, j)] = ft
    end

    for e in E(g)
        g[e, :cap] = max(1, ceil(Int, g[e, :cap] * rand(d)))
    end
    g, [Demand(k..., v) for (k, v) in D]
end
export random_walk
"new lazy dijkstra state"
function new_state(g, src::U, ::AbstractMatrix{T}) where {T,U}
    state = LazyDijkstraState{T,U}(
        src,
        zeros(Int, nv(g)),
        fill(typemax(T), nv(g)),
        zeros(Bool, nv(g)),
        PriorityQueue{U,T}()
    )
    state.distance[src] = 0
    state.q[src] = zero(T)
    state
end
abstract type AbstractArcFilter end
valid(::AbstractArcFilter, a, b) = error("Not implemented")
struct AllArcS end
const AllArc = AllArcS()
@inline valid(::AllArcS, _, _) = true

"""
Dijkstra algorithm from src to dst, 
if lazy is true the algorithm will no calculate all of the dist
and will stop early,
filter is a function that can disable arcs dynamically, assumed to be deterministic wrt to its input

"""
function lazy_dijkstra(
    g::AbstractGraph,
    src::I,
    dst::I;
    w::AbstractMatrix{T},
    state::LazyDijkstraState{T, I},
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

            alt = d + convert(T, w[u, v])

            if !visited[v] && distance[v] > alt
                parent[v] = u
                distance[v] = alt
                q[v] = alt
            end
        end

        if lazy && u == dst
            return d, state
        end
    end
    return distance[dst], state
end

function ford_fulkerson(mg, lg_adj, lg_cap, lg, ag_adj, ag, l, o, d, v, w, P, air_cap)
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
        idp, irp, jrp = split_air(nv(ag), nv(lg), i, j)
        air_cap * l[idp][irp, jrp] - ax[idp][irp, jrp]
    end
    djf(i, j) = cap(i, j) > 0
    cost = 0
    while v > 1e-10
        c, state = lazy_dijkstra(mg, o, d; w=w, filter=djf, state=new_state(mg, o, w))
        p = dijkstra_to_path(state, o, d)
        Δv = min(minimum(cap(i, j) for (i, j) in make_pair(p)), v)

        cost += Δv * c
        v = max(0, v - Δv)
        
        for (i, j) in make_pair(p)
            if max(i, j) ≤ nv(lg)
                lx[i, j] += Δv
            elseif min(i, j) ≤ nv(lg)
                nothing
            else
                idp, irp, jrp = split_air(nv(ag), nv(lg), i, j)
                ax[idp][irp, jrp] += Δv
            end
        end
        push!(paths, p)
        push!(flows, Δv)
    end
    paths, (flow=flows, lx=lx, ax=ax, cost=cost)
end
"convert dijkstra state to paths"
function dijkstra_to_path(
    pred::AbstractVector{U}, 
    s::U, t::U) where {U}

    c = t
    res = [t]
    while c != s
        c = pred[c]
        push!(res, c)
    end
    reverse!(res)
    res
end  


@inline dijkstra_to_path(state::LazyDijkstraState, 
    s::U, t::U) where U = dijkstra_to_path(
        state.parent, s, t)


"make a highlevel graph by stacking the graph ag P times on the graph lg"
function combgraph(lg, ag, a2l, P; download_cost, upload_cost)
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
        s = src(e) + nv(lg) + (p - 1) * nv(ag)
        d = dst(e) + nv(lg) + (p - 1) * nv(ag)
        
        add_edge!(mg, s, d)
    end

    for p in 1:P,
        (s, d) in a2l
    
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

function mk_grid(x, y)
    graph = MetaDiGraph(x * y)
    for i in 1:x,
        j in 1:y
        
        ind = to_ind(x, y, i , j)
        set_prop!(graph, ind, :x, i * 10)
        set_prop!(graph, ind, :y, j * 10)
    end
    @assert all(has_prop(graph, i, :x) for i in 1:(x * y))
    graph
end

function mk_land_graph(x, y; cap_range, cost_range)    
    lg = mk_grid(x, y)
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
"make the air graph corresponding to the grid"
function mk_air_graph(x, y, skip)
    ag = mk_grid(x, y)
    for i in 1:skip:x,
        j in 1:skip:y
        ind = to_ind(x, y, i , j)

        j > skip && add_land_edge(ag, ind, to_ind(x, y, i, j - skip))
        j <= y - skip && add_land_edge(ag, ind, to_ind(x, y, i, j + skip))
        i > skip && add_land_edge(ag, ind, to_ind(x, y, i - skip, j))
        i <= x - skip && add_land_edge(ag, ind, to_ind(x, y, i + skip, j))
    end
    ag, vp = induced_subgraph(ag, argmax(length, connected_components(ag)))
    a2l = Dict((i, vp[i]) for i in eachindex(vp))
    
    l2a = Dict(map(reverse, collect(a2l)));
   
    ag, a2l, l2a
end

"faster sum for jump variables, i think"
function sum_affine(x, acc = AffExpr(0.0))
    for i in x
        add_to_expression!(acc, i)
    end
    acc
end

"Solve problem on land (l=0)"
function init_solve_land(D::Vector{Demand},
    lg; 
    feas_only=false, 
    w_cap::SparseMatrixCSC=graph_sparse_weight(lg, key=:cap),
    w_cost::SparseMatrixCSC=graph_sparse_weight(lg, key=:cost),
    adj=make_adj_matrix(lg),
    model = multiflow_gurobi())

    @assert nnz(w_cost) == nnz(w_cap) == nnz(adj)
    K = length(D)

    x_ = @variable(model, [1:nnz(adj), 1:K], lower_bound = 0)
    x = besparse(adj, x_)
    
    cape = [sum_affine(x[k].nzval[i] for k in 1:K) for i in 1:nnz(adj)]

    
    deme = [sum_affine((-x[k][i, j] for j in ON(lg, i)),
                sum_affine(x[k][j, i] for j in IN(lg, i)))
                for k in 1:K, i=1:nv(lg)]

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
"""
prepare a solver to solve problem on land (l=0),
the weights have to be given afterwards
"""
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
"helper for init_solve_land"
function set_comodity_demand(state, o, d, v, k)
    set_normalized_rhs(state.dem[k, o], -v)
    set_normalized_rhs(state.dem[k, d], v)
end
"helper for init_solve_land"
function set_comodity_var(state, o, d, k, u, v=1)
    set_normalized_coefficient(state.dem[k, o], u, v)
    set_normalized_coefficient(state.dem[k, d], u, -v)
end
"solve the OPAC with a solver"
function solve(D::Vector{Demand}, P::Int, lg, ag, l2a; air_cost=1_000, air_cap=10_000, L=4000, 
    model = multiflow_gurobi(),
    lg_adj=make_adj_matrix(lg),
    ag_adj=make_adj_matrix(ag),
    lg_cap=graph_sparse_weight(lg, key=:cap),
    lg_cost=graph_sparse_weight(lg, key=:cost),
    ag_length=graph_sparse_weight(ag, key=:length),
    upload_cost, download_cost)
    
    K = length(D)
    
    @variables(model, begin
        x_[1:ne(lg), 1:K] >= 0 # land flow
        y_[1:ne(ag), 1:P, 1:K] >= 0 # air flow
        l_[1:ne(ag), 1:P], Bin # pigeon paths
        sp[1:P, 1:nv(ag)], Bin # start node of the pigeon path, the cycles of each pigeon need to contain sp
        a[1:P], Bin # pigeon activation
        u[1:P, 1:K, 1:nv(ag)] >= 0 # upload data from job k to pigeon p at air node v
        d[1:P, 1:K, 1:nv(ag)] >= 0 # download
        U[1:P, 1:nv(ag)]#, (lower_bound=0, upper_bound=nv(ag)) # MTZ like potential variable
    end)
    x = besparse(lg_adj, x_)
    y = besparse(ag_adj, y_)
    l = besparse(ag_adj, l_)

    @constraints(model, begin
            [p=1:P-1], a[p] >= a[p+1] # give solutions order
            
            

            [p=1:P, i=1:ne(ag)], l[p].nzval[i] <= a[p] # cativation  
            

            [p=1:P], sum_affine(l[p].nzval[i] * ag_length.nzval[i] for i in 1:ne(ag)) ≤ L # max travel distance

            [p=1:P, i=V(ag)], 
            sum_affine(l[p][i, j] for j in ON(ag, i))  == 
            sum_affine(l[p][j, i] for j in IN(ag, i))  # the pigeon paths should form loops
            
            [p=1:P, i=V(ag)], sum(l[p][i, j] for j in ON(ag, i)) <= 1 # loops should be cycles
            
            [p=1:P, i=V(ag), j=ON(ag, i)], U[p, i] - U[p, j] +
             nv(ag) * (l[p][i, j] - 1 - sp[p, i] - sp[p, j]) ≤ -1 # mtz potential with variable sp

            [p=1:P], sum_affine(sp[p, i] for i in V(ag)) <= a[p] # there is one starting point

            [i=1:ne(lg)], sum_affine(x[k].nzval[i] for k in 1:K) ≤ lg_cap.nzval[i] # land cap

            [p=1:P, i=1:ne(ag)], 
            sum_affine(y[p, k].nzval[i] for k in 1:K) ≤ l[p].nzval[i] * air_cap # pigeon data cap

            [k=1:K, i=V(lg)], 
                    source_sink(D[k], i) + 
                    uplink(P, l2a, d, u, k, i) +
                    sum_affine(x[k][j, i] for j in IN(lg, i)) - 
                    sum_affine(x[k][i, j] for j in ON(lg, i)) == 0
            [p=1:P, k=1:K, i=V(ag)],
                u[p, k, i] -
                d[p, k, i] +
                sum_affine(y[p, k][j, i] for j in IN(ag, i)) - 
                sum_affine(y[p, k][i, j] for j in ON(ag, i)) == 0
        end)

    @objective(model, Min,
        sum_affine((lg_cost.nzval[i] * x[k].nzval[i] for k in 1:K for i in 1:ne(lg)),
        sum_affine((air_cost * i for i in a),
        sum_affine((upload_cost * i for i in u), 
        sum_affine(download_cost * i for i in d))))
    );
    optimize!(model)
    model
end

rep(((x_min, x_max), (y_min, y_max), (im_x, im_y), (mar_x, mar_y)), x, y) = Point(
    (x - x_min) / (x_max - x_min) * im_x + mar_x,
    (y - y_min) / (y_max - y_min) * im_y + mar_y
)

source_sink(d, i) = ((i == d.o) - (i == d.d)) * d.v
"transformations information to plot the graph"
function mkstate(g)
    x = extrema(V(g)) do v
        g[v][:x]
    end
    y = extrema(V(g)) do v
        g[v][:y]
    end
    (x, y, (1000, 1000), (20, 20)), (1040, 1040)
end
"ecol and vcol give the color of the edge"
function draw_plot(g; vcol=(g, i)->(0, 0, 0, 0.5), ecol=(g, i, j)->(0, 0, 0, 0.5), fil=(g, x, y) -> true, ps=2, fn=:png)
    state, img_s = mkstate(g)
    
    drawing = Drawing(img_s..., fn)
    background("white")
  
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
    for i in V(g)
        setcolor(vcol(g, i)...)
        circle(rep(state, g[i][:x], g[i][:y]), ps, :fill)
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
"helper for the solver, caclualte the total download - upload for a commondity k"
uplink(P, l2a, d, u, k, v) = if v ∈ keys(l2a)
    let i = l2a[v]
        sum(d[p, k, i] - u[p, k, i] for p in 1:P)
    end
else
    0
end
# TODO, make non allocating
"benders subproblem solver (multicomodity flow) with respective dual vector"
function subproblem(D::Vector{Demand}, P, lg, ag, l2a, air_cap, l; 
        ag_adj=make_adj_matrix(ag), 
        lg_adj=make_adj_matrix(lg),
        lg_cap=graph_sparse_weight(lg; key=:cap),
        lg_cost=graph_sparse_weight(lg; key=:cost),
        upload_cost,
        download_cost,
        model=multiflow_gurobi())

    
    K = length(D)
    @variables(model, begin
        x_[1:ne(lg), 1:K] >= 0 # land flow
        y_[1:ne(ag), 1:P, 1:K] >= 0 # air flow
        u[1:P, 1:K, 1:nv(ag)] >= 0 # upload data from job k to pigeon p at air node v
        d[1:P, 1:K, 1:nv(ag)] >= 0 # download
        
    end)
    y = besparse(ag_adj, y_)
    x = besparse(lg_adj, x_)
    
    @constraints(model, begin
            [i=1:ne(lg)], sum_affine(x[k].nzval[i] for k in 1:K) ≤ lg_cap.nzval[i] # land cap
            [k=1:K, i=1:nv(lg)], 
                source_sink(D[k], i) + 
                uplink(P, l2a, d, u, k, i) +
                sum_affine(x[k][j, i] for j in IN(lg, i)) - 
                sum_affine(x[k][i, j] for j in ON(lg, i)) == 0
            [p=1:P, k=1:K, i=1:nv(ag)],
                u[p, k, i] -
                d[p, k, i] +
                sum_affine(y[p, k][j, i] for j in IN(ag, i)) - 
                sum_affine(y[p, k][i, j] for j in ON(ag, i)) == 0
            
            con_[i=1:ne(ag), p=1:P], 
                sum_affine(-y[p, k].nzval[i] for k in 1:K) >= 
                    -l[p].nzval[i] * air_cap # pigeon data cap
        end)
     
    @objective(model, Min,
        sum_affine((lg_cost.nzval[i] * x[k].nzval[i] 
            for k in 1:K 
            for i in 1:ne(lg)),
        sum_affine((upload_cost * i for i in u),
        sum_affine(download_cost * i for i in d))))
    
    optimize!(model)

    if has_values(model)
    # @assert termination_status(model) ∈ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        cond = (dual.(con_))

        return (
            θ = objective_value(model),
            π = besparse(ag_adj, cond)
        )
    else
        return nothing
    end
end
"benders master problem"
function master_problem(D::Vector{Demand}, P::Int, lg, ag, l2a; air_cost=1_000, air_cap=10_000, L=4000,
    adj_matrix = make_adj_matrix(ag),
    model = multiflow_gurobi(),
    inner_model::F=multiflow_gurobi,
    ag_length=graph_sparse_weight(ag, key=:length),
    inner_time=600,
    K = length(D),
    upload_cost,
    download_cost) where F
    
    @variables(model, begin
        
        l_[1:ne(ag), 1:P], Bin # pigeon paths
        sp[1:P, 1:nv(ag)], Bin # start node of the pigeon path, the cycles of each pigeon need to contain sp
        a[1:P], Bin # pigeon activation
        U[1:P, 1:nv(ag)]#, (lower_bound=0, upper_bound=nv(ag)) # MTZ like potential variable
        θ >= 0
    end)
    
    l = besparse(adj_matrix, l_)
    
    @constraints(model, begin
            [p=1:P-1], a[p] >= a[p+1] # give solutions order
            [p=1:P, i=1:ne(ag)], l[p].nzval[i] <= a[p] # cativation  
            
            [p=1:P], sum_affine(l[p].nzval[i] * ag_length.nzval[i] for i in 1:ne(ag)) ≤ L # max travel distance

            [p=1:P, i=1:nv(ag)], 
                sum_affine(l[p][i, j] for j in ON(ag, i))  == 
                sum_affine(l[p][j, i] for j in IN(ag, i))  # the pigeon paths should form loops
                    
            [p=1:P, i=1:nv(ag)], sum_affine(l[p][i, j] for j in ON(ag, i)) <= 1 # loops should be cycles
            [p=1:P, i=1:nv(ag), j=ON(ag, i)], U[p, i] - U[p, j] + 
            nv(ag) * (l[p][i, j] - 1 - sp[p, i] - sp[p, j]) ≤ -1 # mtz potential with variable sp

            [p=1:P], sum_affine(sp[p, i] for i in V(ag)) <= a[p] # there is one starting point
            
        end)
    l_i = [besparse(adj_matrix, 0.001) for _ in 1:P]
    l_k = zeros(ne(ag), P)
    lg_adj = make_adj_matrix(lg)

    function my_callback(cb_data)
        η = .5
        begin
            l_k .= callback_value.(cb_data, l_)    :: Matrix{Float64}
            for p in 1:P
                l_i[p].nzval .= (1 - η) .* l_i[p].nzval .+ η .* clamp.(l_k[:, p], 0.0, 1.0)
            end
            ret = subproblem(D, P, lg, ag, l2a, air_cap, l_i;
                ag_adj=adj_matrix, lg_adj=lg_adj,
                upload_cost=upload_cost, download_cost=download_cost, 
                model=inner_model(inner_time))
            if ret === nothing
                return
            end
            cut = @build_constraint( 
                sum_affine((-air_cap * ret.π[p].nzval[i] * (l[p].nzval[i] - l_i[p].nzval[i])
                    for p=1:P for i=1:ne(ag)), 
                AffExpr(ret.θ)) <= θ)
            MOI.submit(model, MOI.LazyConstraint(cb_data), cut)
        end
        return
    end
    MOI.set(model, MOI.LazyConstraintCallback(), my_callback)
    @objective(model, Min, sum_affine((i * air_cost for i in a), AffExpr(0.0) + θ))
    optimize!(model)
    model
end


"benders master problem, using colgen"
function master_colgen(D::Vector{Demand}, P::Int, mg, lg, ag, l2a; air_cost=1_000, air_cap=10_000, L=4000,
    adj_matrix = make_adj_matrix(ag),
    model = multiflow_gurobi(),
    ag_length=graph_sparse_weight(ag, key=:length),
    K = length(D),
    inner_time=600,
    upload_cost,
    download_cost)
    @variables(model, begin
        
        l_[1:ne(ag), 1:P], Bin # pigeon paths
        sp[1:P, 1:nv(ag)], Bin # start node of the pigeon path, the cycles of each pigeon need to contain sp
        a[1:P], Bin # pigeon activation
        U[1:P, 1:nv(ag)], (lower_bound=0, upper_bound=nv(ag)) # MTZ like potential variable
        θ >= 0
    end)
    
    l = besparse(adj_matrix, l_)
    
    @constraints(model, begin
            [p=1:P-1], a[p] >= a[p+1] # give solutions order
            [p=1:P, i=1:ne(ag)], l[p].nzval[i] <= a[p] # cativation  
            
            [p=1:P], sum_affine(l[p].nzval[i] * ag_length.nzval[i] for i in 1:ne(ag)) ≤ L # max travel distance

            [p=1:P, i=1:nv(ag)], 
                sum_affine(l[p][i, j] for j in ON(ag, i))  == 
                sum_affine(l[p][j, i] for j in IN(ag, i))  # the pigeon paths should form loops
                    
            [p=1:P, i=1:nv(ag)], sum_affine(l[p][i, j] for j in ON(ag, i)) <= 1 # loops should be cycles
            [p=1:P, i=1:nv(ag), j=ON(ag, i)], U[p, i] - U[p, j] + 
            nv(ag) * (l[p][i, j] - 1 - sp[p, i] - sp[p, j]) ≤ -1 # mtz potential with variable sp

            [p=1:P], sum_affine(sp[p, i] for i in V(ag)) <= a[p] # there is one starting point
            
        end)
    
    l_i = [besparse(adj_matrix, 0.) for _ in 1:P]
    s0 = colgen(init_colgen(D, mg, lg, ag, l_i, nothing, upload_cost, download_cost, air_cap=air_cap))
    for i in l_i
        i.nzval .= 1.0
    end
    p0 = deepcopy(s0.p)
    su = colgen(init_colgen(D, mg, lg, ag, l_i, 0, upload_cost, download_cost; ps=deepcopy(p0), air_cap=air_cap))
    @constraint(model, objective_value(su.model) <= θ)
    @constraint(model, θ <= objective_value(s0.model))

    l_k = zeros(ne(ag), P)
    lg_adj = make_adj_matrix(lg)
    for i in l_i
        i.nzval .= 0.001
    end
    
    function my_callback(cb_data)
        η = .5
        begin
            l_k .= callback_value.(cb_data, l_)    :: Matrix{Float64}
            for p in 1:P
                l_i[p].nzval .= (1 - η) .* l_i[p].nzval .+ η .* clamp.(l_k[:, p], 0.0, 1.0)
            end

            s = colgen(init_colgen(D, mg, lg, ag, l_i, 0, 
                upload_cost, download_cost; 
                ps=deepcopy(p0), air_cap=air_cap, 
                model=multiflow_gurobi(inner_time)))

            cut = @build_constraint(sum_affine((
                -air_cap * s.ag.d[p].nzval[i] * (l[p].nzval[i] - l_i[p].nzval[i])
                    for p=1:P for i=1:ne(ag)), 
                        AffExpr(objective_value(s.model))) ≤ θ)
            MOI.submit(model, MOI.LazyConstraint(cb_data), cut)
        end
        return
    end
    MOI.set(model, MOI.LazyConstraintCallback(), my_callback)
    @objective(model, Min, sum_affine((i * air_cost for i in a), AffExpr(0.0) + θ))
    
    optimize!(model)
    model
end


"convert loop free flow to paths"
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


"utility to convert [a, b, c] to [(a, b), (b, c)]"
@inline make_pair(p) = zip(p, @view p[2:end])
"calculate cost of path"
function path_cost(w::AbstractMatrix{T}, p) where T
    cost = zero(T)
    for (i, j) in make_pair(p)
        cost += w[i, j]
    end
    cost 
end
"special strucutre to calculate the cost of a path on the merged graph WITH ADJUSTED COST"
struct RWM <: AbstractMatrix{Float64}
    c::SparseMatrixCSC{Float64, Int}
    wl::SparseMatrixCSC{Float64, Int}
    wa::Vector{SparseMatrixCSC{Float64, Int}}
    vl::Int
    va::Int
    upload_cost::Float64
    download_cost::Float64
end
Base.size(w::RWM) = let l = w.va * length(w.wa) + w.vl
    (l, l)
end
@inline function Base.getindex(r::RWM, i::Int, j::Int)
    if max(i, j) ≤ r.vl
        r.c[i, j] + r.wl[i, j]
    elseif i ≤ r.vl
        r.upload_cost
    elseif j ≤ r.vl
        r.download_cost
    else
        idp, irp, jrp = split_air(r.va, r.vl, i, j)
        
        r.wa[idp][irp, jrp]
    end

end
"special strucutre to calculate the cost of a path on the merged graph with ORIGINAL COST"
struct RC <: AbstractMatrix{Float64}
    c::SparseMatrixCSC{Float64, Int}
    n::Int
    upload_cost::Int
    download_cost::Int
end
Base.size(w::RC) = (w.n, w.n)
@inline function Base.getindex(r::RC, i::Int, j::Int)
    if max(i, j) ≤ r.n
        r.c[i, j]
    elseif i ≤ r.n
        r.upload_cost
    else
        r.download_cost
    end
end
"find the correspoding pigeon and air graph given the arc"
function split_air(nvag::Int, nvlg::Int, i::Int, j::Int)
    idp, irp = divrem(i - nvlg - 1, nvag) .+ (1, 1)
    jdp, jrp = divrem(j - nvlg - 1, nvag) .+ (1, 1)
    @assert idp == jdp
    (idp, irp, jrp) :: NTuple{3, Int}
end
"helper for type stability"
@inline float_dual(x) = (dual(x)::Float64)
"""
D: demands
mg: merged graph
lg: land graph
ag: air graph
ub: upper bound 
    - if nothing, we initialize with only phase 1 variables, 
    - if non zero we initialize with both phase variables, and 
    - if 0, we don't use phase 1 variables)

"""

function init_colgen(D, mg, lg, ag, l, ub, upload_cost, download_cost;
    P = length(l),
    model = multiflow_gurobi(),
    lg_cap = graph_sparse_weight(lg; key=:cap),
    lg_adj = make_adj_matrix(lg),
    lg_cost = graph_sparse_weight(lg; key=:cost),
    ag_adj = make_adj_matrix(ag),
    air_cap,
    wc = RC(lg_cost, nv(lg), upload_cost, download_cost),
    ps = [ford_fulkerson(mg, lg_adj, lg_cap, lg, ag_adj, ag, l, d.o, d.d, d.v, wc, P, air_cap)[1] for d in D] 
    # ps = [Vector{Int}[] for _ in 1:length(D)]
    # ps
    )

    @assert ub === nothing || ub >= 0
    if ub == 0
        phase1 = fill(0.0, length(D))
    else
        @variable(model, phase1[i=1:length(D)] >= 0)
    end

    @constraint(model, lg_w_[i=1:nnz(lg_cap)], 0 <= lg_cap.nzval[i])
    lg_w = besparse(lg_adj, lg_w_)
    
    @constraint(model, ag_w_[i=1:nnz(ag_adj), p=1:P], 0 <= air_cap * l[p].nzval[i])
    ag_w = besparse(ag_adj, ag_w_)
    
    
    nvag = nv(ag)
    nvlg = nv(lg)
    @assert length(ps) == length(D)
    fs = [@variable(model, [1:length(pp)], lower_bound=0) for pp in ps]

    @constraint(model, dem[k=1:length(D)], sum_affine(fs[k]) + phase1[k] == D[k].v)
    
    for (pp, ff) in zip(ps, fs),
        (p, f) in zip(pp, ff),
        (i, j) in make_pair(p)
        
        if max(i, j) ≤ nv(lg)
            set_normalized_coefficient(lg_w[i, j], f, 1)
        elseif min(i, j) > nv(lg)
            idp, irp, jrp = split_air(nvag, nvlg, i, j)
            set_normalized_coefficient(ag_w[idp][irp, jrp], f, 1)
        end
        
    end
    if ub === nothing
        @objective(model, Min, sum_affine(phase1))
        
    elseif ub > 0
        @objective(model, Min, 
                    sum_affine(path_cost(wc, p) * f 
                    for (pp, ff) in zip(ps, fs) 
                        for (p, f) in zip(pp, ff)) + (ub + 1) * sum_affine(phase1))
    else
        @objective(model, Min, 
        sum_affine(path_cost(wc, p) * f 
        for (pp, ff) in zip(ps, fs) 
            for (p, f) in zip(pp, ff)))
    end
    optimize!(model)

    @assert termination_status(model) == MOI.OPTIMAL
    d_lg_w = besparse(lg_adj, abs.(dual.(lg_w_)))::SparseMatrixCSC{Float64, Int}
    
    d_ag_w = besparse(ag_adj, abs.(float_dual.(ag_w_))::Matrix{Float64})::Vector{SparseMatrixCSC{Float64, Int}}
    π = dual.(dem)
    w = RWM(deepcopy(lg_cost), d_lg_w, d_ag_w, nv(lg), nv(ag), upload_cost, download_cost)
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

function load_dual(state)
    optimize!(state.model)
    @assert termination_status(state.model) == MOI.OPTIMAL
    
    state.lg.d.nzval .= abs.(dual.(state.lg.w_))
    for p in 1:state.P
        state.ag.d[p].nzval .= abs.(dual.(state.ag.w_[p]))
    end
    state.π .= dual.(state.dem)
    nothing
end
"add new col to model"
function register_path(state, dem, p, f)
    set_normalized_coefficient(dem, f, 1)
    for (i, j) in make_pair(p)
        if max(i, j) ≤ state.lg.nv
            set_normalized_coefficient(state.lg.w[i, j], f, 1)
        elseif min(i, j) > state.lg.nv
            idp, irp, jrp = split_air(state.ag.nv, state.lg.nv, i, j)
            set_normalized_coefficient(state.ag.w[idp][irp, jrp], f, 1)
        end
    end
    nothing
end
function set_phase2_obj(state)
    @objective(state.model, Min, 
        sum_affine(path_cost(state.wc, p) * f 
        for (pp, ff) in zip(state.p, state.f) 
            for (p, f) in zip(pp, ff)))
    nothing
end
function add_path(phase, s)
    new_path = 0
    cost = 0.0
    for (orig, es) in s.group
        state_w = new_state(s.mg, orig, s.w)
        for (i, d) in es
            @assert d.o == orig
            e, _ = lazy_dijkstra(s.mg, d.o, d.d; w=s.w, state=state_w, 
            # filter=(i, j) -> max(i, j) <= s.lg.nv || 
            #     min(i, j) <= s.lg.nv || let (idp, irp, jrp) = split_air(s.ag.nv, s.lg.nv, i, j)
            #     s.l[idp][irp, jrp] > 0
            #     end
                )
            
            cost += e
            p = dijkstra_to_path(state_w, d.o, d.d)
            
            if p ∉ s.p[i]
                @assert p[1] == d.o
                @assert p[end] == d.d
                
                c = path_cost(s.wc, p)
                new_path += 1

                f = @variable(s.model, lower_bound=0, start=0)
                push!(s.p[i], p)
                push!(s.f[i], f)

                if phase == 2
                    set_objective_coefficient(s.model, f, c)
                end
                register_path(s, s.dem[i], p, f)
            end
        end
    end
    
    cost1 = 0.0
    for i in 1:nnz(s.lg.adj)
        cost1 += s.lg.d.nzval[i] * s.lg.cap.nzval[i]
    end
    for p in 1:s.P,
        i in 1:nnz(s.ag.adj)
        cost1 += s.ag.d[p].nzval[i] * s.l[p].nzval[i] * s.air_cap
    end
    
    new_path, cost - cost1
end

# TODO parallelize djikstra
function colgen(state; max_time=Inf64)
    model, phase1 = state.model, state.phase1
    phase = state.ub == 0 ? 2 : 1
    now = time()
    for iter in 1:100
        if phase == 1 && maximum(value, phase1)  < 1e-10
            if state.ub === nothing
                state.w.c.nzval .= state.lg.cost.nzval
            end
            set_phase2_obj(state)        
            fix.(phase1, 0; force=true)
            load_dual(state)
            phase = 2
        end

        obj = objective_value(state.model)
        new_path, cost = add_path(phase, state)
        if  phase == 2 && obj == cost
            load_dual(state)
            break
        end
        if time() - now >= max_time
            break
        end
        if new_path == 0            
            (phase != 1 || maximum(value, phase1) < 1e-10) && break
        else
            load_dual(state)
        end
    end
    phase != 2 && println("phase 1!!!")
    state
end
export colgen, init_colgen
export master_problem, subproblem, master_colgen
export init_solve_land, set_comodity_demand
export solve, make_adj_matrix
export colgen, init_colgen
export mk_land_graph, combgraph, mk_air_graph, gen_instance, Demand
export write_instance, load_instance, Instance
export draw_plot, multiflow_cplex, multiflow_gurobi, besparse
end