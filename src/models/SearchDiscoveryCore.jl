abstract type SD <: Model end

"""
*Search and Discovery* (SD) core model. This model is a base model for all models that are subtypes of the Search and Discovery model. 

# Fields:  
- `β::Vector{T}`: vector of preference weights. 
- `Ξ::T`: baseline Ξ.
- `ρ::Vector{T}`: parameters governing decrease of Ξ across positions.
- `ξ::T`: baseline ξ.
- `ξρ::Vector{T}`: parameters governing decrease of ξ across positions.
- `cd::Union{T, Nothing}`: discovery costs. Initialized as nothing by to avoid computational cost. Can be updated through `calculate_costs!(m, data; kwargs...)`. 
- `cs::Union{T, Nothing}`: baseline search costs. Initialized in the same way as `cd`, and is also added in `calculate_costs!(m, data; kwargs...)`.
- `cs_h::Union{Vector{T}, T}`: position-specific search costs. Initialized in the same way as `cd`, and is also added in `calculate_costs!(m, data; kwargs...)`.
- `dE::Distribution`: distribution of ε_{ij}.
- `dV::Distribution`: distribution of ν_{ij}.
- `dU0::Distribution`: distribution of u_{i0}. 
- `dW::Distribution`: distribution of ω_{ij}.
- `zdfun::String`: select functional form f(h, Ξ, ρ) that determines the discovery value in position h. 
- `zsfun::String`: select functional form f(h, ξ, ξρ) that determines the search value in position h.
- `unobserved_heterogeneity::Dict`: dictionary of unobserved heterogeneity parameters and options. 
"""
@with_kw mutable struct SDCore{T} <: SD where {T <: Real}
    β::Vector{T}
    Ξ::T
    ρ::Vector{T}
    ξ::T
    ξρ::Vector{T}
    cd::Union{T, Nothing} = nothing
    cs::Union{T, Nothing} = nothing
    cs_h::Union{Vector{T}, Nothing} = nothing
    dE::Distribution
    dV::Distribution
    dU0::Distribution
    dW::Distribution
    zdfun::String
    zsfun::String
    unobserved_heterogeneity::Dict = Dict()

    @assert ρ[1]<=0 "ρ[1] must be less or equal to zero for weakly decreasing discovery value across positions."
end

""" 
*Data* type for the core Search and Discovery model. Indexing is based on sessions. 

# Fields:
- `consumer_ids::Vector{Int}`: consumer id for each session 
- `product_ids::Vector{Vector{Int}}`: product ids for each session in order
- `product_characteristics::Vector{Matrix{T}}`: product characteristics matrix
- `positions::Vector{Vector{Int}}`: positions for each session in order
- `search_paths::Union{Vector{Vector{Int}}, Nothing}`: search paths for each session, can be nothing
- `consideration_sets::Vector{Vector{Bool}}`: consideration sets for each session, booleans whether searched or not
- `purchase_indices::Vector{Int}`: which product within session is purchased
- `min_discover_indices::Vector{Int}`: which product within session is the lowest position clicked on
- `stop_indices::Vector{Int}`: which product within session is stopped a
"""
@with_kw mutable struct DataSD{T} <: Data where {T <: Real}
    consumer_ids::Vector{Int}
    product_ids::Vector{Vector{Int}}
    product_characteristics::Vector{Matrix{T}}
    positions::Vector{Vector{Int}}
    consideration_sets::Vector{Vector{Bool}}
    purchase_indices::Vector{Int}

    # The following fields are optional and not always part of the data 
    min_discover_indices::Union{Vector{Int}, Nothing} = nothing
    search_paths::Union{Vector{Vector{Int}}, Nothing} = nothing
    stop_indices::Union{Vector{Int}, Nothing} = nothing

    # Check that all vectors have the same length (number of sessions)
    @assert length(product_ids) == length(product_characteristics) == length(positions) ==
            length(consideration_sets) == length(purchase_indices)
    @assert isnothing(search_paths) || length(search_paths) == length(product_ids)
    @assert isnothing(min_discover_indices) ||
            length(min_discover_indices) == length(product_ids)
    @assert isnothing(stop_indices) || length(stop_indices) == length(product_ids)
end
# overload == so that can compare two DataSD objects
import Base.==
function ==(d1::DataSD, d2::DataSD)
    return d1.consumer_ids == d2.consumer_ids && d1.product_ids == d2.product_ids &&
           d1.product_characteristics == d2.product_characteristics &&
           d1.positions == d2.positions && d1.consideration_sets == d2.consideration_sets &&
           d1.purchase_indices == d2.purchase_indices &&
           d1.min_discover_indices == d2.min_discover_indices &&
           d1.search_paths == d2.search_paths && d1.stop_indices == d2.stop_indices
end

# Define base functions for working with the core data
function length(d::DataSD)
    return length(d.product_ids)
end
function getindex(d::DataSD, elements...)
    i = vcat(elements...)

    return DataSD(d.consumer_ids[i], d.product_ids[i], d.product_characteristics[i],
        d.positions[i], d.consideration_sets[i], d.purchase_indices[i],
        isnothing(d.min_discover_indices) ? nothing : d.min_discover_indices[i], isnothing(d.search_paths) ?
                                                                                 nothing :
                                                                                 d.search_paths[i],
        isnothing(d.stop_indices) ? nothing : d.stop_indices[i])
end

function eachindex(d::DataSD)
    return eachindex(d.product_ids)
end

function sessions_with_clicks(d::DataSD)
    has_click = x -> x[1] > 0
    return findall(has_click, d.search_paths)
end

function sessions_with_purchase(d::DataSD)
    sp = Int64[]
    for i in 1:length(d)
        if d.product_ids[i][d.purchase_indices[i]] > 0 # if product_id > 0 -> not outside option /  purchase 
            push!(sp, i)
        end
    end

    return sp
end

# Data generation 
function generate_data(m::SDCore, n_consumers, n_sessions_per_consumer;
        n_A0 = 1, n_d = 1,
        products = generate_products(n_consumers * n_sessions_per_consumer;),
        kwargs...)
    n_sessions = n_consumers * n_sessions_per_consumer

    # Set seed (is stable across threads) 
    set_seed(kwargs)

    # Unpack products
    product_ids, product_characteristics = products

    # Create positions based on number of alternatives per position
    positions = [vcat(zeros(Int64, 1 + n_A0),
                     repeat(collect(Int64, 1:((length(product_ids[i]) - 1 - n_A0) / n_d)),
                         inner = n_d)) for i in 1:n_sessions]

    # Generate search paths 
    paths, consideration_sets, indices_purchase, indices_stop, utility_purchases = generate_search_paths(
        m, product_ids, product_characteristics, positions; kwargs...)

    # Create consumer indices mapping consumers into sessions 
    consumer_ids = repeat(1:n_consumers, inner = n_sessions_per_consumer)

    # Get last product that consumer MUST have discovered, i.e., all products on same position as the lowest one that was clicked on
    indices_min_discover = get_indices_min_discover(consideration_sets, positions)

    # Create data object
    data = DataSD(consumer_ids, product_ids, product_characteristics, positions,
        consideration_sets, indices_purchase, indices_min_discover, paths, indices_stop)

    # Return together with purchase utilities 
    return data, utility_purchases
end

function update_positions!(data::DataSD, nA0, nd)
    # Update positions
    for i in eachindex(data)
        n_prod = length(data.positions[i])
        new_positions = vcat(fill(0, 1 + nA0),
            repeat(collect(Int64, 1:((n_prod - 1 - nA0) / nd)), inner = nd))
        if length(new_positions) < n_prod # last discovery reveals fewer than nd products 
            new_positions = vcat(
                new_positions, fill(new_positions[end] + 1, n_prod - length(new_positions)))
        end
        data.positions[i] .= new_positions
    end
end

function generate_data(m::SDCore, d::DataSD; kwargs...)

    # Set seed (is stable across threads) 
    set_seed(kwargs)

    # Generate new paths
    paths, consideration_sets, indices_purchase, indices_stop, utility_purchases = generate_search_paths(
        m, d.product_ids, d.product_characteristics, d.positions; kwargs...)

    # Get indices of minimum discovered product. Can be turned off to save time during simulation of many paths.
    compute_min_discover_indices = get(kwargs, :compute_min_discover_indices, true)
    indices_min_discover = if compute_min_discover_indices
        get_indices_min_discover(consideration_sets, d.positions)
    else
        nothing
    end

    # Update and return data object
    data = DataSD(d.consumer_ids, d.product_ids, d.product_characteristics, d.positions,
        consideration_sets, indices_purchase, indices_min_discover, paths, indices_stop)

    return data, utility_purchases
end

function generate_search_paths(
        m::SDCore, product_ids, product_characteristics, positions; kwargs...)

    # Extract keyword arguments 
    conditional_on_search = get(kwargs, :conditional_on_search, false)
    conditional_on_search_iter = get(kwargs, :conditional_on_search_iter, 100)

    # Number of consumers and sessions
    n_products = length.(product_ids)
    max_products_per_session = maximum(n_products)
    n_sessions = length(product_ids)

    # Extract other values 
    zdfun = get_functional_form(m.zdfun)
    zsfun = get_functional_form(m.zsfun)

    # Create empty vectors to store outputs
    paths = [zeros(Int, n - 1) for n in n_products]
    consideration_sets = [fill(false, n) for n in n_products]
    indices_purchase = zeros(Int, n_sessions)
    indices_stop = [n for n in n_products]
    utility_purchases = zeros(Float64, n_sessions)

    # Get draws from kwargs
    draws_shocks = get_precomputed_draws(kwargs)
    store_draws = get(kwargs, :store_draws, false)

    # Define chunks for parallelization. Each chunk is a range of sessions for which a single task 
    # creates the search path.
    _, data_chunks = get_chunks(n_sessions)

    # note: unpacking distributions here and passing into function saves a lot of allocations
    @unpack dU0, dE, dV, dW = m

    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            # Create local variables that the thread can work with. Pre-allocation circumvenst allocations in the loop.
            local u = zeros(Float64, max_products_per_session)
            local zs = zeros(Float64, max_products_per_session)
            local v = zeros(Float64, max_products_per_session)

            # Loop over sessions in the chunk
            for i in chunk

                # Reset 
                u .= typemin(Float64)
                zs .= typemin(Float64)
                v .= 0

                fill_path_i!(paths, consideration_sets, indices_purchase,
                    indices_stop, utility_purchases,
                    m, i, dU0, dE, dV, dW,
                    product_ids, product_characteristics, positions,
                    u, zs, v, zdfun, zsfun, draws_shocks, store_draws)

                # If conditional on click, iterate until have at least one 
                if conditional_on_search
                    iter = 1
                    while paths[i][1] == 0 && iter <= conditional_on_search_iter
                        fill_path_i!(paths, consideration_sets, indices_purchase,
                            indices_stop, utility_purchases,
                            m, i, dU0, dE, dV, dW,
                            product_ids, product_characteristics, positions,
                            u, zs, v, zdfun, zsfun, draws_shocks, store_draws)
                        iter += 1
                    end
                    if iter > conditional_on_search_iter
                        @warn "Did not generate a click in conditional_on_search_iter iterations for session $i."
                    end
                end
            end
        end
    end

    # Execute tasks
    fetch.(tasks)

    return paths, consideration_sets, indices_purchase, indices_stop, utility_purchases
end

function fill_path_i!(paths, consideration_sets, indices_purchase, indices_stop,
        utility_purchases,
        m, i, dU0, dE, dV, dW,
        product_ids, product_characteristics, positions,
        u, zs, v, zdfun, zsfun,
        draws_shocks, store_draws;
        debug_print = false)
    # Define variables tracking state during search 
    max_u = typemin(eltype(u)) # current max utility 
    max_zs = typemin(eltype(zs))  # current max search value
    ind_p = 0 # index of product to purchase
    ind_s = 0 # index of product to search
    pos = 0 # current position 
    ns = 0 # number of searches
    ij = 0  # Index tracking current product 
    n_prod = length(product_ids[i]) # number of products for consumer

    # Extract shock draws 
    u0_draws, e_draws, v_draws, w_draws = draws_shocks

    # Fill reservation values in initial awareness set
    for j in eachindex(u)
        if positions[i][j] > 0 # only for initial awareness set, indicated with position = 0. 
            # Update index of current product 
            ij = j - 1 # last 
            break
        end

        # Outside option 
        if product_ids[i][j] == 0
            # draw outside option utility 
            u[j] = take_or_generate_draw!(u0_draws, dU0, i, 1, store_draws) +
                   product_characteristics[i][1, end] * m.β[end]
            max_u = u[j]
            ind_p = 1
        else
            # Fill in reservation value for product j
            # note: storing v_j draw as also enters utility 
            v[j] = take_or_generate_draw!(v_draws, dV, i, j, store_draws)
            xβ = @views product_characteristics[i][j, :]' * m.β
            zs[j] = xβ + zsfun(m.ξ, m.ξρ, positions[i][j]) + v[j] +
                    take_or_generate_draw!(w_draws, dW, i, j, store_draws)
            # Update max search value and index
            if zs[j] > max_zs
                max_zs = zs[j]
                ind_s = j
            end
        end
    end

    # Update position and get discovery value to next one to be revealed
    pos += 1
    zd = zdfun(m.Ξ, m.ρ, pos)

    if i == 1 && debug_print
        println("############################")
        println("ij = ", ij)
        println("n_prod = ", n_prod)
        println("zs[1:5] = ", zs[1:5])
        println("initial zd = ", zd)
        println("initial max_zs = ", max_zs)
        println("initial max_u = ", max_u)
    end

    # Loop through discovering more products and searching 
    while true
        no_further_discoveries = pos > positions[i][end]
        # discover more products
        if zd >= max_u && zd >= max_zs && !no_further_discoveries
            if i == 1 && debug_print
                println("############################")
                println("DISCOVERY ")
                println("ij = ", ij)
                println("n_prod = ", n_prod)
                println("zs[1:5] = ", zs[1:5])
                println("u[1:5] = ", u[1:5])
                println("v[1:5] = ", v[1:5])
                println("Position ", pos)
                println("zd = ", zd)
                println("max_zs = ", max_zs)
                println("max_u = ", max_u)
            end

            # Update reservation values for products in next position 
            for j in (ij + 1):n_prod
                # Reached next position 
                if positions[i][j] > pos
                    ij = j - 1 # current product is last in position
                    break
                end

                # Update reservation value and max 
                v[j] = take_or_generate_draw!(v_draws, dV, i, j, store_draws)
                xβ = @views product_characteristics[i][j, :]' * m.β

                zs[j] = xβ + zsfun(m.ξ, m.ξρ, positions[i][j]) + v[j] +
                        take_or_generate_draw!(w_draws, dW, i, j, store_draws)
                if zs[j] > max_zs
                    max_zs = zs[j]
                    ind_s = j
                end
            end

            # Update discovery value and position 
            pos += 1    # next position 
            zd = zdfun(m.Ξ, m.ρ, pos) # next position discovery value

        # search next product 
        elseif max_zs >= max_u && (max_zs >= zd || no_further_discoveries)
            if i == 1 && debug_print
                println("############################")
                println("SEARCH  ")
                println("Position ", pos)
                println("zd = ", zd)
                println("zs[1:5] = ", zs[1:5])
                println("u[1:5] = ", u[1:5])
                println("v[1:5] = ", v[1:5])
                println("max_zs = ", max_zs)
                println("max_u = ", max_u)
                println("ind_s = ", ind_s)
            end

            # Increase number of searches and fill in next search 
            ns += 1
            paths[i][min(ns, end)] = ind_s
            consideration_sets[i][ind_s] = true

            # Set search value to neg. infinity so that it is not searched again
            zs[ind_s] = typemin(eltype(zs))

            # Get utility of searched product 
            # note: recovering previously stored v_j draw as enters utility and search value 
            u[ind_s] = take_or_generate_draw!(e_draws, dE, i, ind_s, store_draws) +
                       v[ind_s]

            for h in eachindex(m.β) # note: adding xβ this way to avoid allocations
                u[ind_s] += product_characteristics[i][ind_s, h] * m.β[h]
            end

            # Update max utility 
            if u[ind_s] > max_u
                max_u = u[ind_s]
                ind_p = ind_s
            end
            # Find next product to search. Note, all undiscovered and already-searched products have zs=-Inf, so that never chosen. 
            max_zs, ind_s = findmax(zs)

            # Stop and buy 
        elseif (max_u > zd || no_further_discoveries) && max_u > max_zs
            if i == 1 && debug_print
                println("############################")
                println("PURCHASE  ")
                println("ij = ", ij)
                println("n_prod = ", n_prod)
                println("zs[1:5] = ", zs[1:5])
                println("u[1:5] = ", u[1:5])
                println("v[1:5] = ", v[1:5])

                println("Position ", pos)
                println("zd = ", zd)
                println("max_zs = ", max_zs)
                println("max_u = ", max_u)
                println("ind_p = ", ind_p)
            end

            # Fill in purchase and stop indices 
            indices_purchase[i] = ind_p
            indices_stop[i] = pos > positions[i][end] ? length(positions[i]) : ij # special case when discovered all positions, in which case ij was not updated 
            utility_purchases[i] = max_u

            if i == 1 && debug_print
                println("utiltiy_purchase = ", utility_purchases[i])
            end
            break

        else
            error("Should not reach this point.")
        end
    end
    return nothing
end

function get_indices_min_discover(consideration_sets, positions)
    index_click_lowest_position = Array{Union{Int, Nothing}, 1}([findlast(C)
                                                                 for C in consideration_sets])
    index_click_lowest_position[isnothing.(index_click_lowest_position)] .= 1 # set to one for those who did not click
    indices_min_discover = [findlast(positions[i] .==
                                     positions[i][index_click_lowest_position[i]])
                            for i in eachindex(index_click_lowest_position)]

    return indices_min_discover
end

function generate_draws_with_search(
        m::SDCore, d::DataSD, n_sim, conditional_on_search_iter; kwargs...)

    # Set seed (is stable across threads) 
    set_seed(kwargs)

    # Set up storage for draws across simulations 
    draws_u0_store = Vector{Float64}[]
    draws_e_store = []
    draws_v_store = []
    draws_w_store = []

    for s in 1:n_sim

        # Pre-allocate draws for each session
        draws_u0 = fill(typemin(Float64), length(d))
        max_n_products = maximum(length.(d.product_ids))
        draws_e = fill(typemin(Float64), length(d), max_n_products)
        draws_v = fill(typemin(Float64), length(d), max_n_products)
        draws_w = fill(typemin(Float64), length(d), max_n_products)

        generate_search_paths(
            m, d.product_ids, d.product_characteristics, d.positions; kwargs...,
            conditional_on_search_iter, conditional_on_search = true,
            store_draws = true, draws_u0, draws_e, draws_v, draws_w)

        # Replace draws that were not filled (because never reached) with draws from respective distributions
        for pair in [[draws_u0, m.dU0], [draws_e, m.dE], [draws_v, m.dV], [draws_w, m.dW]]
            replace_typemin_draws!(pair[1], pair[2])
        end

        # Store draws for this simulation
        push!(draws_u0_store, draws_u0)
        push!(draws_e_store, draws_e)
        push!(draws_v_store, draws_v)
        push!(draws_w_store, draws_w)
    end

    return (draws_u0_store, draws_e_store, draws_v_store, draws_w_store)
end

# Cost computations 
"""
	caculate_costs!(m::SD, d, n_draws_cd, seed; force_recompute = true)

Calculate search and discovery costs for the Search and Discovery model `m` and data `d`. Uses `n_draws` to calculate the discovery costs using the distribution of characteristics in the data. If `force_recompute` is true, the costs are recomputed even if they are already present in the model.
"""
function calculate_costs!(m::SDCore, d, n_draws_cd;
        force_recompute = true,
        cd_kwargs...)
    # Search costs 
    if isnothing(m.cs) || force_recompute
        m.cs = calculate_search_cost(m)
        m.cs_h = calculate_position_specific_search_costs(m, d)
    end
    # Discovery costs
    if isnothing(m.cd) || force_recompute
        m.cd = calculate_discovery_cost(m, d, n_draws_cd; cd_kwargs...)
    end
    return nothing
end

""" 
	calculate_search_cost(m::SD)

Calculate search costs for the SD model given `ξ` and the distribution of ε `dE`. 
"""
function calculate_search_cost(m::SD)
    ξ = m.ξ
    F = m.dE

    return quadgk(e -> (1 - cdf(F, e)), ξ, maximum(F))[1]
end

""" 
	calculate_position_specific_search_costs(m::SD, d::DataSD)

Calculate position-specific search costs for the SD model given `ξ`, `ξρ` and the distribution of ε `dE`. 
"""
function calculate_position_specific_search_costs(m::SD, d::DataSD)
    F = m.dE
    zsfun = get_functional_form(m.zsfun)
    max_n_products = maximum(length.(d.product_ids))
    ξ_j = [zsfun(m.ξ, m.ξρ, h) for h in 1:max_n_products]

    return [quadgk(e -> (1 - cdf(F, e)), ξ, maximum(F))[1] for ξ in ξ_j]
end

"""
	calculate_discovery_cost(m::SD, d::DataSD, n_draws; kwargs...)
"""
function calculate_discovery_cost(m::SD, d::DataSD, n_draws; kwargs...)

    # Set seed (is stable across threads) 
    set_seed(kwargs)

    # characteristics matrix without outside option 

    # Get distribution of expected utilities xβ across all products in data 
    chars = vcat([d.product_characteristics[i][d.product_ids[i] .> 0, :]
                  for i in eachindex(d)]...) # excludes outside option 
    xβ = chars * m.β

    # Get discovery value at position where beliefs are correct
    # NEEDS TO BE UPDATED
    zdfun = get_functional_form(m.zdfun)
    position_at_which_correct_beliefs = get(
        kwargs, :position_at_which_correct_beliefs, calculate_mean_position(d))
    zd = zdfun(m.Ξ, m.ρ, position_at_which_correct_beliefs)

    # Calculate discovery costs by sampling effective values using the empirical xβ distribution and taste shock assumptions 
    W = zeros(Float64, n_draws)
    fill_values_cd_compute!(W, m, xβ, zd)

    # Return mean of the values 
    return mean(W)
end

function fill_values_cd_compute!(W, m, xβ, zd::T) where {T <: Real}
    _, data_chunks = get_chunks(length(W))

    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            for i in chunk
                e = rand(m.dE)
                v = rand(m.dV)
                w = rand(m.dW)
                xβi = rand(xβ)
                W[i] = max(zero(T), xβi + v + min(m.ξ + w, e) - zd)
            end
        end
    end

    fetch.(tasks)

    return nothing
end

function calculate_mean_position(d)
    # Get positions without outside option 
    positions = 0:maximum(vcat(d.positions...))
    return round(Int, mean(positions[positions .> 0]))
end

# Reservation values / inverse calculations for costs 
"""
	calculate_discovery_value(G::Normal, m::SD)

Calculate the discovery value zd given cs, cd, and ξ from SD model `m`. Assumes that pre-search values xβ + v + w follow normal distribution `G`.

"""

function calculate_discovery_value(G::Normal, m::SD)
    ξ, cs, cd = m.ξ, m.cs, m.cd

    if typeof(m.dE) <: Normal == false
        throw(ArgumentError("Discovery value computation currently only defined for normal distribution of ε."))
    end

    zd = if integrate_cdfsingle(cd, ξ, cs, mean(G), std(G)) - cd ≈ -cd  # case where no convergence 
        -cd
    elseif cd <= 0 || (std(G) > 1e9 && cd <= 1e8)
        Inf
    else
        fzero(t -> integrate_cdfsingle(t, ξ, cs, mean(G), std(G)) - cd, cd)
    end
    return zd
end

function integrate_cdfsingle(z, ξ, cs, μ, σ)
    a = z - μ
    b = σ
    f(x) = pdf(Normal(), x)
    F(x) = cdf(Normal(), x)

    return (1 - F((z - ξ - μ) / σ)) * (μ - z - cs) + σ * f((z - ξ - μ) / σ) +
           (z - μ) * bvncdf(a / sqrt(1 + b^2), (μ + ξ - z) / σ, -b / sqrt(1 + b^2)) -
           σ * (-b / sqrt(1 + b^2) * f(a / sqrt(1 + b^2)) *
            (1 - F((z - ξ - μ) / σ * sqrt(1 + b^2) - a * b / sqrt(1 + b^2))) +
            F(a - b * (z - ξ - μ) / σ) * f((z - ξ - μ) / σ)) +
           1.0 / sqrt(1 + b^2) * f(a / sqrt(1 + b^2)) *
           (1 - F((z - ξ - μ) / σ * sqrt(1 + b^2) - a * b / sqrt(1 + b^2)))
end

function calculate_ξ(m::SD)
    cs = m.cs
    F = m.dE
    if F == Normal() # Faster way when having std normal, and also structured to be suitable for Autodiff
        function fz_N(cs)
            fzero(ξ -> -ξ + ξ * cdf(F, ξ) + pdf(F, ξ) - cs, -abs(cs) * 10, 100 * std(F))
        end
        ξ = fz_N(cs)
        return ξ
    else
        fz(cs) = fzero(ξ -> zs_inner_integral(ξ, F) - cs, -cs, 30 * std(F))
        ξ = fz(cs)
        return ξ
    end
end

# Consumer welfare
"""
	calculate_welfare(m::SDCore, data::DataSD, n_sim; method = "effective_values", kwargs...)

Calculate consumer welfare for the Search and Discovery model `m` using the data `data` and `n_sim` simulation draws. 
	
# Keyword arguments:
- `method::String`: method to use for welfare calculation. Default is `"effective_values"`, where welfare is calculated using effective values. Alternatively can be `"simulate_paths"`, where welfare is calculated by simulating search paths.
- `kwargs...`: additional keyword arguments passed to the method used for welfare calculation.
"""
function calculate_welfare(m::SDCore, data::DataSD, n_sim;
        method = "effective_values",
        kwargs...)

    # Assert costs are part of model
    if isnothing(m.cs) || isnothing(m.cd)
        throw(ArgumentError("Search and discovery costs not calculated. Run calculate_costs! first."))
    end

    if method == "simulate_paths"
        if var(m.dW) > 0
            throw(ArgumentError("Computing welfare using path simulations is not implemented for non-constant search cost shocks. Use effective values instead."))
        end
        if m.ξρ[1] != 0
            throw(ArgumentError("Computing welfare using path simulations is not implemented for position-specific search costs. Use effective values instead."))
        end
        return calculate_welfare_simpaths(m, data, n_sim; kwargs...)
    elseif method == "effective_values"
        return calculate_welfare_effective_values(m, data, n_sim; kwargs...)
    else
        throw(ArgumentError("Method $method not recognized."))
    end
end

function calculate_welfare_simpaths(m::SDCore, data::DataSD, n_sim; kwargs...)

    # Set seed 
    set_seed(kwargs)

    # Average across all consumers 
    utility_choice_avg = zeros(Float64, n_sim)
    search_costs_avg = zeros(Float64, n_sim)
    discovery_costs_avg = zeros(Float64, n_sim)
    n_ses = length(data)

    # Average conditional on click 
    utility_choice_conditional_on_search = zeros(Float64, n_sim)
    search_costs_conditional_on_search = zeros(Float64, n_sim)
    discovery_costs_conditional_on_search = zeros(Float64, n_sim)

    # Average conditional on purchase
    utility_choice_conditional_on_purchase = zeros(Float64, n_sim)
    search_costs_conditional_on_purchase = zeros(Float64, n_sim)
    discovery_costs_conditional_on_purchase = zeros(Float64, n_sim)

    # Iterate over simulation draws and calculate welfare measures for each 
    for sim in 1:n_sim

        # Get draws shocks if provided 
        draws_u0, draws_e, draws_v, draws_w = get_precomputed_draws_indexed(kwargs, sim)

        # Generate data from new seed 
        new_seed = rand(1:(10^9))
        d_sim, utility_purchases = generate_data(m, data; kwargs...,
            seed = new_seed, draws_u0, draws_e, draws_v, draws_w)

        # Compute paid costs 
        search_costs, discovery_costs = calculate_costs_in_sessions(m, d_sim)

        # Add averages 
        utility_choice_avg[sim] = sum(utility_purchases) / n_ses
        search_costs_avg[sim] = sum(search_costs) / n_ses
        discovery_costs_avg[sim] = sum(discovery_costs) / n_ses

        # Conditional on click: compute only for sessions with at least one click
        i_ses_with_clicks = sessions_with_clicks(d_sim)
        n_sessions_with_clicks = length(i_ses_with_clicks)

        search_costs, discovery_costs = calculate_costs_in_sessions(
            m, d_sim[i_ses_with_clicks])
        utility_choice_conditional_on_search[sim] = sum(utility_purchases[i_ses_with_clicks]) /
                                                    n_sessions_with_clicks
        search_costs_conditional_on_search[sim] = sum(search_costs) / n_sessions_with_clicks
        discovery_costs_conditional_on_search[sim] = sum(discovery_costs) /
                                                     n_sessions_with_clicks

        # Conditional on purchase
        i_ses_with_purchase = sessions_with_purchase(d_sim)
        n_sessions_with_purchase = length(i_ses_with_purchase)
        search_costs, discovery_costs = calculate_costs_in_sessions(
            m, d_sim[i_ses_with_purchase])

        utility_choice_conditional_on_purchase[sim] = sum(utility_purchases[i_ses_with_purchase]) /
                                                      n_sessions_with_purchase
        search_costs_conditional_on_purchase[sim] = sum(search_costs) /
                                                    n_sessions_with_purchase
        discovery_costs_conditional_on_purchase[sim] = sum(discovery_costs) /
                                                       n_sessions_with_purchase
    end

    welfare_avg = utility_choice_avg - search_costs_avg - discovery_costs_avg
    welfare_conditional_on_search = utility_choice_conditional_on_search -
                                    search_costs_conditional_on_search -
                                    discovery_costs_conditional_on_search
    welfare_conditional_on_purchase = utility_choice_conditional_on_purchase -
                                      search_costs_conditional_on_purchase -
                                      discovery_costs_conditional_on_purchase

    # Show warning if no session with at least one click 
    if isempty(sessions_with_clicks(data))
        @warn "No sessions with at least one click. Conditional welfare is NaN."
    end

    # Show warning if no session with at least one purchase
    if isempty(sessions_with_purchase(data))
        @warn "No sessions with at least one purchase. Conditional welfare is NaN."
    end

    # Return averages across simulations as Dict 
    result = Dict()
    result[:average] = Dict(
        :welfare => mean(welfare_avg),
        :utility => mean(utility_choice_avg),
        :search_costs => mean(search_costs_avg),
        :discovery_costs => mean(discovery_costs_avg))
    result[:conditional_on_search] = Dict(:welfare => mean(welfare_conditional_on_search),
        :utility => mean(utility_choice_conditional_on_search),
        :search_costs => mean(search_costs_conditional_on_search),
        :discovery_costs => mean(discovery_costs_conditional_on_search))
    result[:conditional_on_purchase] = Dict(
        :welfare => mean(welfare_conditional_on_purchase),
        :utility => mean(utility_choice_conditional_on_purchase),
        :search_costs => mean(search_costs_conditional_on_purchase),
        :discovery_costs => mean(discovery_costs_conditional_on_purchase))

    return result
end

function calculate_costs_in_sessions(m::SDCore, d)

    # Assert costs are part of model
    if isnothing(m.cs) || isnothing(m.cd)
        throw(ArgumentError("Search and discovery costs not calculated. Run calculate_costs! first."))
    end

    # paid search costs are the number of searches multiplied with search costs
    search_costs = [m.cs * sum(path > 0 for path in d.search_paths[i])
                    for i in eachindex(d)]

    # number of discovery is equal to the position of the last product discovered because 
    # position = 0 is initial awareness set. So if stops on position =1 is one discovery, and so on. 
    discovery_costs = [m.cd * d.positions[i][d.stop_indices[i]] for i in eachindex(d)]

    return search_costs, discovery_costs
end

function calculate_welfare_effective_values(m::SD, d::DataSD, n_sim; kwargs...)

    # Set seed
    set_seed(kwargs)

    # Average across all consumers 
    eff_value_choice_avg = zeros(Float64, n_sim)
    discovery_costs_avg = zeros(Float64, n_sim)

    # Average conditional on click 
    eff_value_choice_conditional_on_search = zeros(Float64, n_sim)
    discovery_costs_conditional_on_search = zeros(Float64, n_sim)

    # Average conditional on purchase
    eff_value_choice_conditional_on_purchase = zeros(Float64, n_sim)
    discovery_costs_conditional_on_purchase = zeros(Float64, n_sim)

    # Pre-allocate vectors to store welfare measures
    n_ses = length(d)
    vectors_to_fill = preallocate_welfare_vectors(n_ses)

    # Extract funtional forms 
    zdfun = get_functional_form(m.zdfun)
    zsfun = get_functional_form(m.zsfun)

    # Loop over sims and calculate welfare measures for each
    for s in 1:n_sim

        # Retrieve draws for current simulation (if available)
        draws_s = get_precomputed_draws_indexed(kwargs, s)

        # Generate welfare measures 
        welfare_measures = _calculate_welfare_effective_values(
            m, d, vectors_to_fill, zdfun, zsfun,
            draws_s)

        # Unpack and fill in welfare measures
        eff_value_choice_avg[s], discovery_costs_avg[s] = welfare_measures[1]
        eff_value_choice_conditional_on_search[s], discovery_costs_conditional_on_search[s] = welfare_measures[2]
        eff_value_choice_conditional_on_purchase[s], discovery_costs_conditional_on_purchase[s] = welfare_measures[3]
    end

    welfare_avg = eff_value_choice_avg - discovery_costs_avg
    welfare_conditional_on_search = eff_value_choice_conditional_on_search -
                                    discovery_costs_conditional_on_search
    welfare_conditional_on_purchase = eff_value_choice_conditional_on_purchase -
                                      discovery_costs_conditional_on_purchase

    # Return averages across simulations as Dict 
    result = Dict()
    result[:average] = Dict(
        :welfare => mean(welfare_avg),
        :eff_value_choice => mean(eff_value_choice_avg),
        :discovery_costs => mean(discovery_costs_avg))
    result[:conditional_on_search] = Dict(:welfare => mean(welfare_conditional_on_search),
        :eff_value_choice => mean(eff_value_choice_conditional_on_search),
        :discovery_costs => mean(discovery_costs_conditional_on_search))
    result[:conditional_on_purchase] = Dict(
        :welfare => mean(welfare_conditional_on_purchase),
        :eff_value_choice => mean(eff_value_choice_conditional_on_purchase),
        :discovery_costs => mean(discovery_costs_conditional_on_purchase))
    return result
end

function preallocate_welfare_vectors(n_ses)

    # Average across all consumers 
    eff_value_choice_avg = zeros(Float64, n_ses)
    discovery_costs_avg = zeros(Float64, n_ses)

    # Average conditional on click 
    eff_value_choice_conditional_on_search = zeros(Float64, n_ses)
    discovery_costs_conditional_on_search = zeros(Float64, n_ses)
    clicked = fill(false, n_ses) # track number of clicks to calculate conditional 

    # Average conditional on purchase
    eff_value_choice_conditional_on_purchase = zeros(Float64, n_ses)
    discovery_costs_conditional_on_purchase = zeros(Float64, n_ses)

    purchased = fill(false, n_ses) # track number of purchases to calculate conditional

    vectors_to_fill = (eff_value_choice_avg, discovery_costs_avg,
        eff_value_choice_conditional_on_search, discovery_costs_conditional_on_search,
        clicked, eff_value_choice_conditional_on_purchase,
        discovery_costs_conditional_on_purchase, purchased)

    return vectors_to_fill
end

function _calculate_welfare_effective_values(
        m::SDCore, d::DataSD, vectors_to_fill, zdfun, zsfun, draws_shocks)
    n_ses = length(d)

    # Define chunks for parallelization. Each chunk is a range of sessions for which a single task 
    # creates the search path.
    _, data_chunks = get_chunks(n_ses)

    max_products_per_session = maximum(length.(d.product_ids))

    # note: unpacking distributions here and passing into function saves a lot of allocations
    @unpack dU0, dE, dV, dW = m

    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin

            # Define local variables for each thread (pre-allocation)
            local u = zeros(Float64, max_products_per_session)
            local zs = zeros(Float64, max_products_per_session)
            local ws = zeros(Float64, max_products_per_session)
            local ws_tilde = zeros(Float64, max_products_per_session)
            local e = zeros(Float64, 1)
            local v = zeros(Float64, 1)
            local w = zeros(Float64, 1)

            vectors_preallocated = (u, zs, ws, ws_tilde, e, v, w)

            for i in chunk
                u .= typemin(Float64)
                zs .= typemin(Float64)
                ws .= typemin(Float64)
                fill_welfare_effective_values!(vectors_to_fill, vectors_preallocated,
                    m, zdfun, zsfun,
                    d, i,
                    draws_shocks,
                    dU0, dE, dV, dW)
            end
        end
    end

    fetch.(tasks)

    @views eff_value_choice_avg, discovery_costs_avg, eff_value_choice_conditional_on_search, discovery_costs_conditional_on_search, clicked, eff_value_choice_conditional_on_purchase, discovery_costs_conditional_on_purchase, purchased = vectors_to_fill

    n_click = sum(clicked)
    n_purch = sum(purchased)

    # Return averages across simulations
    return (sum(eff_value_choice_avg), sum(discovery_costs_avg)) ./ n_ses,
    (sum(eff_value_choice_conditional_on_search),
        sum(discovery_costs_conditional_on_search)) ./ n_click,
    (sum(eff_value_choice_conditional_on_purchase),
        sum(discovery_costs_conditional_on_purchase)) ./ n_purch
end

function fill_welfare_effective_values!(vectors_to_fill, vectors_preallocated,
        m, zdfun, zsfun,
        d, i,
        draws_shocks,
        dU0, dE, dV, dW)

    # fill in search and effective values for session i
    fill_uzw_values!(vectors_preallocated, m, zdfun, zsfun, d, i, draws_shocks,
        dU0, dE, dV, dW)

    # Extract pre-allocated vectors   
    u, zs, ws, ws_tilde, _ = vectors_preallocated

    # extract vectors to track welfare measures
    eff_value_choice_avg, discovery_costs_avg, eff_value_choice_conditional_on_search, discovery_costs_conditional_on_search, clicked, eff_value_choice_conditional_on_purchase, discovery_costs_conditional_on_purchase, purchased = vectors_to_fill

    wm, im = findmax(ws) # find max value and index of effective values

    position_chosen = d.positions[i][im]
    wm_tilde = ws_tilde[im] # get w_tilde for chosen alternative

    # Find number of discoveries 
    pos = position_chosen + 1
    zd = zdfun(m.Ξ, m.ρ, pos)
    continue_discovery = zd > wm_tilde
    while continue_discovery
        pos += 1
        zd = zdfun(m.Ξ, m.ρ, pos)
        if wm >= zd || pos >= d.positions[i][end]
            continue_discovery = false
        end
    end
    last_position_discovered = ndiscoveries = pos - 1

    # Fill in welfare measures
    eff_value_choice_avg[i] = wm_tilde
    discovery_costs_avg[i] = m.cd * ndiscoveries

    # Conditional on purchase 
    has_purchase = d.product_ids[i][im] > 0 # purchased if not outside option
    if has_purchase
        eff_value_choice_conditional_on_purchase[i] = wm_tilde
        discovery_costs_conditional_on_purchase[i] = m.cd * ndiscoveries
        purchased[i] = true
    end

    # Conditional on click
    has_click = false

    if has_purchase
        has_click = true
    else
        for j in eachindex(d.product_ids[i])
            if d.product_ids[i][j] == 0  # skip outside option 
                if im > 1
                    has_click
                    break
                end
                continue
            end
            if d.positions[i][j] < position_chosen && zs[j] >= wm  # discovered before chosen alternative
                has_click = true
                break
            elseif d.positions[i][j] <= last_position_discovered && zs[j] >= wm_tilde # discovered at the same time or after chosen alternative. 
                has_click = true
                break
            end
        end
    end

    if has_click
        eff_value_choice_conditional_on_search[i] = wm_tilde
        discovery_costs_conditional_on_search[i] = m.cd * ndiscoveries
        clicked[i] = true
    end
end

function fill_uzw_values!(vectors_preallocated, m, zdfun, zsfun, d, i, draws_shocks,
        dU0, dE, dV, dW)
    @views chars = d.product_characteristics[i]
    @views positions = d.positions[i]
    @views product_ids = d.product_ids[i]

    u, zs, ws, ws_tilde, e, v, w = vectors_preallocated
    u0_draws, e_draws, v_draws, w_draws = draws_shocks

    for j in eachindex(product_ids)
        # Outside option
        if product_ids[j] == 0
            u[j] = take_or_generate_draw!(u0_draws, dU0, i, j, false) +
                   chars[j, end] * m.β[end]
            ws[j] = u[j]
            ws_tilde[j] = u[j]
            # zs not used 
        else
            # Take draws 
            e[1] = take_or_generate_draw!(e_draws, dE, i, j, false)
            v[1] = take_or_generate_draw!(v_draws, dV, i, j, false)
            w[1] = take_or_generate_draw!(w_draws, dW, i, j, false)

            # Fill in utility
            xβ = @views chars[j, :]' * m.β
            u[j] = xβ + e[1] + v[1]

            # Fill in search value 
            ξ_j = zsfun(m.ξ, m.ξρ, positions[j])
            zs[j] = xβ + ξ_j + v[1] + w[1]

            # Fill in effective value 
            ws[j] = xβ + v[1] + min(ξ_j + w[1], e[1])
            ws_tilde[j] = ws[j]

            if positions[j] > 0 # only account for discovery value when not in initial awareness set
                zd_j = zdfun(m.Ξ, m.ρ, positions[j])
                ws[j] = min(ws[j], zd_j) # + eps(eltype(ws)) * ws[j]
            end
        end
    end
    return nothing
end

# Fit evaluations 
function calculate_fit_measures(m::SDCore, data::DataSD, n_sim; kwargs...)

    # Set seed 
    set_seed(kwargs)

    # Initialize empty arrays (otherwise error of not found)
    click_stats = []
    purchase_stats = []
    stop_probabilities = []

    # Track statistics across simulations to get percentiles 
    click_probability_per_pos = zeros(Float64, maximum(length.(data.product_ids)), n_sim)
    purchase_probability_per_pos = zeros(Float64, maximum(length.(data.product_ids)), n_sim)

    conditional_on_search = get(kwargs, :conditional_on_search, false)

    # Generate data from new seed 
    sim_seeds = rand(1:1000000, n_sim)

    n_sim_without_nan_purchase = 0
    for s in 1:n_sim

        # note: generating unconditional data is much faster than conditional data. So we generate unconditional data and 
        # update the statistics to condition on clicks, using that, e.g., P(click pos 1 | click) = P(click pos 0) / P(click)
        d_sim = generate_data(m, data; kwargs..., seed = sim_seeds[s],
            compute_min_discover_indices = false, conditional_on_search = false)[1]

        # Compute fit statistics 
        click_stats_i, purchase_stats_i, stop_probability_i = calculate_statistics_from_data(d_sim)

        # Fill in statistics
        if s == 1
            click_stats = click_stats_i
            purchase_stats = purchase_stats_i
            stop_probabilities = stop_probability_i
        else
            click_stats += click_stats_i
            stop_probabilities += stop_probability_i

            # Special case for purchases: if no purchases in entire dataset, we get NaN for characteristics purchased and position (divided by n_purchases)
            # In this case, we do not use statistic for where purchases are made and characteristics purchased 

            if isnan(purchase_stats[4]) == false && isnan(purchase_stats_i[4]) == false
                purchase_stats += purchase_stats_i
                n_sim_without_nan_purchase += 1
            else
                purchase_stats[1:2] += purchase_stats_i[1:2]

                if isnan(purchase_stats[4]) && isnan(purchase_stats_i[4]) == false
                    purchase_stats[3:4] = purchase_stats_i[3:4]
                    n_sim_without_nan_purchase += 1
                end
            end
        end

        # Fill in statistics for percentiles
        click_probability_per_pos[:, s] = click_stats_i[2]
        purchase_probability_per_pos[:, s] = purchase_stats_i[2]
    end

    # Compute average 
    click_stats_sim = click_stats ./ n_sim

    purchase_stats_sim = vcat(
        purchase_stats[1:2] ./ n_sim, purchase_stats[3:4] ./ n_sim_without_nan_purchase)
    stop_probablities_sim = stop_probabilities ./ n_sim

    if conditional_on_search
        prob_at_least_one_click = click_stats_sim[3]
        click_stats_sim[1:3] ./= prob_at_least_one_click
        purchase_stats_sim[1:2] ./= prob_at_least_one_click
        click_probability_per_pos ./= prob_at_least_one_click
        purchase_probability_per_pos ./= prob_at_least_one_click
        # note: cannot use same rescaling for stop probabilities as P(stop in h | no click ) != 0 
    end

    # Get stats for data 
    click_stats_data, purchase_stats_data, stop_probabilities_data = calculate_statistics_from_data(data)

    # Create dictionaries with different stats 
    click_stats_sim = Dict(:no_clicks_per_session => click_stats_sim[1],
        :click_probability_per_position => click_stats_sim[2],
        :probability_at_least_one_click_in_session => click_stats_sim[3],
        :mean_characteristics_clicked => click_stats_sim[4],
        :mean_position_clicked => click_stats_sim[5])
    click_stats_data = Dict(:no_clicks_per_session => click_stats_data[1],
        :click_probability_per_position => click_stats_data[2],
        :probability_at_least_one_click_in_session => click_stats_data[3],
        :mean_characteristics_clicked => click_stats_data[4],
        :mean_position_clicked => click_stats_data[5])

    purchase_stats_sim = Dict(:purchase_probability => purchase_stats_sim[1],
        :purchase_probability_per_position => purchase_stats_sim[2],
        :characteristics_purchased => purchase_stats_sim[3],
        :mean_position_purchased => purchase_stats_sim[4])

    purchase_stats_data = Dict(:purchase_probability => purchase_stats_data[1],
        :purchase_probability_per_position => purchase_stats_data[2],
        :characteristics_purchased => purchase_stats_data[3],
        :mean_position_purchased => purchase_stats_data[4])

    # Compute lower/upper bound based on given percentile 
    percentile_across_sims = get(kwargs, :percentile, 0.95)
    sort!(click_probability_per_pos, dims = 2)
    sort!(purchase_probability_per_pos, dims = 2)
    lb_click = click_probability_per_pos[
        :, ceil(Int, (1 - percentile_across_sims) * n_sim)]
    ub_click = click_probability_per_pos[:, floor(Int, percentile_across_sims * n_sim)]
    lb_purchase = purchase_probability_per_pos[
        :, ceil(Int, (1 - percentile_across_sims) * n_sim)]
    ub_purchase = purchase_probability_per_pos[
        :, floor(Int, percentile_across_sims * n_sim)]

    # Return dictionary with all statistics
    return Dict(:click_stats_data => click_stats_data,
        :click_stats_sim => click_stats_sim,
        :purchase_stats_data => purchase_stats_data,
        :purchase_stats_sim => purchase_stats_sim,
        :stop_probabilities_data => stop_probabilities_data,
        :stop_probabilities_sim => stop_probablities_sim,
        :bounds => [(lb_click, ub_click), (lb_purchase, ub_purchase)])
end

function calculate_statistics_from_data(d::DataSD)

    # Set up click statistics that will be filled in by looping over sessions 
    maximum_n_prod = maximum(length.(d.product_ids))
    clicks_per_pos = zeros(Int, maximum_n_prod)
    n_click_conditional_on_search = 0
    n_at_least_one_click = 0
    with_outside_option = d.product_ids[1][1] == 0
    characteristics_clicked = zeros(
        Float64, size(d.product_characteristics[1], 2) - with_outside_option)
    position_click = 0

    # Purchase statistics
    purchases_per_pos = zeros(Int, maximum_n_prod)
    characteristics_purchased = zeros(
        Float64, size(d.product_characteristics[1], 2) - with_outside_option)
    position_purchase = 0

    # Stopping probability
    stop_probabilities = zeros(Float64, maximum_n_prod)

    # Loop over sessions and fill in statistics
    for i in eachindex(d)
        # Click statistics
        clicked = false
        n_clicks_i = 0
        for j in eachindex(d.product_ids[i])
            if d.consideration_sets[i][j]
                clicked = true
                clicks_per_pos[j] += 1
                characteristics_clicked .+= d.product_characteristics[i][
                    j, 1:(end - with_outside_option)] # if outside option, last characteristic is dummy 
                n_clicks_i += 1
                position_click += d.positions[i][j]
            end
        end

        if clicked
            n_click_conditional_on_search += n_clicks_i
            n_at_least_one_click += 1
        end

        # Purchase statistics
        if d.purchase_indices[i] > 1
            j = d.purchase_indices[i]
            purchases_per_pos[j] += 1
            characteristics_purchased .+= d.product_characteristics[i][
                j, 1:(end - with_outside_option)] # if outside option, last characteristic is dummy
            position_purchase += d.positions[i][j]
        end

        # Stopping probability
        if !isnothing(d.stop_indices)
            stop_probabilities[d.stop_indices[i]] += 1
        end
    end

    n_clicks = sum(@views clicks_per_pos)
    n_purchases = sum(@views purchases_per_pos)

    n_ses = length(d)

    # Gather click statistics
    no_clicks_per_session = n_clicks / n_ses
    click_probability_per_position = clicks_per_pos ./ n_ses
    probability_at_least_one_click_in_session = n_at_least_one_click / n_ses
    mean_characteristics_clicked = characteristics_clicked / n_clicks
    mean_position_clicked = position_click / n_clicks
    click_stats = [no_clicks_per_session, click_probability_per_position,
        probability_at_least_one_click_in_session,
        mean_characteristics_clicked, mean_position_clicked]

    # Gather purchase statistics
    purchase_probability = n_purchases / n_ses
    purchase_probability_per_pos = purchases_per_pos ./ n_ses
    characteristics_purchased = characteristics_purchased / n_purchases
    mean_position_purchased = position_purchase / n_purchases
    purchase_stats = [purchase_probability, purchase_probability_per_pos,
        characteristics_purchased, mean_position_purchased]

    # Get stopping probabilities mean 
    stop_probabilities /= n_ses

    return click_stats, purchase_stats, stop_probabilities
end

function plot_across_positions(stats; kwargs...)

    # Extract statistics 
    clicks_per_pos_sim = stats[:click_stats_sim][:click_probability_per_position]
    clicks_per_pos_data = stats[:click_stats_data][:click_probability_per_position]

    purchases_per_pos_sim = stats[:purchase_stats_sim][:purchase_probability_per_position]
    purchases_per_pos_data = stats[:purchase_stats_data][:purchase_probability_per_position]

    # Extract bounds
    bounds = stats[:bounds]
    lb_click = bounds[1][1]
    ub_click = bounds[1][2]
    lb_purchase = bounds[2][1]
    ub_purchase = bounds[2][2]

    # x-axis 
    sel = if ub_click[1] == 0 # outside option -> no clicks on first element -> don't plot 
        2:length(lb_click)
    else
        1:length(lb_click) # outside option -> no clicks on first element 
    end
    x = 1:length(lb_click[sel])

    # set nice color  
    base_color = RGB(62 / 255, 100 / 255, 125 / 255)

    # Extract y-axis labels from keywords
    ylabel1 = get(kwargs, :ylabel_purch, "Purchase prob.")
    ylabel2 = get(kwargs, :ylabel_click, "Click prob.")

    # Extract y-axis tick format from keywords
    ytickformat = get(kwargs, :ytickformat, "{:.3f}")

    fig = Figure()
    ax_purch = Axis(fig[1, 1],
        xlabel = "Position",
        ylabel = ylabel1,
        ytickformat = ytickformat)

    # Plot purchase probabiltiy 
    band!(ax_purch, x, lb_purchase[sel], ub_purchase[sel], color = (base_color, 0.2))
    lines!(ax_purch, purchases_per_pos_sim[sel],
        label = "Predicted", color = base_color, linewidth = 3)
    lines!(ax_purch, purchases_per_pos_data[sel], label = "Data", color = :black,
        linestyle = :dash, linewidth = 2)
    axislegend(; framevisible = false)
    hidespines!(ax_purch, :t, :r) # only top and right

    # Plot click probability
    ax_clicks = Axis(fig[2, 1],
        xlabel = "Position",
        ylabel = ylabel2,
        ytickformat = ytickformat)
    band!(ax_clicks, x, lb_click[sel], ub_click[sel], color = (base_color, 0.2))
    lines!(ax_clicks, clicks_per_pos_sim[sel],
        label = "Predicted", color = base_color, linewidth = 3)
    lines!(ax_clicks, clicks_per_pos_data[sel], label = "Data", color = :black,
        linestyle = :dash, linewidth = 2)
    hidespines!(ax_clicks, :t, :r) # only top and right

    return fig
end

function evaluate_fit(m::SDCore, data::DataSD, n_sim; kwargs...)

    # Calculate fit measures 
    fit_measures = calculate_fit_measures(
        m, data, n_sim; return_bounds = true, kwargs...)

    # Put together plot 
    fig = plot_across_positions(fit_measures; kwargs...)
    # Plot fit measures 
    show_plot = get(kwargs, :show_plot, true)
    if show_plot
        display(fig)
    end

    return fit_measures, fig
end

# Estimation 
function prepare_arguments_likelihood(
        m::M, estimator::Estimator, d::DataSD; kwargs...) where {M <: SD}

    # Get functional forms 
    zdfun = get_functional_form(m.zdfun)
    zsfun = get_functional_form(m.zsfun)

    # get data arguments 
    data_arguments = prepare_data_arguments_likelihood(d)

    # Keep fixed seed: either random or provided by kwargs 
    seed = get(kwargs, :seed, rand(1:(10^9)))

    return data_arguments..., zdfun, zsfun, seed
end

function prepare_data_arguments_likelihood(d::DataSD)
    all_possible_positions = d.positions[argmax(maximum.(d.positions))]
    has_search = sum.(d.consideration_sets) .> 0
    has_purchase = [d.product_ids[i][d.purchase_indices[i]] > 0 for i in eachindex(d)]

    return all_possible_positions, has_search, has_purchase
end

function vectorize_parameters(m::M; kwargs...) where {M <: SD}
    # Default estimate all parameters 
    θ = if !haskey(kwargs, :fixed_parameters)
        θ = vcat(m.β, m.Ξ, m.ρ, m.ξ, m.ξρ)
    else
        fixed_parameters = get(kwargs, :fixed_parameters, nothing)
        if !isnothing(fixed_parameters)
            θ = eltype(m.β)[]
            if !fixed_parameters[1]
                θ = vcat(θ, m.β)
            end
            if !fixed_parameters[2]
                θ = vcat(θ, m.Ξ)
            end
            if !fixed_parameters[3]
                θ = vcat(θ, m.ρ)
            end
            if !fixed_parameters[4]
                θ = vcat(θ, m.ξ)
            end
            if !fixed_parameters[5]
                θ = vcat(θ, m.ξρ)
            end
        end
        θ
    end

    θ = add_distribution_parameters(m, θ, kwargs)

    return θ
end

function loglikelihood(θ::Vector{T}, model::M, estimator::SmoothMLE, data::DataSD,
        args...; kwargs...) where {M <: SD, T <: Real}

    # Extract arguments 
    all_possible_positions, has_search, has_purchase, zdfun, zsfun, seed = args

    # Extract parameters implied by θ 
    β, Ξ, ρ, ξ, ξρ, ind_last_par = extract_parameters(model, θ; kwargs...)
    dE, dV, dU0 = extract_distributions(model, θ, ind_last_par; kwargs...)

    debug_print = get(kwargs, :debug_print, false)
    if debug_print
        println("θ = $θ")
        println("β = $β")
        println("Ξ = $Ξ")
        println("ρ = $ρ")
        println("ξ = $ξ")
        println("ξρ = $ξρ")
        println("dE = $dE")
        println("dV = $dV")
        println("dU0 = $dU0")
    end

    if !isnothing(ρ) && ρ[1] > 0
        return -T(MAX_NUMERICAL)
    end

    # Pre-compute search and discovery values across positions -> same for all consumers 
    zd_h = isnothing(zdfun) ? Ξ : [zdfun(Ξ, ρ, pos) for pos in all_possible_positions]
    zs_h = isnothing(zsfun) ? ξ : [zsfun(ξ, ξρ, pos) for pos in all_possible_positions]

    if debug_print
        if !isnothing(zdfun)
            println("zd_h = $(zd_h[1:min(5, end)])")
        end
        if !isnothing(zsfun)
            println("zs_h = $(zs_h[1:min(5, end)])")
        end
    end

    # Set seed for random number generation
    Random.seed!(seed)

    # Extract number of draws 
    n_draws = estimator.options_numerical_integration.n_draws
    n_draws_purchase = estimator.options_numerical_integration.n_draws_purchases

    # Define chunks for parallelization. Each chunk is a range of consumers for which a single task 
    # calculates and sums up the likelihood. 
    _, data_chunks = get_chunks(length(data))

    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin

            # Pre-allocate arrays per task -> avoid memory allocation by not having to re-create these arrays for every consumer 
            local L = zero(T)

            for i in chunk  # Iterate over consumers in chunk 
                # Do inner likelihood calculations based on pre-allocated arrays
                if has_search[i] == 0 # Case 1: no clicks (implies also no purchase)
                    L += ll_no_searches(
                        model, zd_h, zs_h, β, dV, dU0, data, i, n_draws, false)
                elseif has_purchase[i] == 0 # Case 2: Some clicks but no purchase 
                    L += ll_search_no_purchase(
                        model, zd_h, zs_h, β, dE, dV, dU0, data, i, n_draws)
                else # Case 3: Purchase a product 
                    L += ll_purchase(
                        model, zd_h, zs_h, β, dE, dV, dU0, data, i, n_draws_purchase)
                end
            end

            return L # Return likelihood for chunk
        end
    end

    LL1 = sum(fetch.(tasks))

    LL2 = if estimator.conditional_on_search
        tasks = map(data_chunks) do chunk
            Threads.@spawn begin
                local L = zero(T)

                for i in chunk  # Iterate over consumers in chunk 
                    L += ll_no_searches(
                        model, zd_h, zs_h, β, dV, dU0, data, i, n_draws, true)
                end

                return L
            end
        end

        sum(fetch.(tasks))
    else
        zero(T)
    end

    LL = LL1 - LL2
    if get(kwargs, :debug_print, false)
        println("LL = $LL")
        println("LL1 = $LL1")
        println("LL2 = $LL2")
    end

    # prevent Inf values, helps AD
    if isinf(LL) || isnan(LL) || LL >= 0
        return -T(1e100)
    else
        return LL
    end
end

# Parameter handling 
function extract_parameters(m::M, θ::Vector{T}; kwargs...) where {M <: SD, T <: Real}
    n_beta = length(m.β)
    n_ρ = length(m.ρ)
    n_ξρ = length(m.ξρ)

    # track where in parameter vector we are and move it. 
    ind_current = 1

    # Default: estimate all parameters
    if !haskey(kwargs, :fixed_parameters)
        β = θ[1:n_beta]
        ind_current += n_beta
        Ξ = θ[ind_current]
        ind_current += 1
        ρ = θ[ind_current:(ind_current + n_ρ - 1)]
        ind_current += n_ρ
        ξ = θ[ind_current]
        ind_current += 1
        ξρ = θ[ind_current:(ind_current + n_ξρ - 1)]
        ind_current += n_ξρ
        return β, Ξ, ρ, ξ, ξρ, ind_current
    end

    # If keyword supplied, don't estimate parameters indicated in fixed_parameters
    fixed_parameters = get(kwargs, :fixed_parameters, nothing)
    β = if fixed_parameters[1]
        T.(m.β)
    else
        ind_current += n_beta
        θ[1:n_beta]
    end
    Ξ = if fixed_parameters[2]
        T(m.Ξ)
    else
        ind_current += 1
        θ[ind_current - 1]
    end
    ρ = if fixed_parameters[3]
        T.(m.ρ)
    else
        ind_current += n_ρ
        θ[ind_current:(ind_current + n_ρ - 1 - 1)]
    end
    ξ = if fixed_parameters[4]
        T(m.ξ)
    else
        ind_current += 1
        θ[ind_current - 1]
    end
    ξρ = if fixed_parameters[5]
        T.(m.ξρ)
    else
        ind_current += n_ξρ
        θ[ind_current:(ind_current + n_ξρ - 1 - 1)]
    end

    if get(kwargs, :debug_print, false)
        println("β = $β")
        println("Ξ = $Ξ")
        println("ρ = $ρ")
        println("ξ = $ξ")
        println("ξρ = $ξρ")
    end
    return β, Ξ, ρ, ξ, ξρ, ind_current
end

function construct_model_from_pars(θ::Vector{T}, m::M; kwargs...) where {M <: SD, T <: Real}

    # Extract parameters from vector, some may be fixed through kwargs 
    β, Ξ, ρ, ξ, ξρ, ind_last_par = extract_parameters(model, θ; kwargs...)
    dE, dV, dU0 = extract_distributions(m, θ, ind_last_par; kwargs...)

    # Construct model from parameters 
    m_new = SDCore{T}(; β, Ξ, ρ, ξ, ξρ, dE, dV, dU0, zdfun = m.zdfun, zsfun = m.zsfun)

    return m_new
end

"""
Construct shock distributions using variances in vector θ. Starts from index c. 
"""
function extract_distributions(m::M, θ::Vector{T}, c; kwargs...) where {M <: SD, T <: Real}

    # Default: don't estimate any variance 
    if !haskey(kwargs, :distribution_options)
        dE = m.dE # convoluted way to allow for distributions other than Normal 
        dV = m.dV
        dU0 = m.dU0
        return dE, dV, dU0
    end

    estimation_shock_distributions = get(kwargs, :distribution_options, nothing)

    # Extract distributions
    dE = if estimation_shock_distributions[1]
        dE = eval(nameof(typeof(m.dE)))(params(m.dE)[1:(end - 1)]..., abs(θ[c]))
        c += 1
        dE
    else
        m.dE
    end

    dV = if estimation_shock_distributions[2]
        dV = eval(nameof(typeof(m.dV)))(params(m.dV)[1:(end - 1)]..., abs(θ[c]))
        c += 1
        dV
    else
        m.dV
    end

    dU0 = if estimation_shock_distributions[3] && estimation_shock_distributions[1]
        dE
    elseif estimation_shock_distributions[3] && estimation_shock_distributions[2]
        dV
    elseif estimation_shock_distributions[4]
        np = length(params(m.dU0))
        eval(nameof(typeof(m.dU0)))(params(m.dU0)[1:(end - np + 1)]..., abs.(θ[c:end])...)
    else
        m.dU0
    end

    return dE, dV, dU0
end

function add_distribution_parameters(m::M, θ, kwargs) where {M <: SD}

    # Default: don't estimate any variance 
    if !haskey(kwargs, :distribution_options)
        return θ
    end
    estimation_shock_distributions = get(kwargs, :distribution_options, nothing)
    # Extract distributions
    if estimation_shock_distributions[1]
        θ = vcat(θ, params(m.dE)[end])
    end
    if estimation_shock_distributions[2]
        θ = vcat(θ, params(m.dV)[end])
    end
    if estimation_shock_distributions[4]
        θ = vcat(θ, params(m.dU0)[2:end])
    end

    return θ
end

function add_indices_min_discover!(d::DataSD)
    min_discover_indices = get_indices_min_discover(d.consideration_sets, d.positions)

    d.min_discover_indices = min_discover_indices
    return nothing
end