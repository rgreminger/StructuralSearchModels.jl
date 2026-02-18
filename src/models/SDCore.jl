"""
    InformationStructureSpecification{T} <: AbstractSpecification
Specification for the information structure in the Search and Discovery model. This specification includes the parameter γ for the search value, as well as selectors for the characteristics that enter both the search value and utility through xβ, and the characteristics that enter only the search value through xγ. By default, is initialized with everything as nothing, which means `γ` is not used, and all characteristics are used for both xβ.

## Fields:
- `γ::Vector{T}`: Vector of parameters for the search value. If `empty` (default), no search value is used.
- `κ::Vector{T}`: Vector of parameters for the utility only characteristics. If `empty`, no utility only characteristics.
- `indices_characteristics_β_union::Union{UnitRange{Int}, Vector{Int}}`: All characteristics that enter the search value and utility through `xβ` in at least one session.
- `indices_characteristics_γ_union::Union{UnitRange{Int}, Vector{Int}}`: All characteristics that enter the search value and utility through `xγ` in at least one session.
- `indices_characteristics_κ_union::Union{UnitRange{Int}, Vector{Int}}`: All characteristics that enter the search value and utility through `xκ` in at least one session.
- `indices_characteristics_β_individual::Union{UnitRange{Int}, Vector{Int}, Vector{UnitRange{Int}}, Vector{Vector{Int}}}`. By default is the same as `indices_characteristics_β_union`, but can be set to individual indices for each session.
- `indices_characteristics_γ_individual::Union{UnitRange{Int}, Vector{Int}, Vector{UnitRange{Int}}, Vector{Vector{Int}}}`. By default is the same as `indices_characteristics_γ_union`, but can be set to individual indices for each session.
- `indices_characteristics_κ_individual::Union{UnitRange{Int}, Vector{Int}, Vector{UnitRange{Int}}, Vector{Vector{Int}}}`. By default is the same as `indices_characteristics_κ_union`, but can be set to individual indices for each session.
"""
@with_kw mutable struct InformationStructureSpecification{T} <: AbstractSpecification where {T <: Real}
    γ::Vector{T}
    κ::Vector{T}

    indices_characteristics_β_union::Union{UnitRange{Int}, Vector{Int}}
    indices_characteristics_γ_union::Union{UnitRange{Int}, Vector{Int}}
    indices_characteristics_κ_union::Union{UnitRange{Int}, Vector{Int}}

    indices_characteristics_β_individual::Union{UnitRange{Int}, Vector{Int},
        Vector{UnitRange{Int}}, Vector{Vector{Int}}} = indices_characteristics_β_union
    indices_characteristics_γ_individual::Union{UnitRange{Int}, Vector{Int},
        Vector{UnitRange{Int}}, Vector{Vector{Int}}} = indices_characteristics_γ_union
    indices_characteristics_κ_individual::Union{UnitRange{Int}, Vector{Int},
        Vector{UnitRange{Int}}, Vector{Vector{Int}}} = indices_characteristics_κ_union
end

function InformationStructureSpecification(n::Int)
    γ = zeros(n)
    κ = zeros(n)
    indices_characteristics_β = 1:n-1
    indices_characteristics_γ = 1:0
    indices_characteristics_κ = 1:0

    return InformationStructureSpecification(γ, κ, indices_characteristics_β,
        indices_characteristics_γ, indices_characteristics_κ)
end

function InformationStructureSpecification(γ, κ, indices_characteristics_β,
        indices_characteristics_γ, indices_characteristics_κ)

    return InformationStructureSpecification(γ, κ, indices_characteristics_β,
        indices_characteristics_γ, indices_characteristics_κ,
        indices_characteristics_β, indices_characteristics_γ, indices_characteristics_κ)
end

function ==(s1::InformationStructureSpecification, s2::InformationStructureSpecification)
    return s1.γ == s2.γ && s1.κ == s2.κ &&
        s1.indices_characteristics_β_union == s2.indices_characteristics_β_union &&
        s1.indices_characteristics_γ_union == s2.indices_characteristics_γ_union &&
        s1.indices_characteristics_κ_union == s2.indices_characteristics_κ_union &&
        s1.indices_characteristics_β_individual == s2.indices_characteristics_β_individual &&
        s1.indices_characteristics_γ_individual == s2.indices_characteristics_γ_individual &&
        s1.indices_characteristics_κ_individual == s2.indices_characteristics_κ_individual
end

function all_characteristics_on_list(m::SDModel)
    length(m.information_structure.indices_characteristics_γ_union) == 0
end

"""
Search and Discovery (SD) core model. This model is a base model for all models that are subtypes of `SDModel`. It implements the most general specification using three shocks and both functional forms for ξ and Ξ. Currently, there is no estimation method for this model, but it is used internally to generate data.

- uᵢⱼ = xⱼβ + xⱼκ + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ(h) = xⱼβ + xⱼγ + ξ(h) + νᵢⱼ + ωᵢⱼ, ωᵢⱼ ~ dW
- uᵢ₀ = β0 + ηᵢ , ηᵢ ~ dU0
- zd(h) = zdfun(Ξ, ρ, pos) with ρ ≤ 0
- ξ(h) = zsfun(ξ, ξρ, pos)
- For the estimation, the specification of `xⱼβ`, `xⱼκ`, and `xⱼγ` are determined by `information_structure`.

# Fields:
- `β::Vector{T}`: Vector of preference weights.
- `Ξ::T`: Baseline Ξ for position 1 (not demeaned).
- `ρ::Union{T, Vector{T}} `: Parameters governing decrease of Ξ across positions.
- `ξ::T`: Baseline ξ.
- `ξρ::Union{T, Vector{T}} `: Parameters governing decrease of ξ across positions.
- `dE::Distribution`: Distribution of εᵢⱼ.
- `dV::Distribution`: Distribution of νᵢⱼ.
- `dU0::Distribution`: Distribution of ηᵢ.
- `dW::Distribution`: Distribution of ωᵢⱼ.
- `zdfun::String`: Select functional form f(h, Ξ, ρ) that determines the discovery value in position h.
- `zsfun::String`: Select functional form f(h, ξ, ξρ) that determines the search value in position h.
- `information_structure::InformationStructureSpecification{T}`: Specification of information structure, including `γ`, `κ` and characteristics for `β`, `γ`, and `κ`. See `InformationStructureSpecification` for details.
- `cs::Union{Vector{Vector{T}}, Nothing}`: Search costs on a product level. Initialized as `nothing` and only used for welfare calculations. Vector of vector, matching structure in data. Can be added through `calculate_costs!(m, data; kwargs...)`.
- `cd::Union{Vector{T}, Nothing}`: Discovery costs, specific to sessions. Initialized as `nothing` and only used for welfare calculations. Can be updated through `calculate_costs!(m, data; kwargs...)`.
- `heterogeneity::HeterogeneitySpecification`: Specification of heterogeneity (unobserved and observed) in the model.
"""
@with_kw mutable struct SDCore{T} <: SDModel where {T <: Real}
    β::Vector{T}
    Ξ::T
    ρ::Union{T, Vector{T}}
    ξ::T
    ξρ::Union{T, Vector{T}}
    dE::Distribution
    dV::Distribution
    dU0::Distribution
    dW::Distribution
    zdfun::String
    zsfun::String

    cs::Union{Vector{Vector{T}}, Nothing} = nothing
    cd::Union{Vector{T}, Nothing} = nothing

    information_structure::InformationStructureSpecification{T} = InformationStructureSpecification(length(β))
    heterogeneity::HeterogeneitySpecification{T} = HeterogeneitySpecification()

    @assert sum(ρi > 0 for ρi in ρ) == 0 "All elements of ρ must be less or equal to zero for weakly decreasing discovery value across positions."
    @assert length(information_structure.γ) == length(β) "Length of γ must be equal to length of β."
    @assert length(information_structure.κ) == length(β) "Length of κ must be equal to length of β."
end



"""
*Data* type for the core Search and Discovery model. Indexing is based on sessions. See the tutorials for examples on how such data can be simulated or constructed.

# Fields:
- `consumer_ids::Vector{Int}`: consumer id for each session.
- `product_ids::Vector{Vector{Int}}`: vector of vectors of product ids for each session.
- `product_characteristics::Vector{Matrix{T}}`: product characteristics matrix.
- `positions::Vector{Vector{Int}}`: positions for each session.
- `consideration_sets::Vector{Vector{Bool}}`: consideration sets for each session, booleans whether searched or not.
- `purchase_indices::Vector{Int}`: which product within session is purchased.
- `min_discover_indices::Union{Vector{Int}, Nothing}`: index of last product must have been discovered in session (lowest position at which click occurred). This is mainly used during estimation, and can be constructed in with the `fill_indices_min_discover!` function.
- `stop_indices::Union{Vector{Int}, Nothing}`: index of last product discovered in session (at which discovery stops). Is `nothing` if not available (e.g., when scrolling is not observed).
- `session_characteristics::Union{Vector{Vector{T}}, Nothing}`: session characteristics for each session. Is `nothing` if not available (default).
- `search_paths::Union{Vector{Vector{Int}}, Nothing}`: search paths for each session. Is `nothing` if search order is not available.
"""
@with_kw mutable struct DataSD{T} <: Data where {T <: Real}

    consumer_ids::Vector{Int}
    product_ids::Vector{Vector{Int}}
    product_characteristics::Vector{Matrix{T}}
    positions::Vector{Vector{Int}}
    session_characteristics::Union{Vector{Vector{T}}, Nothing} = nothing

    # Search paths and consideration sets
    consideration_sets::Vector{Vector{Bool}}
    purchase_indices::Vector{Int}
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

    # Ensure consumer ids
    @assert issorted(consumer_ids)
    @assert unique(consumer_ids) == collect(1:consumer_ids[end])
end

function DataSD(consumer_ids, product_ids, product_characteristics, positions,
                session_characteristics,
                consideration_sets, purchase_indices, min_discover_indices,
                search_paths, stop_indices)

    # Convert inputs to the correct types
    consumer_ids = Vector{Int64}(consumer_ids)
    product_ids = Vector{Vector{Int64}}(product_ids)
    product_characteristics = Vector{Matrix{Float64}}(product_characteristics)
    positions = Vector{Vector{Int64}}(positions)

    consideration_sets = Vector{Vector{Bool}}(consideration_sets)
    purchase_indices = Vector{Int64}(purchase_indices)

    if !isnothing(min_discover_indices)
        min_discover_indices = Vector{Int64}(min_discover_indices)
    end
    if !isnothing(search_paths)
        search_paths = Vector{Vector{Int64}}(search_paths)
    end
    if !isnothing(stop_indices)
        stop_indices = Vector{Int64}(stop_indices)
    end
    if !isnothing(session_characteristics)
        session_characteristics = Vector{Vector{Float64}}(session_characteristics)
    end

    return DataSD(consumer_ids, product_ids, product_characteristics, positions,
        session_characteristics,
        consideration_sets, purchase_indices, min_discover_indices, search_paths, stop_indices)
end

function ==(d1::DataSD, d2::DataSD)
    return d1.consumer_ids == d2.consumer_ids && d1.product_ids == d2.product_ids &&
        d1.product_characteristics == d2.product_characteristics &&
        d1.positions == d2.positions && d1.consideration_sets == d2.consideration_sets &&
        d1.purchase_indices == d2.purchase_indices &&
        d1.min_discover_indices == d2.min_discover_indices &&
        d1.search_paths == d2.search_paths && d1.stop_indices == d2.stop_indices &&
        d1.session_characteristics == d2.session_characteristics
end

# Define base functions for working with the core data
function length(d::DataSD)
    return length(d.product_ids)
end
function getindex(d::DataSD, elements...)
    i = vcat(elements...)

    if length(i) != length(unique(i))
        throw(ArgumentError("Cannot index into DataSD with duplicate indices."))
    end
    ids_new = transform_to_consecutive_ids(d.consumer_ids[i])

    return DataSD(ids_new, d.product_ids[i], d.product_characteristics[i],
        d.positions[i],
        isnothing(d.session_characteristics) ? nothing : d.session_characteristics[i],
        d.consideration_sets[i], d.purchase_indices[i],
        isnothing(d.min_discover_indices) ? nothing : d.min_discover_indices[i],
        isnothing(d.search_paths) ? nothing : d.search_paths[i],
        isnothing(d.stop_indices) ? nothing : d.stop_indices[i],
        )
end

"""
    merge_data(data1::DataSD, data2::DataSD)

Merge two `DataSD` objects into a single one by concatenating all fields. If the combined `consumer_ids` contain duplicates, they are replaced with consecutive integers.
"""
function merge_data(data1::DataSD, data2::DataSD)
    consumer_ids = vcat(data1.consumer_ids, data2.consumer_ids)
    if length(unique(consumer_ids)) != length(consumer_ids) # redo consumer ids if duplicates
        consumer_ids = 1:length(consumer_ids)
    end

    return DataSD(
        consumer_ids,
        vcat(data1.product_ids, data2.product_ids),
        vcat(data1.product_characteristics, data2.product_characteristics),
        vcat(data1.positions, data2.positions),
        isnothing(data1.session_characteristics) ? nothing : vcat(data1.session_characteristics, data2.session_characteristics),
        vcat(data1.consideration_sets, data2.consideration_sets),
        vcat(data1.purchase_indices, data2.purchase_indices),
        isnothing(data1.min_discover_indices) ? nothing : vcat(data1.min_discover_indices, data2.min_discover_indices),
        isnothing(data1.search_paths) ? nothing : vcat(data1.search_paths, data2.search_paths),
        isnothing(data1.stop_indices) ? nothing : vcat(data1.stop_indices, data2.stop_indices)
    )
end

function eachindex(d::DataSD)
    return eachindex(d.product_ids)
end

function get_n_consumers(data::DataSD)
    return data.consumer_ids[end]
end

function sessions_with_clicks(d::DataSD)
    has_click = x -> x[1] > 0
    return findall(has_click, d.search_paths)
end

function sessions_with_purchase(d::DataSD)
    sp = Int64[]
    for i in eachindex(d)
        if d.product_ids[i][d.purchase_indices[i]] > 0 # if product_id > 0 -> not outside option /  purchase
            push!(sp, i)
        end
    end

    return sp
end

function drop_undiscovered_products!(d)
    for i in eachindex(d)
        stop_index = d.stop_indices[i]
        d.product_characteristics[i] = d.product_characteristics[i][1:stop_index, :]
        d.product_ids[i] = d.product_ids[i][1:stop_index]
    end
    return nothing
end


function construct_indices_characteristics(m::M, d::DataSD, var::Symbol) where M <: SDModel

    n_ses = length(d)
    s = m.information_structure

    # Expand to individual if only unit range or vector
    if var == :β
        indices_characteristics_β = if typeof(s.indices_characteristics_β_individual) <: UnitRange{Int} || typeof(s.indices_characteristics_β_individual) <: Vector{Int}
            fill(s.indices_characteristics_β_individual, n_ses)
        else
            s.indices_characteristics_β_individual
        end
        return indices_characteristics_β
    elseif var == :γ
        indices_characteristics_γ = if typeof(s.indices_characteristics_γ_individual) <: UnitRange{Int} ||
            typeof(s.indices_characteristics_γ_individual) <: Vector{Int}

            indices_characteristics_γ = fill(s.indices_characteristics_γ_individual, n_ses)
        else
            indices_characteristics_γ = s.indices_characteristics_γ_individual
        end
        return indices_characteristics_γ
    elseif var == :κ
        indices_characteristics_κ = if typeof(s.indices_characteristics_κ_individual) <: UnitRange{Int} || typeof(s.indices_characteristics_κ_individual) <: Vector{Int}
            indices_characteristics_κ = fill(s.indices_characteristics_κ_individual, n_ses)
        else
            indices_characteristics_κ = s.indices_characteristics_κ_individual
        end
        return indices_characteristics_κ
    else
        throw(ArgumentError("var must be either :β, :γ, or :κ."))
    end
end

function construct_mapping_characteristics(m::M, d::DataSD; kwargs...) where M <: SDModel

    mc = if haskey(kwargs, :mapping_characteristics)
        get(kwargs, :mapping_characteristics, nothing)
    else
        # If not provided, construct mapping from characteristics to indices
        indices_characteristics_β = construct_indices_characteristics(m, d, :β)
        indices_characteristics_γ = construct_indices_characteristics(m, d, :γ)
        indices_characteristics_κ = construct_indices_characteristics(m, d, :κ)

        mc = [indices_characteristics_β, indices_characteristics_γ, indices_characteristics_κ]

        types = [typeof(indices_characteristics_β),
                 typeof(indices_characteristics_γ),
                 typeof(indices_characteristics_κ)]

        if any(x -> x == Vector{Vector{Int64}}, types)
            # If any of the indices are vectors, convert to vector of vectors -> makes sure
            # that all types same for compiler
            mc = [collect.(x) for x in mc]
        end
        mc
    end

    return mc
end


# Data generation
function generate_data(m::SDCore, n_consumers, n_sessions_per_consumer;
        n_A0 = 1, n_d = 1,
        products = generate_products(n_consumers * n_sessions_per_consumer, MvNormal(I(length(m.β) - 1))),
        session_characteristics = nothing,
        drop_undiscovered_products = false,
        kwargs...)
    n_sessions = n_consumers * n_sessions_per_consumer

    # Unpack products
    product_ids, product_characteristics = deepcopy(products)
    #note: need copy as otherwise undiscovered products are also dropped from input

    # Create positions based on number of alternatives per position
    positions = [vcat(zeros(Int64, 1 + n_A0),
                     repeat(collect(Int64, 1:((length(product_ids[i]) - 1 - n_A0) / n_d)),
                         inner = n_d)) for i in 1:n_sessions]

    # Create consumer indices mapping consumers into sessions
    consumer_ids = repeat(1:n_consumers, inner = n_sessions_per_consumer)

    # Create empty consideration sets and purchase indices
    n_products = length.(product_ids)
    consideration_sets = [fill(false, n) for n in n_products]
    purchase_indices = zeros(Int, n_sessions)
    search_paths = [zeros(Int, n - 1) for n in n_products]
    stop_indices = [n for n in n_products]
    min_discover_indices = nothing

    # Put together as data object
    data = DataSD(consumer_ids, product_ids, product_characteristics, positions,
        session_characteristics,
        consideration_sets, purchase_indices, min_discover_indices, search_paths, stop_indices)

    # Generate search paths
    utility_purchases = zeros(Float64, length(data))
    generate_search_paths!(data, utility_purchases, m; kwargs...)
    fill_indices_min_discover!(data)

    if drop_undiscovered_products
        drop_undiscovered_products!(data)
    end

    # Return together with purchase utilities
    return data
end

"""
    update_positions!(data::DataSD, nA0, nd)
Update positions in `data` to fit the number of alternatives in initial awareness set `nA0` and the number of alternatives per position `nd`.
"""
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

    run_compatibility_checks(m, d)

    d1 = deepcopy(d)

    # Generate new paths
    utility_purchases = zeros(Float64, length(d1))
    generate_search_paths!(d1, utility_purchases, m; kwargs...)

    # Drop information initially not observed
    if isnothing(d.stop_indices)
        d1.stop_indices = nothing
    end
    if isnothing(d.search_paths)
        d1.search_paths = nothing
    end

    return_with_utilities = get(kwargs, :return_with_utilities, false)
    if return_with_utilities
        return d1, utility_purchases
    else
        return d1
    end
end

function generate_search_paths!(data::DataSD, utility_purchases, m::SDCore; kwargs...)

    # Get rng
    rng = get_rng(kwargs)

    # Extract keyword arguments
    conditional_on_search = get(kwargs, :conditional_on_search, false)
    conditional_on_search_iter = get(kwargs, :conditional_on_search_iter, 100)

    # Number of consumers and sessions
    n_products = length.(data.product_ids)
    max_products_per_session = maximum(n_products)
    n_sessions = length(data.product_ids)

    # Extract other values
    zdfun = get_functional_form(m.zdfun)
    zsfun = get_functional_form(m.zsfun)

    # Get draws from kwargs
    draws_shocks = get_precomputed_draws(kwargs)

    draws_u0, draws_e, draws_v, draws_w = draws_shocks

    # Draw consumer-level shocks
    draws_η = take_or_generate_consumer_shocks(m, data, kwargs)

    # get maximum pos at which to stop
    max_j = get(kwargs, :max_j, typemax(Int))

    # get indices of characteristics that enter search value if not provided
    mapping_characteristics = construct_mapping_characteristics(m, data; kwargs...)

    # Define chunks for parallelization. Each chunk is a range of sessions for which a single task
    # creates the search path.
    _, data_chunks = get_chunks(n_sessions)

    # Set up missing variables (will be fully overwritten)
    if isnothing(data.search_paths)
        data.search_paths = copy(data.consideration_sets)
    end
    if isnothing(data.stop_indices)
        data.stop_indices = copy(data.purchase_indices)
    end

    # note: unpacking here and passing into function saves a lot of allocations
    @unpack dU0, dE, dV, dW = m

    # Pre-compute discovery and search values for all possible positions (saves time when having more involved function)
    zd_h, zs_h = precompute_discovery_and_search_values(m, data)

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

                @views fill_path_i!(data, utility_purchases,
                    m, i, dU0, dE, dV, dW,
                    u, zs, v, zd_h, zs_h,
                    mapping_characteristics,
                    rng, max_j,
                    draws_u0, draws_e, draws_v, draws_w, draws_η)

                # If conditional on click, iterate until have at least one
                if conditional_on_search
                    iter = 1
                    while data.search_paths[i][1] == 0 && iter <= conditional_on_search_iter
                        # Reset
                        u .= typemin(Float64)
                        zs .= typemin(Float64)
                        v .= 0

                        fill_path_i!(data, utility_purchases,
                            m, i, dU0, dE, dV, dW,
                            u, zs, v, zd_h, zs_h,
                            mapping_characteristics,
                            rng, max_j,
                            draws_u0, draws_e, draws_v, draws_w, draws_η)
                        iter += 1
                    end
                    if iter > conditional_on_search_iter
                        @warn "Did not generate a click in $conditional_on_search_iter iterations for session $i."
                    end
                end
            end
        end
    end

    # Execute tasks
    fetch.(tasks)

    # Get last product that consumer MUST have discovered, i.e., all products on same position as the lowest one that was clicked on
    if get(kwargs, :compute_min_discover_indices, true)
        fill_indices_min_discover!(data)
    end


    return nothing
end

function precompute_discovery_and_search_values(m::SDCore,
        data::DataSD)

    zdfun = get_functional_form(m.zdfun)
    zsfun = get_functional_form(m.zsfun)
    positions_longest = data.positions[argmax(maximum.(data.positions))]
    all_possible_positions = sort(unique(positions_longest))
    zd_h = isnothing(m.zdfun) ?
        fill(m.Ξ, length(all_possible_positions)) :
        [zdfun(m.Ξ, m.ρ, pos) for pos in all_possible_positions]
    zs_h = isnothing(m.zsfun) ?
        fill(m.ξ, length(positions_longest)) :
        [zsfun(m.ξ, m.ξρ, j) for j in positions_longest]

    return zd_h, zs_h
end

function fill_path_i!(data::DataSD, utility_purchases,
        m, i, dU0, dE, dV, dW,
        u, zs, v, zd_h, zs_h,
        mapping_characteristics,
        rng, max_j,
        draws_u0, draws_e, draws_v, draws_w, draws_η;
        debug_print = false)

    # Extract data
    @unpack product_ids, consideration_sets, purchase_indices,
        positions, product_characteristics, search_paths, stop_indices = data

    # Reset
    search_paths[i] .= 0
    consideration_sets[i] .= false

    # Define variables tracking state during search
    max_u = typemin(eltype(u)) # current max utility
    max_zs = typemin(eltype(zs))  # current max search value
    ind_p = 0 # index of product to purchase
    ind_s = 0 # index of product to search
    pos = 0 # current position
    ns = 0 # number of searches
    ij = 0  # Index tracking current product
    n_prod = length(product_ids[i]) # number of products for consumer

    # Extract characteristics indices
    ixb = mapping_characteristics[1][i]
    ixg = mapping_characteristics[2][i]
    ixk = mapping_characteristics[3][i]

    @unpack β, Ξ, ρ, ξ, ξρ = m
    @unpack γ, κ = m.information_structure
    if has_heterogeneity(m)
        β, γ, κ, Ξ, ρ, ξ, ξρ = construct_individual_parameters(m, i, data, draws_η)
    end

    # Fill reservation values in initial awareness set
    for j in eachindex(positions[i])
        if positions[i][j] > 0 # only for initial awareness set, indicated with position = 0.
            # Update index of current product
            ij = j - 1 # last
            break
        end

        # Outside option
        if product_ids[i][j] == 0
            # draw outside option utility
            u[j] = take_or_generate_draw!(draws_u0, rng, dU0, 1) +
                   product_characteristics[i][1, end] * β[end]
            max_u = u[j]
            ind_p = 1
        else
            # Fill in reservation value for product j
            # note: storing v_j draw as also enters utility
            v[j] = take_or_generate_draw!(draws_v, rng, dV, j)
            xβ, xγ, _ = @views construct_util_parts(product_characteristics, i, j, β, γ, κ,
                ixb, ixg, ixk)

            zs[j] = xβ + xγ + zs_h[j] + v[j] +
                    take_or_generate_draw!(draws_w, rng, dW, j)

            # Update max search value and index
            if zs[j] > max_zs
                max_zs = zs[j]
                ind_s = j
            end
        end
    end

    # Update position and get discovery value to next one to be revealed
    pos += 1
    zd = zd_h[pos + 1]

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

        no_further_discoveries = pos > positions[i][end] || ij + 1 > max_j
        # discover more products
        if !no_further_discoveries && zd >= max_u && zd >= max_zs
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
                if positions[i][j] > pos || j + 1 > max_j
                    ij = j - 1 # current product is last in position
                    break
                end

                # Update reservation value and max
                v[j] = take_or_generate_draw!(draws_v, rng, dV, j)
                xβ, xγ, _ = @views construct_util_parts(product_characteristics, i, j, β, γ, κ,
                    ixb, ixg, ixk)

                zs[j] = xβ + xγ + zs_h[j] + v[j] +
                        take_or_generate_draw!(draws_w, rng, dW, j)
                if zs[j] > max_zs
                    max_zs = zs[j]
                    ind_s = j
                end
            end

            # Update discovery value and position
            pos += 1    # next position
            if pos + 1 <= length(zd_h)
                zd = zd_h[pos + 1]
            end

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
            search_paths[i][min(ns, end)] = ind_s
            consideration_sets[i][ind_s] = true

            # Set search value to neg. infinity so that it is not searched again
            zs[ind_s] = typemin(eltype(zs))

            # Get utility of searched product
            # note: recovering previously stored v_j draw as enters utility and search value
            u[ind_s] = v[ind_s]
            u[ind_s] += take_or_generate_draw!(draws_e, rng, dE, ind_s) # take new draw for e

            xβ, _, xκ = @views construct_util_parts(product_characteristics, i, ind_s, β, γ, κ,
                ixb, ixg, ixk)

            u[ind_s] += xβ + xκ

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
            purchase_indices[i] = ind_p
            utility_purchases[i] = max_u
            stop_indices[i] = if pos > positions[i][end] || ij + 1 > max_j # special case when discovered all positions, in which case ij was not updated
                length(positions[i])
            else
                ij
            end


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

@inline function construct_util_parts(product_characteristics, i, j, β, γ, κ, ixb, ixg, ixk; no_κ = false)

    xβ = if isempty(ixb)
        zero(eltype(β))
    else
        # @views product_characteristics[i][j, ixb]' * β[ixb]
        @inbounds sum(product_characteristics[i][j, k] * β[k] for k in ixb)
    end
    T = typeof(xβ)
    xγ = if isempty(ixg)
        zero(T)
    else
        # @views product_characteristics[i][j, ixg]' * γ[ixg]
        @inbounds sum(product_characteristics[i][j, k] * γ[k] for k in ixg)
    end
    xκ = if isempty(ixk) || no_κ
        zero(T)
    else
        # @views product_characteristics[i][j, ixk]' * κ[ixk]
        @inbounds sum(product_characteristics[i][j, k] * κ[k] for k in ixk)
    end

    return xβ::T, xγ::T, xκ::T
end

function get_indices_min_discover(d::DataSD)
    @unpack consideration_sets, positions, stop_indices = d

    index_click_lowest_position = Array{Union{Int, Nothing}, 1}([findlast(C)
                                                                 for C in consideration_sets])
    index_click_lowest_position[isnothing.(index_click_lowest_position)] .= 1 # set to one for those who did not click


    indices_min_discover = if isnothing(stop_indices)
        [findlast((positions[i] .==
                                positions[i][index_click_lowest_position[i]]))
                    for i in eachindex(index_click_lowest_position)]

    else
        [findlast((positions[i][1:stop_indices[i]] .==
                                     positions[i][index_click_lowest_position[i]]))
                            for i in eachindex(index_click_lowest_position)]
    end

    return indices_min_discover
end

# Cost computations
"""
	calculate_costs!(model::SDModel, data::DataSD, n_draws...; force_recompute = true, kwargs...)

Calculate search and discovery costs for the `model <: SDModel` using `data <: DataSD` and add/update them in the `model`. Uses `n_draws` to calculate the different costs using the distribution of characteristics in the data. If `n_draws` is a single element, the same number of draws are used to compute `cs` and `cd`. Otherwise, the first element is used to compute `cs` and the second to compute `cd`. If `force_recompute` is true (default), the costs are recomputed even if they are already present in the model.

Discovery costs are given by E[max{0, xβγ_demeaned + ε - Ξ}] and depend on the distribution of characteristics in the data as well as the distribution of ε. The expectation is computed by sampling from the distribution of characteristics in the data and the distribution of ε. Discovery costs are thus computed under the assumption that  beliefs are correct across positions.

Search costs are given by E[max{0, xβ_detail_demeaned + ε - ξj}], where ξj is the position (or product) specific search value. The expectation is computed by sampling from the distribution of characteristics in the data and the distribution of ε. If all characteristics are revealed on the list, then the search cost is instead computed as E[1 - F(ξj)], where F is the distribution of ε. See (7) in Greminger (2022) and online appendix EC.2.
"""
function calculate_costs!(m::SDCore, d::DataSD, n_draws...;
        force_recompute = true,
        kwargs...)

    n_draws_cs = n_draws[1]
    n_draws_cd = length(n_draws) == 1 ? n_draws_cs : n_draws[2] # if only one element, use same number of draws for discovery costs

    # Search costs
    if isnothing(m.cs) || force_recompute
        m.cs = calculate_search_costs(m, d, n_draws_cs; kwargs...)
    end
    # Discovery costs
    if isnothing(m.cd) || force_recompute
        m.cd = calculate_discovery_costs(m, d, n_draws_cd; kwargs...)
    end
    return nothing
end

function calculate_search_costs(m::SDModel, d::DataSD, n_draws; kwargs...)

    if has_heterogeneity(m)
        throw(ArgumentError("Heterogeneity not yet supported for search costs."))
    end

    # Unpack
    @unpack ξ, ξρ, dE = m
    zsfun = get_functional_form(m.zsfun)

    # Compute ξj for every product in data
    product_ids = vcat(d.product_ids...)
    n = length(product_ids)
    positions = vcat(d.positions...)
    ξj = zeros(n)

    all_possible_positions = sort(unique(positions)) # all possible positions across all sessions
    if minimum(all_possible_positions) != 0
        throw(ArgumentError("Positions in `d` should start at 0."))
    end
    ξh = [zsfun(ξ, ξρ, pos) for pos in all_possible_positions]

    Threads.@threads for i in eachindex(ξj)
        if product_ids[i] == 0 # outside option
            ξj[i] = Inf
        else
            ξj[i] = ξh[positions[i]+1] # +1 because positions start at 0
        end
    end

    # Compute costs for every unique combination of ξj and xκ specification
    cs = zeros(n)
    rng = get_rng(kwargs)

    lookup_i_xκ, xκ = get_xv_spec(m, d, :κ; kwargs...) # get different specifications of where characteristics are revealed

    # Create unique combinations of (ξj, xκ specification index)
    # Only for products that are not outside option (ξj < Inf)
    valid_indices = findall(ξj .< Inf)
    ξj_xκ_combinations = unique([(ξj[i], lookup_i_xκ[i]) for i in valid_indices])

    Threads.@threads for (ξ, xκ_idx) in ξj_xκ_combinations

        # Find all products with this combination of ξj and xκ specification
        i_ξj_xκ = findall((ξj .== ξ) .& (lookup_i_xκ .== xκ_idx))

        # Calculate search cost for this combination
        cs_ξj_xκ = @views _calculate_search_costs(m, ξ, dE, xκ[:, xκ_idx], n_draws, rng)

        # Assign to all products with this combination
        cs[i_ξj_xκ] .= cs_ξj_xκ
    end

    # Store in vector of vectors, matching the structure of d
    r = create_index_row_to_session(d) # create mapping from session to row in data
    cs_vec = [zeros(Float64, length(d.product_ids[i])) for i in eachindex(d)]
    for i in eachindex(d)
        cs_vec[i] .= cs[r .== i]
    end

    return cs_vec
end

function get_xγ(m, d) # get xβγ for all products

    chars = vcat(d.product_characteristics...)

    ig = m.information_structure.indices_characteristics_γ_individual

    xγ = zeros(size(chars, 1)) # pre-allocate xβγ for all products
    Threads.@threads for i in eachindex(xγ)
        xγ[i] = @views chars[i, ig]' * m.information_structure.γ[ig]
    end

    return xγ
end


function get_xv_spec(m::SDModel, d::DataSD, var::Symbol; kwargs...)

    chars = vcat(d.product_characteristics...) # all characteristics across all sessions
    chars = chars[vcat(d.product_ids...) .> 0, :] # exclude outside option

    # unique speicifications of where characteristics are revealed
    ixvar = StructuralSearchModels.construct_indices_characteristics(m, d, var)

    # Get all combinations of the two
    unique_ix = unique(ixvar)
    xvar = zeros(size(chars, 1), length(unique_ix)) # pre-allocate xβκ_spec for all products

    V = if var == :κ
        m.information_structure.κ
    elseif var == :γ
        m.information_structure.γ
    elseif var == :β
        m.β
    else
        throw(ArgumentError("var must be either :β, :γ, or :κ."))
    end

    Threads.@threads for i in eachindex(unique_ix)
        uix = unique_ix[i] # unique index for this specification
        n_char_k = length(uix) # number of characteristics in this specification
        xvar[:, i] = if n_char_k > 0
            @views chars[:, uix] * V[uix]
        else
            zeros(size(chars, 1)) # no characteristics in this specification
        end
    end

    # Create lookup table to map from row k in session i to xβκ
    r = create_index_row_to_session(d) # create mapping from session to row in data

    lookup_i_xvar = zeros(Int, length(r)) # pre-allocate lookup table

    Threads.@threads for i_ses in eachindex(d)
        lookup_i_xvar[r .== i_ses] .= findfirst([ixvar[i_ses] == unique_ix[i] for i in eachindex(unique_ix)])
        # println(findfirst([ixκ[i_ses] == unique_ixκ[i] for i in eachindex(unique_ixκ)])) # number of characteristics in this session
    end

    return lookup_i_xvar, xvar
end

function create_index_row_to_session(d::DataSD)
    p = vcat(d.product_ids...)
    n = length(p)
    i_ses = zeros(Int, n) # all product ids across all sessions

    c = 0
    for i in eachindex(i_ses) # note: assumes new session starts with pid=0 for the outside option
        if p[i] > 0
            i_ses[i] = c
        else
            c += 1
            i_ses[i] = c # outside option
        end
    end

    @assert c > 1 "There should be an outside option with pid=0 in the data."
    return i_ses
end


function _calculate_search_costs(m::SDCore, ξj::T, dE::Distribution, xκ, n_draws, rng) where T <: Real

    cs = all_characteristics_on_list(m) ?
        calculate_search_cost_all_characteristics_on_list(ξj, dE) :
        calculate_search_cost_hidden_characteristics(ξj, dE, xκ, n_draws, rng)

    return cs
end

function calculate_search_cost_hidden_characteristics(ξ, dE, xκ, n_draws, rng)

    # Calculates search costs based on cs = E[max(0, xκ -mean(xκ) ε - ξ) | x_list].
    # - Expectation E[] is conditional on x_list reflecting consumers' beliefs.
    # - assumption is that conditional on x_list, xκ is independent of ε and has mean zero.
    #   hence, we demean xκ before calculating costs.
    xκ_demeaned = xκ .- mean(xκ)

    cs = mean(max(0, rand(rng, xκ_demeaned) + rand(rng, dE) - ξ) for i in 1:n_draws)
    return cs
end

function calculate_search_cost_all_characteristics_on_list(ξ::T, dE::Distribution) where T
    # If all characteristics are revealed on the list, then we only need to integrate over
    # the distribution of ε, given by m.dE. See (7) in the theory paper and online appendix EC.2.
    return quadgk(e -> (1 - cdf(dE, e)), ξ, maximum(dE))[1]
end


# Previous code for heterogeneous

# n_sessions = length(d)
# cs = zeros(n_sessions)

# _, data_chunks = get_chunks(n_sessions)

# n_cons = length(unique(d.consumer_ids))
# draws_η = take_or_generate_consumer_shocks(m, d, kwargs)

# tasks = map(data_chunks) do chunk
#     Threads.@spawn begin

#         local chars, _, xβ_detail = get_xβ(m, d)

#         for i in chunk
#             β, _, _, ξ, _ = construct_individual_parameters(m, i, d, draws_η)

#             update_xβ_detail!(xβ_detail, chars, β, d, i)

#             cs[i] = mean(max(0, rand(rng, xβ_detail) + rand(rng, dE) - ξ) for i in 1:n_draws)
#         end
#     end
# end

# fetch.(tasks)

# function get_xβ(m::SDModel, d::DataSD; position = nothing)

#     # characteristics matrix without outside option
#     chars = if isnothing(position) # any position
#         vcat([d.product_characteristics[i][d.product_ids[i] .> 0, :]
#                   for i in eachindex(d)]...) # excludes outside option
#     else # only for specific position
#         get_chars_position(d, position)
#     end

#     indices_list_characteristics = d.indices_list_characteristics[1]
#     # will be updated later on when some consumers have different list characteristics

#     xβ_list = if length(indices_list_characteristics) > 0
#         @views chars[:, indices_list_characteristics] * m.β[indices_list_characteristics]
#     else
#         nothing
#     end

#     all_characteristics_on_list = length(indices_list_characteristics) == length(m.β)
#     xβ_detail = if all_characteristics_on_list
#         nothing
#     else
#         indices_characteristics_on_detail_page = setdiff(1:length(m.β), indices_list_characteristics)
#         @views chars[:, indices_characteristics_on_detail_page] * m.β[indices_characteristics_on_detail_page]
#     end

#     return chars, xβ_list, xβ_detail
# end

# function update_xβ_detail!(xβ_detail, chars, β, d, i)

#         indices_list_characteristics = d.indices_list_characteristics[i]
#         indices_characteristics_on_detail_page = setdiff(1:length(β), indices_list_characteristics)
#         xβ_detail .= @views chars[:, indices_characteristics_on_detail_page] * β[indices_characteristics_on_detail_page]
# end

# function update_xβ_list_and_detail!(xβ_list, xβ_detail, chars, β, d, i)

#     indices_list_characteristics = d.indices_list_characteristics[i]

#     if !isnothing(xβ_list)
#         xβ_list = @views chars[:, indices_list_characteristics] * β[indices_list_characteristics]
#     end

#     if !isnothing(xβ_detail)
#         indices_characteristics_on_detail_page = setdiff(1:length(β), indices_list_characteristics)
#         xβ_detail .= @views chars[:, indices_characteristics_on_detail_page] * β[indices_characteristics_on_detail_page]
#     end

# end


# function get_chars_position(d, position)

#    return vcat(
#     [d.product_characteristics[i][d.product_ids[i] .> 0 .&&
#                 d.positions[i][1:length(d.product_ids[i])] .== position, :]
#                 for i in eachindex(d)]...
#    )

# end

# """
# 	calculate_position_specific_search_costs(m::SDModel, d::DataSD)

# Calculate position-specific search costs for the SD model given `ξ`, `ξρ`, the distribution of ε `dE`, and the distribution of xβ_detail from data. See `calculate_search_cost` for details.
# """
# function calculate_position_specific_search_costs(m::SDModel, d::DataSD, n_draws; kwargs...)

#     # Get maximum positions possible
#     _, i_max_products = findmax(length.(d.product_ids))
#     positions_max = d.positions[i_max_products]

#     all_characteristics_on_list = length(d.indices_list_characteristics[1]) == length(m.β)

#     n_rows = length(vcat(d.product_ids...))

#     # same costs all positions -> just fill in cs for all positions
#     if isnothing(m.γ) && m.zsfun == ""
#         cs = calculate_search_cost(m, m.ξ, d, n_draws; kwargs...)
#         return fill(cs, n_rows)
#     end

#     # Extract zsfun
#     zsfun = get_functional_form(m.zsfun)

#     # Same cost per position across consumers and products
#     if isnothing(m.γ) && (!has_heterogeneity(m) ||
#         m.heterogeneity._observed_hs.ξ ==
#         m.heterogeneity._observed_hs.β ==
#         m.heterogeneity._observed_hs.ξρ ==
#         m.heterogeneity._unobserved_hs.ξ ==
#         m.heterogeneity._unobserved_hs.β ==
#         m.heterogeneity._unobserved_hs.ξρ == false ||
#         (all_characteristics_on_list &&
#         m.heterogeneity._observed_hs.ξ ==
#         m.heterogeneity._observed_hs.ξρ ==
#         m.heterogeneity._unobserved_hs.ξ ==
#         m.heterogeneity._unobserved_hs.ξρ == false))

#         @unpack ξ, ξρ, dE = m
#         ξ_j = [zsfun(ξ, ξρ, positions_max[j]) for j in eachindex(positions_max)]

#         # If all characteristics are revealed on the list, then simpler method available
#         # NOTE: will need to update if not all consumers have same number of characteristics on list
#         cs_h = zeros(length(ξ_j))
#         rng = get_rng(kwargs)

#         for j in eachindex(ξ_j)
#             if j > 1 && positions_max[j] == positions_max[j - 1]
#                 cs_h[j] = cs_h[j - 1] # same position as previous one
#                 continue
#             end

#             # Get distribution of xβ_detail for this position
#             _, _, xβ_detail = get_xβ(m, d; position = positions_max[j])

#             # Calculate search cost for this position
#             cs_h[j] =  mean(max(0, rand(rng, xβ_detail) + rand(rng, dE) - ξ_j[j]) for i in 1:n_draws)
#         end

#         positions_rows = vcat(d.positions...)
#         cs = zeros(eltype(cs_h), n_rows)
#         for j in eachindex(positions_max)
#             cs[positions_rows .== positions_max[j]] = cs_h[j]
#         end
#         return cs

#     elseif !isnothing(m.γ) && (!has_heterogeneity(m) ||
#         m.heterogeneity._observed_hs.ξ ==
#         m.heterogeneity._observed_hs.β ==
#         m.heterogeneity._observed_hs.ξρ ==
#         m.heterogeneity._unobserved_hs.ξ ==
#         m.heterogeneity._unobserved_hs.β ==
#         m.heterogeneity._unobserved_hs.ξρ == false ||
#         (all_characteristics_on_list &&
#         m.heterogeneity._observed_hs.ξ ==
#         m.heterogeneity._observed_hs.ξρ ==
#         m.heterogeneity._unobserved_hs.ξ ==
#         m.heterogeneity._unobserved_hs.ξρ == false))

#         @unpack ξ, ξρ, dE = m

#         # Get xb XXX

#         # construct ξ_j
#         ξ_j = xγ + [zsfun(ξ, ξρ, pos) for pos in vcat(d.positions...)]

#         # If all characteristics are revealed on the list, then simpler method available
#         # NOTE: will need to update if not all consumers have same number of characteristics on list
#         cs_h = zeros(length(ξ_j))
#         rng = get_rng(kwargs)

#         for j in eachindex(ξ_j)
#             if j > 1 && positions_max[j] == positions_max[j - 1]
#                 cs_h[j] = cs_h[j - 1] # same position as previous one
#                 continue
#             end

#             # Get distribution of xβ_detail for this position
#             _, _, xβ_detail = get_xβ(m, d; position = positions_max[j])

#             # Calculate search cost for this position
#             cs_h[j] =  mean(max(0, rand(rng, xβ_detail) + rand(rng, dE) - ξ_j[j]) for i in 1:n_draws)
#         end


#     else isnothing(m.γ) # Same cost across products but different across consumers
#         cs_h = [zeros(length(positions_max)) for i in eachindex(d)]
#         rng = get_rng(kwargs)

#         draws_η = take_or_generate_consumer_shocks(m, d, kwargs)

#         # Parallelize across positions
#         _, data_chunks = get_chunks(length(cs_h[1]))

#         tasks = map(data_chunks) do chunk
#             Threads.@spawn begin
#                 local xβ_detail = zeros(Float64, length(positions_max))

#                 for h in chunk
#                     chars = get_chars_position(d, h)

#                     # Loop over consumers and fill in costs
#                     for i in eachindex(d)
#                         @unpack dE, cs = m
#                         if positions_max[h] == 0 # save time by just filling in baseline cost
#                             cs_h[i][h] = typeof(cs) <: Real ? cs : cs[i]
#                             continue
#                         end
#                         # Get individual-specific parameters
#                         β, _, _, ξ, ξρ = construct_individual_parameters(m, i, d, draws_η)
#                         ξ_h = zsfun(ξ, ξρ, positions_max[h])

#                         # If all characteristics are revealed on the list, then simpler method available
#                         all_characteristics_on_list = length(d.indices_list_characteristics[i]) == length(m.β)
#                         if all_characteristics_on_list
#                             cs_h[i][h] = quadgk(e -> (1 - cdf(dE, e)), ξ_h, maximum(dE))[1]
#                         else
#                             update_xβ_detail!(xβ_detail, chars, β, d, i)
#                             cs_h[i][h] = mean(max(0, rand(rng, xβ_detail) + rand(rng, dE) - ξ_h) for i in 1:n_draws)
#                         end
#                     end
#                 end
#             end
#         end

#         fetch.(tasks)
#         return cs_h
#     end
# end

function calculate_discovery_costs(m::SDModel, d::DataSD, n_draws;
        kwargs...)

    if has_heterogeneity(m)
        throw(ArgumentError("Heterogeneity not yet supported for discovery costs."))
    end


    # Extract common values
    zdfun = get_functional_form(m.zdfun)
    rng = get_rng(kwargs)

    # Get xκ, xγ, xβ for all specifications in data, with values in rows and specs in cols
    # Also gets lookup tables to map from row in data to spec col in xβ, xγ, xκ
    lookup_i_xκ, xκ = get_xv_spec(m, d, :κ; kwargs...)
    lookup_i_xγ, xγ = get_xv_spec(m, d, :γ; kwargs...)
    lookup_i_xβ, xβ = get_xv_spec(m, d, :β; kwargs...)

    # Get all possible unique combintaions of specifications from lookup tables
    unique_combinations = unique(hcat(lookup_i_xβ, lookup_i_xγ, lookup_i_xκ), dims=1)

    # Compute costs for each unique specification combination
    n_spec = size(unique_combinations, 1)
    cd_for_each_spec = zeros(n_spec)
    max_pos = maximum(vcat(d.positions...))

    W = zeros(Float64, n_draws)
    for i in axes(unique_combinations, 1)
        # which column in xβ, xγ, xκ to use
        ixβ = unique_combinations[i, 1]
        ixγ = unique_combinations[i, 2]
        ixκ = unique_combinations[i, 3]
        xβ_i = size(xβ, 2) > 1 ? xβ[:, ixβ] : xβ[:]
        xγ_i = size(xγ, 2) > 1 ? xγ[:, ixγ] : xγ[:]
        xκ_i = size(xκ, 2) > 1 ? xκ[:, ixκ] : xκ[:]

        fill_values_cd_compute!(W, rng, m, m.ξ, xβ_i, xγ_i, xκ_i, max_pos)

        cd_for_each_spec[i] = mean(W)
    end

    # Expand cd_for_each_spec to all sessions (not just unique specs)
    cd = zeros(length(lookup_i_xκ))
    for i in axes(unique_combinations, 1)
        ixβ = unique_combinations[i, 1]
        ixγ = unique_combinations[i, 2]
        ixκ = unique_combinations[i, 3]
        rows_to_fill = findall(
            (lookup_i_xβ .== ixβ) .& (lookup_i_xγ .== ixγ) .& (lookup_i_xκ .== ixκ)
        )

        cd[rows_to_fill] .= cd_for_each_spec[i]
    end

    # Reduce to one discovery cost per session (instead of per product in data.product_characteristics)
    r = create_index_row_to_session(d)
    i_to_keep = vcat(1, 1 .+ findall(diff(r) .> 0))

    cd = cd[i_to_keep]

    return cd
end

function fill_values_cd_compute!(W, rng, m, ξ::T, xβ, xγ, xκ, max_pos) where T <: Real

    # Take draws for effective values in parallel
    _, data_chunks = get_chunks(length(W))

    # Demean xκ
    xκ_demeaned = xκ .- mean(xκ)

    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            for i in chunk
                # Draws for a product
                e = rand(rng, m.dE)
                v = rand(rng, m.dV)
                w = rand(rng, m.dW)
                xβi = rand(rng, xβ)
                xγi = rand(rng, xγ)
                xκi = rand(rng, xκ_demeaned)

                # Construct effective value of product i
                W[i] = xβi + xγi + v + min(ξ + w, xκi + e)
                # note: in line with how search costs are computed, i.e., what's revealed on
                # the detail page is mean zero (so demeaned xκ)
                # have E[xβi + xγi + v + min(ξ + w, xκi + e) ] = E[xβi + xγi + v] + E[min(ξ + w, xκi + e) | xγ ]
            end
        end
    end

    # Execute tasks
    fetch.(tasks)

    # Compute mean μ
    μ = mean(W)

    # Recover Ξ and μ1
    μ1, Ξ = calculate_μ1_Ξ(m, μ, max_pos)

    # For expectation, update effective  value to max{0, wi - μ - Ξ}
    Threads.@threads for i in eachindex(W)
        W[i] = max(0, W[i] - μ - Ξ)
    end

    return nothing
end
function calculate_maximum_position(d)
    # Get positions without outside option
    positions = 0:maximum(vcat(d.positions...))
    return round(Int, mean(positions[positions .> 0]))
end

function calculate_μ1_Ξ(m::SDCore, μ::T, max_pos) where T <: Real
    @unpack dW, dV = m

    zdfun = get_functional_form(m.zdfun)

    μ1 = μ - (sum(zdfun(m.Ξ, m.ρ, pos) for pos in 0:max_pos) / (max_pos + 1) - m.Ξ)
    # note: zdfun computes Ξ + some function of ρ, so need to subtract Ξ when using zdfun to compute μ(h)

    # Now get Ξ
    Ξ1 = m.Ξ - μ1

    return μ1, Ξ1
end

# Reservation values / inverse calculations for costs
"""
	calculate_discovery_value(G::Normal, m::SDModel, ξ::T, cs::T, cd::T)
Calculate the discovery value `zd` given `cs`, `cd` and `ξ`. Assumes that pre-search values xβ + v + w follow normal distribution `G`.
"""
function calculate_discovery_value(G::Normal, m::SDModel, ξ::T, cs::T, cd::T) where T <: Real

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

"""
    calculate_ξ(cs, F)
Calculate the search value `ξ` given cost `cs` and distribution `F`.
"""
function calculate_ξ(cs, F)
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

"""
	zs_inner_integral(ξ,F)
Returns ∫_ξ (1-F(ϵ))dϵ.
"""
function zs_inner_integral(ξ,F)
	quadgk(e->(1-cdf(F,e)),ξ,maximum(F))[1]
end


# Consumer welfare
function calculate_welfare(m::SDCore, data::DataSD, n_sim;
        kwargs...)

    # Input checking
    if isnothing(m.cs) || isnothing(m.cd)
        throw(ArgumentError("Search and discovery costs not calculated. Run `calculate_costs!` first."))
    end

    if var(m.dW) > 0
        throw(ArgumentError("Computing welfare using path simulations is not yet implemented for search cost shocks."))
    end

    run_compatibility_checks(m, data)
    return calculate_welfare_simpaths(m, data, n_sim; kwargs...)
end


function calculate_welfare_simpaths(m::SDCore, d::DataSD, n_draws; kwargs...)

    rng = StructuralSearchModels.get_rng(kwargs)
    seed = get(kwargs, :seed, 0)

    n_sessions = length(d)

    welfare_avg = zeros(Float64, n_sessions)
    utility_choice_avg = zeros(Float64, n_sessions)
    search_costs_avg = zeros(Float64, n_sessions)
    discovery_costs_avg = zeros(Float64, n_sessions)

    # Average conditional on click
    welfare_conditional_on_search = zeros(Float64, n_sessions)
    utility_choice_conditional_on_search = zeros(Float64, n_sessions)
    search_costs_conditional_on_search = zeros(Float64, n_sessions)
    discovery_costs_conditional_on_search = zeros(Float64, n_sessions)

    # Average conditional on purchase
    welfare_conditional_on_purchase = zeros(Float64, n_sessions)
    utility_choice_conditional_on_purchase = zeros(Float64, n_sessions)
    search_costs_conditional_on_purchase = zeros(Float64, n_sessions)
    discovery_costs_conditional_on_purchase = zeros(Float64, n_sessions)

    vectors_to_fill = [
        welfare_avg, utility_choice_avg, search_costs_avg, discovery_costs_avg,
        welfare_conditional_on_search, utility_choice_conditional_on_search,
        search_costs_conditional_on_search, discovery_costs_conditional_on_search,
        welfare_conditional_on_purchase, utility_choice_conditional_on_purchase,
        search_costs_conditional_on_purchase, discovery_costs_conditional_on_purchase
    ]

    utility_purchases = zeros(Float64, n_sessions)
    dU0, dE, dV, dW = m.dU0, m.dE, m.dV, m.dW

    zd_h, zs_h = precompute_discovery_and_search_values(m, d)
    mapping_characteristics = StructuralSearchModels.construct_mapping_characteristics(m, d)

    # Iterate over sessions and fill in welfare measures
    d_sim = deepcopy(d)
    d_sim.search_paths = [zeros(Int, length(d_sim.product_ids[i])) for i in eachindex(d_sim)]
    d_sim.stop_indices = [0 for i in eachindex(d_sim.product_ids)]

    max_n_products = get(kwargs, :max_n_products, maximum(length.(d.product_ids)))
    full_draws_for_fixed_seed = get(kwargs, :full_draws_for_fixed_seed, false)

    _, data_chunks = StructuralSearchModels.get_chunks(n_sessions)
    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            # Create local variables that the thread can work with. Pre-allocation circumvenst allocations in the loop.
            local u = zeros(Float64, max_n_products)
            local zs = zeros(Float64, max_n_products)
            local v = zeros(Float64, max_n_products)
            local draws_u0, draws_e, draws_v, draws_w, draws_η =
                get_empty_vectors_for_shocks(full_draws_for_fixed_seed, max_n_products)

            for i in chunk

                welfare_i = calculate_welfare_i(d_sim, utility_purchases, m, i, n_draws,
                    dU0, dE, dV, dW,
                    u, zs, v, zd_h, zs_h,
                    mapping_characteristics,
                    rng, seed,
                    max_n_products,
                    draws_u0, draws_e, draws_v, draws_w, draws_η,
                    full_draws_for_fixed_seed)

                # Fill paid costs
                fill_welfare_measures!(vectors_to_fill, welfare_i, i)

            end
        end
    end

    fetch.(tasks) # wait for all tasks to finish

    welfare_avg = utility_choice_avg - search_costs_avg - discovery_costs_avg
    welfare_conditional_on_search = utility_choice_conditional_on_search -
                                    search_costs_conditional_on_search -
                                    discovery_costs_conditional_on_search
    welfare_conditional_on_purchase = utility_choice_conditional_on_purchase -
                                      search_costs_conditional_on_purchase -
                                      discovery_costs_conditional_on_purchase

    # drop nans due to no purchases or searches
    welfare_conditional_on_search =
        welfare_conditional_on_search[isnan.(welfare_conditional_on_search).==false]
    utility_choice_conditional_on_search =
        utility_choice_conditional_on_search[isnan.(utility_choice_conditional_on_search).==false]
    search_costs_conditional_on_search =
        search_costs_conditional_on_search[isnan.(search_costs_conditional_on_search).==false]
    discovery_costs_conditional_on_search =
        discovery_costs_conditional_on_search[isnan.(discovery_costs_conditional_on_search).==false]
    welfare_conditional_on_purchase =
        welfare_conditional_on_purchase[isnan.(welfare_conditional_on_purchase).==false]
    utility_choice_conditional_on_purchase =
        utility_choice_conditional_on_purchase[isnan.(utility_choice_conditional_on_purchase).==false]
    search_costs_conditional_on_purchase =
        search_costs_conditional_on_purchase[isnan.(search_costs_conditional_on_purchase).==false]
    discovery_costs_conditional_on_purchase =
        discovery_costs_conditional_on_purchase[isnan.(discovery_costs_conditional_on_purchase).==false]


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

function fill_welfare_measures!(vectors_to_fill, welfare_i, i)
    welfare_avg, u_avg, cs_avg, cd_avg,
    welfare_conditional_on_search, u_conditional_on_search,
    cs_conditional_on_search, cd_conditional_on_search,
    welfare_conditional_on_purchase, u_conditional_on_purchase,
    cs_conditional_on_purchase, cd_conditional_on_purchase = vectors_to_fill

    (welfare_i_avg, u_i_avg, cs_i_avg, cd_i_avg),
    (welfare_i_conditional_on_search, u_i_conditional_on_search,
     cs_i_conditional_on_search, cd_i_conditional_on_search),
    (welfare_i_conditional_on_purchase, u_i_conditional_on_purchase,
     cs_i_conditional_on_purchase, cd_i_conditional_on_purchase), _ = welfare_i

    welfare_avg[i] = welfare_i_avg
    u_avg[i] = u_i_avg
    cs_avg[i] = cs_i_avg
    cd_avg[i] = cd_i_avg

    welfare_conditional_on_search[i] = welfare_i_conditional_on_search
    u_conditional_on_search[i] = u_i_conditional_on_search
    cs_conditional_on_search[i] = cs_i_conditional_on_search
    cd_conditional_on_search[i] = cd_i_conditional_on_search

    welfare_conditional_on_purchase[i] = welfare_i_conditional_on_purchase
    u_conditional_on_purchase[i] = u_i_conditional_on_purchase
    cs_conditional_on_purchase[i] = cs_i_conditional_on_purchase
    cd_conditional_on_purchase[i] = cd_i_conditional_on_purchase

    return nothing

end



function calculate_welfare_i(d, utility_purchases, m, i, n_draws,
    dU0, dE, dV, dW,
    u, zs, v, zd_h, zs_h,
    mapping_characteristics,
    rng, seed, max_j,
    draws_u0, draws_e, draws_v, draws_w, draws_η,
    full_draws_for_fixed_seed)

    u_i = 0.0
    cs_i = 0.0
    cd_i = 0.0

    u_i_conditional_on_search = 0.0
    cs_i_conditional_on_search = 0.0
    cd_i_conditional_on_search = 0.0
    n_sim_with_search = 0

    u_i_conditional_on_purchase = 0.0
    cs_i_conditional_on_purchase = 0.0
    cd_i_conditional_on_purchase = 0.0
    n_sim_with_purchase = 0

    n_prod = length(d.product_ids[i])

    if full_draws_for_fixed_seed
        Random.seed!(rng, seed + i) # set seed for each consumer
    end

    @inbounds for dd in 1:n_draws

        # Draw shocks if full evaluation. Otherwise drawn within
        # fill_path_i!, which does not draw for all products
        if full_draws_for_fixed_seed
            draws_u0[1] = rand(rng, dU0)
            for j in eachindex(draws_e)
                if j == 1 # skip taking draw for outside option
                    continue
                end
                draws_e[j] = rand(rng, dE) # draw e
                draws_v[j] = rand(rng, dV) # draw v
                draws_w[j] = rand(rng, dW) # draw w
            end
        end

        # Reset
        u .= typemin(Float64)
        zs .= typemin(Float64)
        v .= 0

        @views fill_path_i!(d, utility_purchases,
            m, i, dU0, dE, dV, dW,
            u, zs, v, zd_h, zs_h,
            mapping_characteristics,
            rng, max_j,
            draws_u0, draws_e, draws_v, draws_w, draws_η)

        u_sim = utility_purchases[i]

        last_search = if d.search_paths[i][1] == 0 # no search
            0
        elseif d.search_paths[i][n_prod] != 0 # last search is 0
            n_prod
        else
            findfirst(x -> x ==0, d.search_paths[i]) - 1 # last search before 0
        end

        cs_sim = if last_search == 0 # no searches
            0.0
            else
                @views sum(m.cs[i][s] for s in d.search_paths[i][1:last_search])  # loop over searches and sum up costs
            end

        cd_sim = d.positions[i][d.stop_indices[i]] * m.cd[i]

        u_i += u_sim
        cs_i += cs_sim
        cd_i += cd_sim

        if d.search_paths[i][1] > 0
            n_sim_with_search += 1
            u_i_conditional_on_search += u_sim
            cs_i_conditional_on_search += cs_sim
            cd_i_conditional_on_search += cd_sim
        end
        if d.purchase_indices[i] > 1
            n_sim_with_purchase += 1
            u_i_conditional_on_purchase += u_sim
            cs_i_conditional_on_purchase += cs_sim
            cd_i_conditional_on_purchase += cd_sim
        end

    end

    w_i = u_i - cs_i - cd_i # welfare for consumer i
    w_i_conditional_on_search = u_i_conditional_on_search - cs_i_conditional_on_search - cd_i_conditional_on_search
    w_i_conditional_on_purchase = u_i_conditional_on_purchase - cs_i_conditional_on_purchase - cd_i_conditional_on_purchase


    return (w_i /n_draws, u_i / n_draws, cs_i / n_draws, cd_i / n_draws),
            (w_i_conditional_on_search / n_sim_with_search,
             u_i_conditional_on_search / n_sim_with_search,
             cs_i_conditional_on_search / n_sim_with_search,
             cd_i_conditional_on_search / n_sim_with_search),
            (w_i_conditional_on_purchase / n_sim_with_purchase,
             u_i_conditional_on_purchase / n_sim_with_purchase,
             cs_i_conditional_on_purchase / n_sim_with_purchase,
                cd_i_conditional_on_purchase / n_sim_with_purchase),
                (n_sim_with_purchase / n_draws, )

end


function get_empty_vectors_for_shocks(full_draws_for_fixed_seed, max_n_products)

    draws_u0, draws_e, draws_v, draws_w, draws_η = nothing, nothing, nothing, nothing, nothing

    if full_draws_for_fixed_seed
        draws_u0 = zeros(Float64, 1)
        draws_e = zeros(Float64, max_n_products)
        draws_v = zeros(Float64, max_n_products)
        draws_w = zeros(Float64, max_n_products)
    end

    return draws_u0, draws_e, draws_v, draws_w, draws_η
end

# Revenues
function calculate_revenues(m::SDCore, d::DataSD, kprice, n_draws; kwargs...)

    # Extract and set seed
    rng = get_rng(kwargs)
    seed = get(kwargs, :seed, 0)

    # Prepare data structures
    n_sessions = length(d)
    dU0, dE, dV, dW = m.dU0, m.dE, m.dV, m.dW

    zd_h, zs_h = precompute_discovery_and_search_values(m, d)

    mapping_characteristics = construct_mapping_characteristics(m, d)


    # Iterate over sessions and fill in welfare measures
    d_sim = deepcopy(d)
    d_sim.search_paths = [zeros(Int, length(d_sim.product_ids[i])) for i in eachindex(d_sim)]
    d_sim.stop_indices = [0 for i in eachindex(d_sim.product_ids)]
    utility_purchases = zeros(Float64, length(d_sim))

    max_n_products = get(kwargs, :max_n_products, maximum(length.(d.product_ids)))
    full_draws_for_fixed_seed = get(kwargs, :full_draws_for_fixed_seed, false)

    revenues = zeros(length(d))
    demands = zeros(length(d))

    _, data_chunks = get_chunks(n_sessions)
    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            # Create local variables that the thread can work with. Pre-allocation circumvenst allocations in the loop.
            local u = zeros(Float64, max_n_products)
            local zs = zeros(Float64, max_n_products)
            local v = zeros(Float64, max_n_products)
            local draws_u0, draws_e, draws_v, draws_w, draws_η =
                get_empty_vectors_for_shocks(full_draws_for_fixed_seed, max_n_products)

            for i in chunk

                rev, dem = calculate_revenues_i(d_sim, utility_purchases, m, kprice, i, n_draws,
                        dU0, dE, dV, dW,
                        u, zs, v, zd_h, zs_h,
                        mapping_characteristics,
                        rng, seed,
                        max_n_products,
                        draws_u0, draws_e, draws_v, draws_w, draws_η,
                        full_draws_for_fixed_seed)

                revenues[i] = rev
                demands[i] = dem
            end
        end
    end


    fetch.(tasks) # wait for all tasks to finish

    return Dict(:revenues => sum(revenues), :demand => sum(demands), :revenues_individual => revenues, :demand_individual => demands)

end

function calculate_revenues_i(d::DataSD, utility_purchases, m::SDCore, kprice, i, n_draws,
    dU0, dE, dV, dW,
    u, zs, v, zd_h, zs_h,
    mapping_characteristics,
    rng, seed, max_j,
    draws_u0, draws_e, draws_v, draws_w, draws_η,
    full_draws_for_fixed_seed)

    revenues = 0.0
    demand = 0.0

    if full_draws_for_fixed_seed
        Random.seed!(rng, seed + i) # set seed for each consumer
    end

    @inbounds for dd in 1:n_draws

        # Draw shocks if full evaluation. Otherwise drawn within
        # fill_path_i!, which does not draw for all products
        if full_draws_for_fixed_seed
            draws_u0[1] = rand(rng, dU0)
            for j in eachindex(draws_e)
                if j == 1 # skip taking draw for outside option
                    continue
                end
                draws_e[j] = rand(rng, dE) # draw e
                draws_v[j] = rand(rng, dV) # draw v
                draws_w[j] = rand(rng, dW) # draw w
            end
        elseif (isnothing(draws_u0) && isnothing(draws_e) &&
                isnothing(draws_v) && isnothing(draws_w) ) == false
            throw(ArgumentError("Draws need to be provided if not full evaluation."))
        end

        # Reset
        u .= typemin(Float64)
        zs .= typemin(Float64)
        v .= 0

        @views fill_path_i!(d, utility_purchases,
            m, i, dU0, dE, dV, dW,
            u, zs, v, zd_h, zs_h,
            mapping_characteristics,
            rng, max_j,
            draws_u0, draws_e, draws_v, draws_w, draws_η)

        # Compute revenues and demand
        j_chosen = d.purchase_indices[i]
        if j_chosen > 1 # j_chosen=1 is outside option -> no purchase
            demand += 1
            revenues += d.product_characteristics[i][j_chosen, kprice]
        end

    end

    return revenues / n_draws, demand / n_draws
end

# Fit evaluations
function calculate_fit_measures(m::SDCore, data::DataSD, n_sim; kwargs...)

    # Set seed
    rng = get_rng(kwargs)

    # Initialize empty arrays (otherwise error of not found)
    click_stats = []
    purchase_stats = []
    stop_probabilities = []

    # Track statistics across simulations to get percentiles
    click_probability_per_pos = zeros(Float64, maximum(length.(data.product_ids)), n_sim)
    purchase_probability_per_pos = zeros(Float64, maximum(length.(data.product_ids)), n_sim)

    conditional_on_search = get(kwargs, :conditional_on_search, false)

    # Generate data from new seed
    sim_seeds = rand(rng, 1:1000000, n_sim)

    n_sim_without_nan_purchase = 0

    d_sim = deepcopy(data)
    utility_purchases = zeros(Float64, length(data))

    for s in 1:n_sim

        # note: generating unconditional data is much faster than conditional data. So we generate unconditional data and
        # update the statistics to condition on clicks, using that, e.g., P(click pos 1 | click) = P(click pos 0) / P(click)
        generate_search_paths!(d_sim, utility_purchases, m; kwargs..., seed = sim_seeds[s],
            conditional_on_search, compute_min_discover_indices = false)

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

# Estimation
function prepare_arguments_likelihood(
        m::M, e::Estimator, d::DataSD; kwargs...) where {M <: SDModel}

    # Get functional forms
    zdfun = get_functional_form(m.zdfun)
    zsfun = get_functional_form(m.zsfun)

    # get data arguments
    data_arguments = prepare_data_arguments_likelihood(d)

    # Keep fixed seed: either random or provided by kwargs
    rng = get_rng(kwargs)
    seed = get(kwargs, :seed, rand(1:(10^9)))

    # Mapping characteristics to parameters
    mapping_characteristics = construct_mapping_characteristics(m, d; kwargs...)

    # Draws for unobserved heterogeneity
    ni_pts_whts = if has_unobserved_heterogeneity(m)
        prepare_draws_unobserved_heterogeneity(m,
            e.numerical_integration_method_heterogeneity, d, rng)
    else
        (nothing, nothing)
    end

    return data_arguments..., zdfun, zsfun, rng, seed,
        mapping_characteristics, ni_pts_whts
end

function prepare_cache(d)
    max_n_products = maximum(length.(d.product_ids))

    # Create cache to store xβ, xγ, xκ for each product (avoids )
    xβγ = DiffCache(zeros(max_n_products))
    xβκ = DiffCache(zeros(max_n_products))

    return (xβγ, xβκ)
end

function prepare_draws_unobserved_heterogeneity(m::M, ni::NIMethod,
        d::D, rng) where {M <: SDModel, D <: DataSD}

    n_dims = length(m.heterogeneity.parameters_with_unobserved_heterogeneity)
    n_cons = get_n_consumers(d)

    return generate_ni_points(n_dims, n_cons, ni, rng)

end

function prepare_data_arguments_likelihood(d::DataSD)
    all_possible_positions = d.positions[argmax(maximum.(d.positions))]
    has_search = sum.(d.consideration_sets) .> 0
    has_purchase = [d.product_ids[i][d.purchase_indices[i]] > 0 for i in eachindex(d)]

    return all_possible_positions, has_search, has_purchase
end

function vectorize_parameters(m::M; kwargs...) where {M <: SDModel}
    γ = isnothing(m.γ) ? eltype(m.β)[] : m.γ
    # Default estimate all parameters
    θ = if !haskey(kwargs, :fixed_parameters)
        θ = vcat(m.β[vcat(m.information_structure.indices_characteristics_β_union, end)],
            m.information_structure.γ[m.information_structure.indices_characteristics_γ_union],
            m.information_structure.κ[m.information_structure.indices_characteristics_κ_union],
            m.Ξ, m.ρ, m.ξ)
        if has_observed_heterogeneity(m)
            θ = vcat(θ, m.heterogeneity.ψ...)
        end
        if has_unobserved_heterogeneity(m)
            θ = add_unobserved_heterogeneity_parameters(m, θ)
        end
        θ
    else
        fixed_parameters = get(kwargs, :fixed_parameters, nothing)
        if !isnothing(fixed_parameters)
            θ = eltype(m.β)[]
            if :β ∉ fixed_parameters
                θ = vcat(θ, m.β[vcat(m.information_structure.indices_characteristics_β_union, end)])
            end
            if :γ ∉ fixed_parameters
                θ = vcat(θ, m.information_structure.γ[m.information_structure.indices_characteristics_γ_union])
            end
            if :κ ∉ fixed_parameters
                θ = vcat(θ, m.information_structure.κ[m.information_structure.indices_characteristics_κ_union])
            end
            if :Ξ ∉ fixed_parameters
                θ = vcat(θ, m.Ξ)
            end
            if :ρ ∉ fixed_parameters
                θ = vcat(θ, m.ρ)
            end
            if :ξ ∉ fixed_parameters
                θ = vcat(θ, m.ξ)
            end
            if has_observed_heterogeneity(m) && :ψ ∉ fixed_parameters
                θ = vcat(θ, m.heterogeneity.ψ...)
            end
            if has_unobserved_heterogeneity(m) && :Σ ∉ fixed_parameters
                θ = add_unobserved_heterogeneity_parameters(m, θ)
            end
        end
        θ
    end

    θ = add_distribution_parameters(m, θ, kwargs)

    return θ
end

function loglikelihood(θ::Vector{T}, model::M, estimator::SMLE, data::DataSD,
        args...; kwargs...) where {M <: SDModel, T <: Real}

    # Extract parameters implied by θ
    β, γ, κ, Ξ, ρ, ξ, ξρ, ind_last_par = extract_parameters(model, θ; kwargs...)
    ψ, U, ind_last_par = extract_heterogeneity_parameters(model, θ, ind_last_par; kwargs...)
    dE, dV, dU0, dW, _= extract_distributions(model, θ, ind_last_par; kwargs...)

    debug_print = get(kwargs, :debug_print, false)
    if debug_print
        println("θ = $θ")
        println("β = $β")
        println("γ = $γ")
        println("κ = $κ")
        println("Ξ = $Ξ")
        println("ρ = $ρ")
        println("ξ = $ξ")
        println("ξρ = $ξρ")
        println("dE = $dE")
        println("dV = $dV")
        println("dU0 = $dU0")
        println("dW = $dW")
        println("ψ = $ψ")
        println("U = $U")
    end

    # If ρ is larger than zero, return extremly small number for optimization to try other
    # values
    return_ind_last_par = get(kwargs, :return_ind_last_par, false)
    if !isnothing(ρ) && ρ[1] > 0 && !return_ind_last_par
        return -T(MAX_NUMERICAL)
    elseif !isnothing(ρ) && ρ[1] > 0
        return -T(MAX_NUMERICAL), ind_last_par
    end


    tasks = if has_unobserved_heterogeneity(model)
                @views construct_tasks_heterogeneous(
                    model, estimator, data,
                    β, γ, κ, Ξ, ρ, ξ, ξρ, ψ, U, dE, dV, dU0, dW,
                    args...)
            else
                @views construct_tasks_homogeneous(
                    model, estimator, data,
                    β, γ, κ, Ξ, ρ, ξ, ξρ, ψ, dE, dV, dU0, dW,
                    args...)

            end

    results_tasks = fetch.(tasks)

    LL1 = sum(t[1] for t in results_tasks)
    LL2 = estimator.conditional_on_search ? sum(t[2] for t in results_tasks) : zero(T)

    LL = LL1 - LL2

    if get(kwargs, :debug_print, false)
        println("LL = $LL")
        println("LL1 = $LL1")
        println("LL2 = $LL2")
    end

    if return_ind_last_par
        # prevent Inf values, helps AD
        if isinf(LL) || isnan(LL) || LL >= 0
            return -T(MAX_NUMERICAL), ind_last_par
        elseif estimator.conditional_on_search && LL2 < -7.0 * length(data) # if P(click) < 0.001
            return -T(MAX_NUMERICAL), ind_last_par
        else
            return LL::T, ind_last_par
        end
    end

    # prevent Inf values, helps AD
    if isinf(LL) || isnan(LL) || LL >= 0
        return -T(MAX_NUMERICAL)
    elseif estimator.conditional_on_search && LL2 < -7.0 * length(data) # if P(click) < 0.001, so logP(click) < -6.9
        return -T(MAX_NUMERICAL)
    else
        return LL::T
    end
end

function construct_tasks_homogeneous(model::M, estimator::SMLE, data::DataSD,
    β::Vector{T}, γ, κ, Ξ, ρ, ξ, ξρ, ψ, dE, dV, dU0, dW,
    args...) where {M <: SDModel, T <: Real}

    # Extract arguments
    all_possible_positions, has_search, has_purchase, zdfun, zsfun, rng,
        seed, mapping_characteristics, _ = args

    # Reset seed
    Random.seed!(rng, seed)

    # Create pre-allocated arrays for search and discovery values (same for all consumers by default)
    zd_h = isnothing(zdfun) ? Ξ : [zdfun(Ξ, ρ, pos) for pos in all_possible_positions]
    zs_h = isnothing(zsfun) ? ξ : [zsfun(ξ, ξρ, pos) for pos in all_possible_positions]

    # Define chunks for parallelization. Each chunk is a range of sessions for which a single task
    # calculates and sums up the likelihood. Looping over sessions as no need to keep track over
    # consumers.
    _, data_chunks = get_chunks(length(data))

    # Extract number of draws
    n_draws = estimator.numerical_integration_method.n_draws

    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin

            # Pre-allocate arrays per task -> avoid memory allocation by not having to re-create these arrays for every consumer
            local logL = zeros(T, 2) # sums up log likelihood across sessions

            # Pre-allocate arrays for heterogeneity
            local zd_hi = isnothing(zdfun) ? nothing : copy(zd_h)
            local zs_hi = copy(zs_h)
            local βi = copy(β)
            local γi = copy(γ)
            local κi = copy(κ)
            local cache = prepare_cache(data) # cache for xβ, xγ, xκ

            for i in chunk  # Iterate over sessions in chunk

                # Do inner likelihood calculations based on pre-allocated arrays
                if has_search[i] == 0 # Case 1: no clicks (implies also no purchase)
                    logL[1] += lik_no_searches(
                        model, zd_hi, zs_hi, βi, γi, κi, dV, dU0, mapping_characteristics,
                        data, i, n_draws, false, rng, true,
                        cache)
                elseif has_purchase[i] == 0 # Case 2: Some clicks but no purchase
                    logL[1] += lik_search_no_purchase(
                        model, zd_hi, zs_hi, βi, γi, κi, dE, dV, dU0, mapping_characteristics,
                        data, i, n_draws, rng, true,
                        cache)
                else # Case 3: Purchase a product
                    logL[1] += lik_purchase(
                        model, zd_hi, zs_hi, βi, γi, κi, dE, dV, dU0, mapping_characteristics,
                        data, i, n_draws, rng, true,
                        cache)
                end

                if estimator.conditional_on_search
                    logL[2] += lik_no_searches(
                        model, zd_h, zs_h, βi, γi, κi, dV, dU0, mapping_characteristics,
                        data, i, n_draws, true, rng, true,
                        cache)
                end
            end

            return logL # Return likelihood for chunk
        end
    end

    return tasks
end


function construct_tasks_heterogeneous(model::M, estimator::SMLE, data::DataSD,
    β::Vector{T}, γ::Vector{T}, κ::Vector{T}, Ξ, ρ, ξ, ξρ, ψ, U, dE, dV, dU0, dW,
    args...) where {M <: SDModel, T <: Real}

    # Extract arguments
    all_possible_positions, has_search, has_purchase, zdfun, zsfun, rng, seed,
        mapping_characteristics, ni_pts_whts = args
    ni_points, ni_weights = ni_pts_whts

    # Reset seed
    Random.seed!(rng, seed)

    # Create pre-allocated arrays for search and discovery values (same for all consumers by default)
    zd_h = isnothing(zdfun) ? Ξ : [zdfun(Ξ, ρ, pos) for pos in all_possible_positions]
    zs_h = isnothing(zsfun) ? ξ : [zsfun(ξ, ξρ, pos) for pos in all_possible_positions]

    # Define chunks for parallelization. Each chunk is a range of consumers for which
    # a single calculates and sums up the likelihood. For each consumer, we integrate out
    # the unobserved heterogeneity
    _, data_chunks = get_chunks(get_n_consumers(data))  # last consumer id is number of consumers

    # Extract number of draws
    n_draws = estimator.numerical_integration_method.n_draws

    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin

            # Pre-allocate arrays per task -> avoid memory allocation by not having to re-create these arrays for every consumer
            local logL = zeros(T, 2)    # sums up log likelihood across consumers
                                        # and heterogeneity draws draws for each consumer
            local Linner = ones(T, 2)   # product of likelihoods for each consumer across sessions given draw
            local Louter = zeros(T, 2)   # sum across (weighted) heterogeneity draws

            # Pre-allocate arrays for heterogeneity
            local zd_hi = isnothing(zdfun) ? nothing : copy(zd_h)
            local zs_hi = copy(zs_h)
            local βi = copy(β)
            local γi = copy(β)
            local κi = isnothing(κ) ? nothing : copy(κ)
            local ρi = isnothing(ρ) ? nothing : copy(ρ)
            local ξρi = isnothing(ξρ) ? nothing : copy(ξρ)
            local ni_points_i = zeros(T, size(ni_points[1])) # pre-allocate draws for each consumer

            for cons_id in chunk  # Iterate over consumers in chunk
                si = searchsorted(data.consumer_ids, cons_id) # get index of sessions belonging to consumer

                Louter .= 0 # reset likelihood for this consumer

                mul!(ni_points_i, U, ni_points[cons_id]) # adjust draws with covariance U * draws

                for di in axes(ni_points[cons_id], 2) # iterate over draws for this consumer

                    shocks = @views ni_points_i[:, di]
                    Ξi, ξi = add_shocks_parameters!(βi, γi, κi, ρi, ξρi, zd_hi, zs_hi, model,
                        β, γ, Ξ, ρ, ξ, ξρ, shocks, all_possible_positions, zdfun, zsfun)

                    # Update discovery and search values in case are reals
                    # (rather than arrays that are updated in place)
                    if isnothing(zdfun)
                        zd_hi = Ξi
                    end
                    if isnothing(zsfun)
                        zs_hi = ξi
                    end

                    if !isnothing(ρi) && ρi[1] > 0
                        Linner[1] += -T(MAX_NUMERICAL)
                        continue
                    end

                    # Reset likelihood for draw
                    Linner .= 1

                    for i in si # iterate over sessions of consumer

                        # Update based on session characteristics, which may differ for same consumer
                        if has_observed_heterogeneity(model)
                            # Update search values, discovery values, and preference parameters for each consumer

                            Ξi, ξi = add_observed_shifters_parameters!(βi, γi, κi, zd_hi, zs_hi, model,
                                Ξi, ρi, ξi, ξρi, ψ,
                                zdfun, zsfun,
                                data.session_characteristics[i],
                                all_possible_positions)

                            # Update discovery and search values in case are reals
                            # (rather than arrays that are updated in place)
                            if isnothing(zdfun)
                                zd_hi = Ξi
                            end
                            if isnothing(zsfun)
                                zs_hi = ξi
                            end

                            if !isnothing(ρi) && ρi[1] > 0
                                Linner[1] += -T(MAX_NUMERICAL)
                                continue
                            end
                        end

                        # Do inner likelihood calculations based on pre-allocated arrays
                        return_log = false
                        if has_search[i] == 0 # Case 1: no clicks (implies also no purchase)
                            Linner[1] *= lik_no_searches(
                                model, zd_hi, zs_hi, βi, γi, κi, dV, dU0, mapping_characteristics,
                                data, i, n_draws, false, rng, return_log)
                        elseif has_purchase[i] == 0 # Case 2: Some clicks but no purchase
                            Linner[1] *=lik_search_no_purchase(
                                model, zd_hi, zs_hi, βi, γi, κi, dE, dV, dU0, mapping_characteristics,
                                data, i, n_draws, rng, return_log)
                        else # Case 3: Purchase a product
                            Linner[1] *= lik_purchase(
                                model, zd_hi, zs_hi, βi, γi, κi, dE, dV, dU0, mapping_characteristics,
                                data, i, n_draws, rng, return_log)
                        end

                        if estimator.conditional_on_search
                            Linner[2] *= lik_no_searches(
                                            model, zd_h, zs_h, βi, γi, κi, dV, dU0, mapping_characteristics,
                                            data, i, n_draws, true, rng, return_log)
                        end
                    end
                    # Sum over draws
                    Louter[1] += ni_weights[di] * Linner[1]
                    Louter[2] += ni_weights[di] * Linner[2]
                end

                logL[1] += lik_return_stable(Louter[1], true)
                logL[2] += lik_return_stable(Louter[2], true)
            end

            return logL # Return likelihood for chunk
        end
    end

    return tasks
end

function unpack_cache(cache, β)

    xβγ = get_tmp(cache[1], β)
    xβκ = get_tmp(cache[2], β)

    return xβγ, xβκ
end

function unpack_cache(cache::Union{Tuple{Vector{T}, Vector{T}},
                                    Vector{Vector{T}}}, β) where {T <: Real}

    xβγ = get_tmp(cache[1], β)
    xβκ = get_tmp(cache[2], β)

    return xβγ, xβκ
end


# Parameter handling
function extract_parameters(m::M, θ::Vector{T}; kwargs...) where {M <: SDModel, T <: Real}
    n_ρ = length(m.ρ)
    n_ξρ = length(m.ξρ)

    # track where in parameter vector we are and move it.
    ind_current = 1

    # Default: estimate all parameters
    if !haskey(kwargs, :fixed_parameters)
        β = T.(m.β) # convert to type T, e.g., Float64
        for i in m.information_structure.indices_characteristics_β_union
            β[i] = θ[ind_current]
            ind_current += 1
        end
        β[end] = θ[ind_current] # last parameter is the outside option
        ind_current += 1

        γ = T.(m.information_structure.γ)
        for i in m.information_structure.indices_characteristics_γ_union
            γ[i] = θ[ind_current]
            ind_current += 1
        end

        κ = T.(m.information_structure.κ)
        for i in m.information_structure.indices_characteristics_κ_union
            κ[i] = θ[ind_current]
            ind_current += 1
        end

        Ξ = θ[ind_current]
        ind_current += 1
        ρ = θ[ind_current:(ind_current + n_ρ - 1)]
        ind_current += n_ρ
        ξ = θ[ind_current]
        ind_current += 1
        ξρ = θ[ind_current:(ind_current + n_ξρ - 1)]
        ind_current += n_ξρ
        return β, γ, κ, Ξ, ρ, ξ, ξρ, ind_current
    end

    # If keyword supplied, don't estimate parameters indicated in fixed_parameters
    fixed_parameters = get(kwargs, :fixed_parameters, nothing)

    β = T.(m.β)
    if :β ∉ fixed_parameters
        n_β = length(m.β)
        for i in 1:n_β
            β[i] = θ[ind_current]
            ind_current += 1
        end
        β[end] = θ[ind_current] # last parameter is the outside option
        ind_current += 1
    end

    γ = T.(m.γ)
    if :γ ∉ fixed_parameters
        n_γ = length(m.information_structure.γ)
        for i in 1:n_γ
            γ[i] = θ[ind_current]
            ind_current += 1
        end
    end

    κ = T.(m.κ)
    if :κ ∉ fixed_parameters
        n_κ = length(m.information_structure.κ)
        for i in 1:n_κ
            κ[i] = θ[ind_current]
            ind_current += 1
        end
    end

    Ξ = if :Ξ ∈ fixed_parameters
        T(m.Ξ)
    else
        ind_current += 1
        θ[ind_current - 1]
    end
    ρ = if :ρ ∈ fixed_parameters
        T.(m.ρ)
    else
        ind_current += n_ρ
        θ[ind_current:(ind_current + n_ρ - 1 - 1)]
    end
    ξ = if :ξ ∈ fixed_parameters
        T(m.ξ)
    else
        ind_current += 1
        θ[ind_current - 1]
    end
    ξρ = if :ξρ ∈ fixed_parameters
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
    return β, γ, κ, Ξ, ρ, ξ, ξρ, ind_current
end

function construct_model_from_pars(θ::Vector{T}, m::M; kwargs...) where {M <: SDModel, T <: Real}

    # Extract parameters from vector, some may be fixed through kwargs
    β, γ, κ, Ξ, ρ, ξ, ξρ, ind_last_par = extract_parameters(m, θ; kwargs...)
    ψ, U, ind_last_par = extract_heterogeneity_parameters(m, θ, ind_last_par; kwargs...)
    dE, dV, dU0, dW, ind_last_par= extract_distributions(m, θ, ind_last_par; kwargs...)

    information_structure = deepcopy(m.information_structure)
    information_structure.γ = γ
    information_structure.κ = κ

    heterogeneity = construct_new_heterogeneity_specification(m, ψ, U)

    # Construct model from parameters
    m_new = SDCore{T}(; β, Ξ, ρ, ξ, ξρ, dE, dV, dU0, dW, zdfun = m.zdfun, zsfun = m.zsfun,
                        information_structure,
                        heterogeneity)

    if get(kwargs, :return_ind_last_par, false)
        return m_new, ind_last_par
    else
        return m_new
    end
end

"""
Construct shock distributions using variances in vector θ. Starts from index c.
"""
function extract_distributions(m::M, θ::Vector{T}, c; kwargs...) where {M <: SDModel, T <: Real}

    # Default: don't estimate any variance
    if !haskey(kwargs, :estimation_shock_variances)
        dE = m.dE # convoluted way to allow for distributions other than Normal
        dV = m.dV
        dU0 = m.dU0
        dW = typeof(m) <: SDCore ? m.dW : nothing
        return dE, dV, dU0, dW, c
    end

    estimation_shock_distributions = get(kwargs, :estimation_shock_variances, nothing)

    # Extract distributions
    dE = if :σ_dE ∈ estimation_shock_distributions
        dE = eval(nameof(typeof(m.dE)))(params(m.dE)[1:(end - 1)]..., abs(θ[c]))
        c += 1
        dE
    else
        m.dE
    end

    dV = if :σ_dV ∈ estimation_shock_distributions
        dV = eval(nameof(typeof(m.dV)))(params(m.dV)[1:(end - 1)]..., abs(θ[c]))
        c += 1
        dV
    else
        m.dV
    end

    dU0 = if :dUequaldE ∈ estimation_shock_distributions && :σ_dE ∈ estimation_shock_distributions
        dE
    elseif :dUequaldV ∈ estimation_shock_distributions && :σ_dV ∈ estimation_shock_distributions
        dV
    elseif :σ_dU0 ∈ estimation_shock_distributions
        np = length(params(m.dU0))
        eval(nameof(typeof(m.dU0)))(params(m.dU0)[1:(end - np + 1)]..., abs.(θ[c:end])...)
    else
        m.dU0
    end

    dW = typeof(m) <: SDCore ? m.dW : nothing # dW currently cannot be estimated

    return dE, dV, dU0, dW, c
end

function add_distribution_parameters(m::M, θ, kwargs) where {M <: SDModel}

    # Default: don't estimate any variance
    if !haskey(kwargs, :estimation_shock_variances)
        return θ
    end
    estimation_shock_distributions = get(kwargs, :estimation_shock_variances, nothing)
    # Extract distributions
    if :σ_dE ∈ estimation_shock_distributions
        θ = vcat(θ, params(m.dE)[end])
    end
    if :σ_dV ∈ estimation_shock_distributions
        θ = vcat(θ, params(m.dV)[end])
    end
    if :σ_dU0 ∈ estimation_shock_distributions
        θ = vcat(θ, params(m.dU0)[2:end]...)
    end

    return θ
end

"""
    fill_indices_min_discover!(data::DataSD)
Use clicks to add indices of minimum position consumers must have discovered in `data`.
"""
function fill_indices_min_discover!(d::DataSD)
    min_discover_indices = get_indices_min_discover(d)

    d.min_discover_indices = min_discover_indices
    return nothing
end

function add_shocks_parameters!(βi, γi, κi, ρi, ξρi, zd_h, zs_h, m::M,
        β, γ, Ξ, ρ, ξ::T, ξρ, ni_points,
        all_possible_positions,
        zdfun, zsfun) where {M <: SDModel, T <: Real}

    # Reset
    βi .= β
    γi .= γ

    if !isnothing(ρi)
        ρi .= ρ
    end
    if !isnothing(ξρ)
        ξρi .= ξρ
    end

    Ξi = Ξ
    ξi = ξ

    hs = m.heterogeneity._unobserved_hs

    c = 1

    for k in hs.β
        βi[k] = βi[k] + ni_points[c]
        c += 1
    end

    for k in hs.γ
        γi[k] = γi[k] + ni_points[c]
        c += 1
    end
    for k in hs.κ
        κi[k] = κi[k] + ni_points[c]
        c += 1
    end

    if hs.Ξ
        Ξi = Ξi + ni_points[c]
        c += 1
    end

    if !isnothing(ρi) # in WM model, returns ρ as nothing when extracting parameters
        for k in hs.ρ
            ρi[k] = ρi[k] + ni_points[c]
            c += 1
        end
    end

    if hs.ξ
        ξi = ξi + ni_points[c]
        c += 1
    end

    if isnothing(ρi) # in WM model, use ρ as only parameter
        for k in hs.ρ
            ξρi[k] = ξρi[k] + ni_points[c]
            c += 1
        end
    else
        for k in hs.ξρ
            ξρi[k] = ξρi[k] + ni_points[c]
            c += 1
        end
    end

    if has_observed_heterogeneity(m) == false
        # with observed heterogeneity, updated later on and can be skipped
        if !isnothing(zdfun)
            update_reservation_values_across_positions!(
            zd_h, Ξi, ρi, zdfun, all_possible_positions) # update discovery values
        end
        if !isnothing(zsfun)
            update_reservation_values_across_positions!(
                zs_h, ξi, ξρi, zsfun, all_possible_positions) # update search values
        end
    end

    return Ξi, ξi
end

function add_observed_shifters_parameters!(βi, γi, κi, zd_h, zs_h, m::M,
        Ξ, ρ, ξ::T, ξρ, ψ,
        zdfun, zsfun,
        session_characteristics,
        all_possible_positions) where {M <: SDModel, T <: Real}

    Ξi = Ξ
    ρi = ρ
    ξi = ξ
    ξρi = ξρ

    heterogeneity_specification = m.heterogeneity._observed_hs

    c = 1

    for k in heterogeneity_specification.β
        βi[k] = βi[k] + session_characteristics' * ψ[c]
        c += 1
    end

    for k in heterogeneity_specification.γ
        γi[k] = γi[k] + session_characteristics' * ψ[c]
        c += 1
    end

    for k in heterogeneity_specification.κ
        κi[k] = κi[k] + session_characteristics' * ψ[c]
        c += 1
    end

    if heterogeneity_specification.Ξ
        Ξi = Ξi + session_characteristics' * ψ[c]
        c += 1
    end

    if !isnothing(ρi) # in WM model, returns ρ as nothing when extracting parameters
        for k in heterogeneity_specification.ρ
            ρi[k] = ρi[k] + session_characteristics' * ψ[c]
            c += 1
        end
    end

    if heterogeneity_specification.ξ
        ξi = ξi + session_characteristics' * ψ[c]
        c += 1
    end

    if isnothing(ρi) # in WM model, use ρ as only parameter
        for k in heterogeneity_specification.ρ
            ξρi[k] = ξρi[k] + session_characteristics' * ψ[c]
            c += 1
        end
    else
        for k in heterogeneity_specification.ξρ
            ξρi[k] = ξρi[k] + session_characteristics' * ψ[c]
            c += 1
        end
    end

    if !isnothing(zdfun)
        update_reservation_values_across_positions!(
        zd_h, Ξi, ρi, zdfun, all_possible_positions) # update discovery values
    end
    if !isnothing(zsfun)
        update_reservation_values_across_positions!(
            zs_h, ξi, ξρi, zsfun, all_possible_positions) # update search values
    end

    return Ξi, ξi
end

function construct_individual_parameters(model::M, i, data::DataSD, draws_η)  where {M <: SDModel}

    @unpack β, Ξ, ρ, ξ, ξρ = model
    @unpack γ, κ = model.information_structure

    if !has_heterogeneity(model)
        return β, γ, κ, Ξ, ρ, ξ, ξρ
    end

    βi = copy(β)
    γi = isnothing(γ) ? γ : copy(γ)
    κi = isnothing(κ) ? κ : copy(κ)
    Ξi = Ξ
    ρi = copy(ρ)
    ξi = ξ
    ξρi = copy(ξρ)
    ψ = model.heterogeneity.ψ

    if has_observed_heterogeneity(model)
        observed_hs = model.heterogeneity._observed_hs
        session_characteristics = @views data.session_characteristics[i]

        c = 1
        for k in observed_hs.β
            βi[k] = βi[k] + session_characteristics' * ψ[c]
            c += 1
        end

        for k in observed_hs.γ
            γi[k] = γi[k] + session_characteristics' * ψ[c]
            c += 1
        end

        for k in observed_hs.κ
            κi[k] = κi[k] + session_characteristics' * ψ[c]
            c += 1
        end

        if observed_hs.Ξ
            Ξi = Ξi + session_characteristics' * ψ[c]
            c += 1
        end

        for k in observed_hs.ρ
            if k == 1 && typeof(ρi) <: Real # if not vector, cannot replace with ρi[k]
                ρi = ρi + session_characteristics' * ψ[c]
            else
                ρi[k] = ρi[k] + session_characteristics' * ψ[c]
            end
            c += 1
        end

        if observed_hs.ξ
            ξi = ξi + session_characteristics' * ψ[c]
            c += 1
        end

        for k in observed_hs.ξρ
            ξρi[k] = ξρi[k] + session_characteristics' * ψ[c]
            c += 1
        end
    end

    if has_unobserved_heterogeneity(model)
        unobserved_hs = model.heterogeneity._unobserved_hs
        draws_ηi = @views draws_η[data.consumer_ids[i], :]

        c = 1
        for k in unobserved_hs.β
            βi[k] = βi[k] + draws_ηi[c]
            c += 1
        end

        for k in unobserved_hs.γ
            γi[k] = γi[k] + draws_ηi[c]
            c += 1
        end

        for k in unobserved_hs.κ
            κi[k] = κi[k] + draws_ηi[c]
            c += 1
        end

        if unobserved_hs.Ξ
            Ξi = Ξi + draws_ηi[c]
            c += 1
        end

        for k in unobserved_hs.ρ
            if k == 1 && typeof(ρi) <: Real # if not vector, cannot replace with ρi[k]
                ρi = ρi + draws_ηi[c]
            else
                ρi[k] = ρi[k] + draws_ηi[c]
            end
            c += 1
        end

        if unobserved_hs.ξ
            ξi = ξi + draws_ηi[c]
            c += 1
        end

        for k in unobserved_hs.ξρ
            ξρi[k] = ξρi[k] + draws_ηi[c]
            c += 1
        end
    end

    return βi, γi, κi, Ξi, ρi, ξi, ξρi
end

function update_reservation_values_across_positions!(z, Ξ, ρ, f::Function,
        all_possible_positions)
    for (j, h) in enumerate(all_possible_positions)
        z[j] = f(Ξ, ρ, h)
    end

    return nothing
end


function run_compatibility_checks(model::SDModel, data::DataSD; kwargs...)

    svs = model.information_structure
    if typeof(svs.indices_characteristics_β_individual) <: Vector{Vector{T}} where T &&
        length(svs.indices_characteristics_β_individual) != length(data)
        throw(ArgumentError("Length of indices_characteristics_β_individual does not match number of sessions in data"))
    end

    if sum(model.β[setdiff(1:end-1, svs.indices_characteristics_β_union)]) != 0
        #note: last index is outside option, which is not part of the indices_characteristics_β_union
        throw(ArgumentError("Model β has non-zero values for indices not in indices_characteristics_β_union"))
    end

    if typeof(svs.indices_characteristics_γ_individual) <: Vector{Vector{T}} where T &&
        length(svs.indices_characteristics_γ_individual) != length(data)
        throw(ArgumentError("Length of indices_characteristics_γ_individual does not match number of sessions in data"))
    end

    if sum(svs.γ[setdiff(1:end, svs.indices_characteristics_γ_union)]) != 0
        throw(ArgumentError("Model γ has non-zero values for indices not in indices_characteristics_γ_union"))
    end

    if typeof(svs.indices_characteristics_κ_individual) <: Vector{Vector{T}} where T &&
        length(svs.indices_characteristics_κ_individual) != length(data)
        throw(ArgumentError("Length of indices_characteristics_κ_individual does not match number of sessions in data"))
    end

    if sum(svs.κ[setdiff(1:end, svs.indices_characteristics_κ_union)]) != 0
        throw(ArgumentError("Model κ has non-zero values for indices not in indices_characteristics_κ_union"))
    end

    if has_heterogeneity(model)
        throw(ArgumentError("Model has heterogeneity, which is not yet supported."))
    end

end
