"""
*Search and Discovery* model `SD{T} <: SDModel`  with the following parameterization: 
- uᵢⱼ = xⱼβ + xⱼκ + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ = xⱼβ + xⱼγ + ξ + νᵢⱼ 
- uᵢ₀ = β0 + ηᵢ , ηᵢ ~ dU0
- zd(h) = zdfun(Ξ, ρ, pos) with ρ ≤ 0
- For the estimation, the specification of `xⱼβ`, `xⱼκ`, and `xⱼγ` are determined by `information_structure`.

## Fields:  
- `β::Vector{T}`: Vector of preference weights. 
- `Ξ::T`: Baseline Ξ for position 1 (not demeaned).
- `ρ::Union{T, Vector{T}}`: Parameter(s) governing decrease of Ξ across positions.
- `ξ::T`: Baseline ξ.
- `dE::Distribution`: Distribution of εᵢⱼ.
- `dV::Distribution`: Distribution of νᵢⱼ.
- `dU0::Distribution`: Distribution of ηᵢ. 
- `zdfun::String`: Select functional form f(Ξ, ρ, h) that determines the discovery value in position h. 
- `information_structure::InformationStructureSpecification{T}`: Specification of information structure, including `γ`, `κ` and characteristics for `β`, `γ`, and `κ`. See `InformationStructureSpecification` for details.
- `cs::Union{Vector{Vector{T}}, Nothing}`: Search costs on a product level. Initialized as `nothing` and only used for welfare calculations. Vector of vector, matching structure in data. Can be added through `calculate_costs!(m, data; kwargs...)`. 
- `cd::Union{Vector{T}, Nothing}`: Discovery costs, specific to sessions. Initialized as `nothing` and only used for welfare calculations. Can be updated through `calculate_costs!(m, data; kwargs...)`. 
- `heterogeneity::HeterogeneitySpecification`: Specification of heterogeneity (unobserved and observed) in the model. By default assumes homogeneous model.
"""
@with_kw mutable struct SD{T} <: SDModel where {T <: Real}
    β::Vector{T}
    Ξ::T
    ρ::Union{T, Vector{T}} 
    ξ::T
    dE::Distribution
    dV::Distribution
    dU0::Distribution
    zdfun::String

    # Parameters with defaults 
    information_structure::InformationStructureSpecification{T} = InformationStructureSpecification(length(β))
    heterogeneity::HeterogeneitySpecification{T} = HeterogeneitySpecification()

    cs::Union{Vector{Vector{T}}, Nothing} = nothing
    cd::Union{Vector{T}, Nothing} = nothing


    @assert ρ[1]<=0 "ρ[1] must be less or equal to zero for weakly decreasing discovery value across positions."
    @assert length(information_structure.γ) == length(β) "Length of γ must be equal to length of β."
    @assert length(information_structure.κ) == length(β) "Length of κ must be equal to length of β."

end

function ==(m1::SD, m2::SD)
    return isequal(m1.β, m2.β) &&
           isequal(m1.Ξ, m2.Ξ) &&
           isequal(m1.ρ, m2.ρ) &&
           isequal(m1.ξ, m2.ξ) &&
           isequal(m1.dE, m2.dE) &&
           isequal(m1.dV, m2.dV) &&
           isequal(m1.dU0, m2.dU0) &&
           isequal(m1.zdfun, m2.zdfun) &&
           isequal(m1.cs, m2.cs) &&
           isequal(m1.cd, m2.cd) && 
           isequal(m1.information_structure, m2.information_structure) &&
            isequal(m1.heterogeneity, m2.heterogeneity)
end

# Add constructors for SD with type conversions for convenience
function convert_ρ(ρ, T)
    if ρ isa AbstractVector
        convert(Vector{T}, ρ)
    else
        convert(T, ρ)
    end 
end
function SD(β, Ξ, ρ, ξ, dE, dV, dU0, zdfun, information_structure, heterogeneity,
    cs, cd)
    Ξ, ξ = promote(Ξ, ξ)
    T = eltype(Ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    return SD(convert(Vector{T}, β),
        Ξ, ρ, ξ, dE, dV, dU0, zdfun, information_structure, heterogeneity, cs, cd)
end

# this one for some reason is necessary as Parameters.jl does not seem to enforce T for β
function SD(β::Vector{Any}, Ξ, ρ, ξ, dE, dV, dU0, zdfun,
        information_structure, heterogeneity, cs, cd)
    Ξ, ξ = promote(Ξ, ξ)
    T = eltype(Ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    return SD(convert(Vector{T}, β), Ξ, ρ, ξ, dE, dV, dU0, zdfun,
        information_structure, heterogeneity, cs, cd)
end

function SD(β, Ξ, ρ, ξ, dE, dV, dU0, zdfun; cs = nothing, cd = nothing,
        information_structure = InformationStructureSpecification(),
        heterogeneity = HeterogeneitySpecification())
    Ξ, ξ = promote(Ξ, ξ)
    T = eltype(Ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    return SD(convert(Vector{T}, β),
             Ξ, ρ, ξ, dE, dV, dU0, zdfun, cs, cd, information_structure, heterogeneity)
end 


# Functions from SDCore to SD
function SDCore(m::SD)
    SDCore(; β = m.β, Ξ = m.Ξ, ρ = m.ρ, ξ = m.ξ, ξρ = [0.0], cs = m.cs, cd = m.cd,
        dE = m.dE, dV = m.dV, dU0 = m.dU0, dW = Normal(0, 0), zdfun = m.zdfun,
        zsfun = "", information_structure = m.information_structure, heterogeneity = m.heterogeneity)
end

function calculate_welfare(
        model::SD, data::DataSD, n_sim; method = "effective_values", kwargs...)
    calculate_welfare(SDCore(model), data, n_sim; method = method, kwargs...)
end

function calculate_revenues(model::SD, data::DataSD, kprice, n_draws; kwargs...)
    return calculate_revenues(SDCore(model), data, kprice, n_draws; kwargs...)
end


function calculate_costs!(m::SD, d, n_draws...;
        force_recompute = true,
        cd_kwargs...)
    m1 = SDCore(m)
    calculate_costs!(m1, d, n_draws...; force_recompute, cd_kwargs...)

    m.cs = m1.cs
    m.cd = m1.cd

    return nothing
end

function generate_data(m::SD, n_consumers, n_sessions_per_consumer; kwargs...)
    generate_data(SDCore(m), n_consumers, n_sessions_per_consumer; kwargs...)
end
generate_data(m::SD, data::DataSD; kwargs...) = generate_data(SDCore(m), data; kwargs...)

function calculate_fit_measures(m::SD, data::DataSD, n_sim; kwargs...)
    calculate_fit_measures(SDCore(m), data, n_sim; kwargs...)
end

# Estimation 
function prepare_arguments_likelihood(m::SD, e::Estimator, d::DataSD; kwargs...)

    # Get functional forms 
    zdfun = get_functional_form(m.zdfun)

    # get data arguments 
    data_arguments = prepare_data_arguments_likelihood(d)

    # Keep fixed seed: either random or provided by kwargs 
    rng = get_rng(kwargs)
    seed = get(kwargs, :seed, rand(rng, 1:(10^9)))

    # Mapping characteristics to parameters
    mapping_characteristics = construct_mapping_characteristics(m, d; kwargs...)

    # Draws for unobserved heterogeneity 
    ni_pts_whts = prepare_draws_unobserved_heterogeneity(m, 
        e.numerical_integration_method_heterogeneity, d, rng)

    return data_arguments..., zdfun, nothing, rng, seed, 
        mapping_characteristics, ni_pts_whts 
end


# Vectorize parameters 
function vectorize_parameters(m::SD; kwargs...)
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

function construct_model_from_pars(θ::Vector{T}, m::SD; kwargs...) where {T <: Real}

    # Extract parameters from vector, some may be fixed through kwargs 
    β, γ, κ, Ξ, ρ, ξ, _, ind_last_par = extract_parameters(m, θ; kwargs...)
    ψ, U, ind_last_par = extract_heterogeneity_parameters(m, θ, ind_last_par; kwargs...)
    dE, dV, dU0, _, ind_last_par = extract_distributions(m, θ, ind_last_par; kwargs...)

    information_structure = deepcopy(m.information_structure)
    information_structure.γ = γ
    information_structure.κ = κ

    heterogeneity = construct_new_heterogeneity_specification(m, ψ, U)

    # Construct model from parameters
    m_new = SD{T}(; β, Ξ, ρ, ξ, dE, dV, dU0, zdfun = m.zdfun,
        information_structure, heterogeneity) 

    if get(kwargs, :return_ind_last_par, false)
        return m_new, ind_last_par
    else
        return m_new
    end
end

function extract_parameters(m::SD, θ::Vector{T}; kwargs...) where {T <: Real}
    n_ρ = length(m.ρ)

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


        γ = if length(m.information_structure.indices_characteristics_γ_union) == 0 
            m.information_structure.γ
        else
            γ = T.(m.information_structure.γ)
            for i in m.information_structure.indices_characteristics_γ_union
                γ[i] = θ[ind_current]
                ind_current += 1
            end
            γ
        end

        κ = if length(m.information_structure.indices_characteristics_κ_union) == 0 
            m.information_structure.κ
        else
            κ = T.(m.information_structure.κ)
            for i in m.information_structure.indices_characteristics_κ_union
                κ[i] = θ[ind_current]
                ind_current += 1
            end
            κ
        end

        Ξ = θ[ind_current]
        ind_current += 1
        ρ = -abs.(θ[ind_current:(ind_current + n_ρ - 1)])
        ind_current += n_ρ
        ξ = θ[ind_current]
        ind_current += 1
        return β, γ, κ, Ξ, ρ, ξ, nothing, ind_current
    end

    # If keyword supplied, don't estimate parameters indicated in fixed_parameters
    fixed_parameters = get(kwargs, :fixed_parameters, nothing)

    β = T.(m.β) 
    if :β ∉ fixed_parameters 
        for i in m.information_structure.indices_characteristics_β_union
            β[i] = θ[ind_current]
            ind_current += 1
        end
        β[end] = θ[ind_current] # last parameter is the outside option
        ind_current += 1
    end 

    γ = T.(m.information_structure.γ) 
    if :γ ∉ fixed_parameters
        for i in m.information_structure.indices_characteristics_γ_union
            γ[i] = θ[ind_current]
            ind_current += 1
        end
    end

    κ = T.(m.information_structure.κ)
    if :κ ∉ fixed_parameters
        for i in m.information_structure.indices_characteristics_κ_union
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
        -abs.(θ[ind_current - n_ρ:ind_current - 1])
    end

    ξ = if :ξ ∈ fixed_parameters
        T(m.ξ)
    else
        ind_current += 1
        θ[ind_current - 1]
    end

    return β, γ, κ, Ξ, ρ, ξ, nothing, ind_current
end

@inline function construct_range_positions(d::DataSD, i, complement)
    n_products = length(d.positions[i])

    min_position_discover = complement ? searchsortedfirst(d.positions[i], 1) - 1 :
                            d.min_discover_indices[i] # if complement 

    range_positions = if isnothing(d.stop_indices)
        min_position_discover:n_products 
    else
        d.stop_indices[i]:d.stop_indices[i]
    end

    return range_positions, min_position_discover, n_products
    
end

function lik_no_searches(m::SD, zd_h::Vector{T}, ξ::T, 
        β::Vector{T}, γ, κ, dV, dU0,
        mapping_characteristics,
        d::DataSD, i::Int, n_draws, complement, rng, return_log,
        cache) where {T <: Real}
    positions = @views d.positions[i]
    
    ixb = mapping_characteristics[1][i]
    ixg = mapping_characteristics[2][i]
    ixk = mapping_characteristics[3][i]

    LL = zero(T)

    range_positions, _, n_products = construct_range_positions(d, i, complement)

    xβγ, _ = unpack_cache(cache, β) 
    xβγ .= typemin(T) # reset 

    @inbounds for dd in 1:n_draws, h in range_positions

        # If not last product in same position or last product, skip 
        if h < n_products && positions[h] == positions[h + 1]
            continue
        end

        # Set lower bound for truncation based on position 
        lb::T = if h < n_products # not yet last position 
            zd_h[h + 1] - β[end] * d.product_characteristics[i][1, end]
        else # no lower bound if last position 
            -T(MAX_NUMERICAL)
        end

        # Set upper bound for truncation based on position
        ub::T = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
            T(MAX_NUMERICAL)
        else
            zd_h[h] - β[end] * d.product_characteristics[i][1, end]
        end

        # Get probability of u0 in bounds and draw for u0 
        prob_u0_in_bounds = trunc_cdf(dU0, lb, ub)
        u0_draw = rand_trunc(rng, dU0, lb, ub) + β[end] * d.product_characteristics[i][1, end]

        # Initialize for probability
        prob_no_search_given_draw = one(T)

        # Loop over products up to one discovered 
        for j in 2:h
            if xβγ[j] == typemin(T) # fill only if not already filled 
                xβ_j, xγ_j, _ = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ, 
                    ixb, ixg, ixk; no_κ = true) 
                xβγ[j] = xβ_j + xγ_j
            end

            zj = xβγ[j] + ξ

            prob_no_search_given_draw *= prob_not_search(m, u0_draw, zj, dV)

            if prob_no_search_given_draw == 0
                prob_no_search_given_draw = T(ALMOST_ZERO_NUMERICAL)  # adding this helps AD 
                break
            end
        end
        if complement
            LL += (1 - prob_no_search_given_draw) * prob_u0_in_bounds
        else
            LL += prob_no_search_given_draw * prob_u0_in_bounds
        end
    end

    return lik_return_stable(LL/n_draws, return_log) 
end

function lik_search_no_purchase(m::SD, zd_h::Vector{T}, ξ::T, β::Vector{T}, γ, κ, dE, dV,
        dU0, mapping_characteristics, d::DataSD, i::Int, n_draws, 
        rng, return_log, cache) where {T <: Real}
    positions = @views d.positions[i]
    consideration_set = @views d.consideration_sets[i]
    ixb = mapping_characteristics[1][i]
    ixg = mapping_characteristics[2][i]
    ixk = mapping_characteristics[3][i]

    xβγ, xβκ = unpack_cache(cache, β) 
    xβγ .= typemin(T) # reset
    xβκ .= typemin(T) # reset


    LL = zero(T)

    range_positions, _, n_products = construct_range_positions(d, i, false)

    @inbounds for dd in 1:n_draws, h in range_positions

        # If not last product in same position or last product, skip 
        if h < n_products && positions[h] == positions[h + 1]
            continue
        end
        # Set lower bound for truncation based on position 
        lb::T = if h < n_products # not yet last position 
            zd_h[h + 1] - β[end] * d.product_characteristics[i][1, end]
        else # no lower bound if last position 
            -T(MAX_NUMERICAL)
        end

        # Set upper bound for truncation based on position
        ub::T = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
            T(MAX_NUMERICAL)
        else
            zd_h[h] - β[end] * d.product_characteristics[i][1, end]
        end

        # Get probability of u0 in bounds and draw for u0 
        prob_u0_in_bounds = trunc_cdf(dU0, lb, ub)
        u0_draw = rand_trunc(rng, dU0, lb, ub) + β[end] * d.product_characteristics[i][1, end]

        prob_searches_given_draw = one(T)

        for j in 2:h
            if consideration_set[j] # searched item 
                if xβγ[j] == typemin(T) # fill only if not already filled 
                    xβ_j, xγ_j, xκ_j = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ, 
                        ixb, ixg, ixk) 
                    xβγ[j] = xβ_j + xγ_j
                    xβκ[j] = xβ_j + xκ_j
                end


                prob_searches_given_draw *= prob_search_not_buy(
                    m, xβγ[j], xβκ[j], ξ, u0_draw, u0_draw, dE, dV)
            else # not searched item
                if xβγ[j] == typemin(T) # fill only if not already filled 
                    xβ_j, xγ_j, _ = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ, 
                        ixb, ixg, ixk; no_κ = true)
                    xβγ[j] = xβ_j + xγ_j
                end


                prob_searches_given_draw *= prob_not_search(m, u0_draw, xβγ[j] + ξ, dV)
            end

            if prob_searches_given_draw == 0
                prob_searches_given_draw = T(1e-100)
                break
            end
        end

        LL += prob_searches_given_draw * prob_u0_in_bounds
    end
    return lik_return_stable(LL/n_draws, return_log) 
end

function lik_purchase(m::SD, zd_h::Vector{T}, ξ::T, β::Vector{T}, γ, κ, dE, dV,
        dU0, mapping_characteristics, d::DataSD, i::Int, n_draws, rng,
        return_log, cache) where {T <: Real}
    positions = @views d.positions[i]
    with_outside_option = d.product_ids[i][1] == 0
    consideration_set = @views d.consideration_sets[i]
    ixb = mapping_characteristics[1][i]
    ixg = mapping_characteristics[2][i]
    ixk = mapping_characteristics[3][i]

    xβγ, xβκ = unpack_cache(cache, β) 
    xβγ .= typemin(T) # reset
    xβκ .= typemin(T) # reset

    LL = zero(T)

    k = d.purchase_indices[i] # index of purchased product

    range_positions, min_position_discover, n_products = construct_range_positions(d, i, false)

    # xb of purchased 
    xβ_k, xγ_k, xκ_k = @views construct_util_parts(d.product_characteristics, i, k, β, γ, κ, 
                    ixb, ixg, ixk)
    xβγ_k = xβ_k + xγ_k
    xβκ_k = xβ_k + xκ_k

    @inbounds for dd in 1:n_draws, h in range_positions, ddd in 1:2

        # If not last product in same position or last product, skip 
        if h < n_products && positions[h] == positions[h + 1]
            continue
        end

        # Reset for each draw
        prob_searches_given_draw = one(T)

        # Set lower bound for truncation of ̃w_k based on position 
        lb::T = if h < n_products # not yet last position 
            zd_h[h + 1] 
        else # no lower bound if last position 
            -T(MAX_NUMERICAL)
        end

        # Set upper bound for truncation of ̃w_k based on position
        ub::T = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
            T(MAX_NUMERICAL)
        elseif positions[h] == positions[min_position_discover] == positions[k] # if position last click same as where purchased -> no upper bound 
            T(MAX_NUMERICAL)
        else
            zd_h[h] 
        end

        # Fill 
        if ddd == 1 # e + xκ_k < ξ +xγ_k -> ̃w_k = u_k 
            e = rand_trunc(rng, dE, -one(T) * MAX_NUMERICAL, ξ + xγ_k - xκ_k)
            v = rand_trunc(rng, dV, lb - xβκ_k - e, ub - xβκ_k - e) # truncate v so that u_k in bounds
            u_k = xβκ_k + e + v
            z_k = xβγ_k + v + ξ 
            prob_draws_in_bounds = trunc_cdf(dE, -one(T) * MAX_NUMERICAL, ξ + xγ_k - xκ_k) *
                                   trunc_cdf(dV, lb - xβκ_k - e, ub - xβκ_k - e)
        else # e + xκ_k >= ξ +xγ_k -> ̃w_k = z_k
            e = rand_trunc(rng, dE, ξ + xγ_k - xκ_k, one(T) * MAX_NUMERICAL)
            v = rand_trunc(rng, dV, lb - ξ - xβγ_k, ub - ξ - xβγ_k) # truncate v so that z_k in bounds 
            u_k = xβκ_k + e + v
            z_k = xβγ_k + v + ξ 
            prob_draws_in_bounds = trunc_cdf(dE, ξ + xγ_k - xκ_k, one(T) * MAX_NUMERICAL) *
                                   trunc_cdf(dV, lb - ξ - xβγ_k, ub - ξ - xβγ_k)
        end

        # Get values only if last click not in initial awareness set 
        wt_k = min(u_k, z_k)
        w_k = positions[h] == 0 ? wt_k : min(wt_k, zd_h[h])

        # Get probability outside option P(u0 < min{wj,zd(j-1)})
        if with_outside_option
            prob_searches_given_draw *= cdf(dU0, w_k - β[end] * d.product_characteristics[i][1, end])
        end

        # Probabilities for other products
        for j in (1 + with_outside_option):h

            if xβγ[j] == typemin(T) # fill only if not already filled 
                xβ_j, xγ_j, xκ_j = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ, 
                    ixb, ixg, ixk) 
                xβγ[j] = xβ_j + xγ_j
                xβκ[j] = xβ_j + xκ_j
            end

            xβγ_j = xβγ[j]
            xβκ_j = xβκ[j]


            # searched, and discovered before the one that was purchased
            if consideration_set[j] && positions[j] < positions[k]
                prob_searches_given_draw *= prob_search_not_buy(
                    m, xβγ_j, xβκ_j, ξ, w_k, w_k, dE, dV)
            # searched, and discovered after the one that was purchased
            elseif consideration_set[j] && j != k && positions[j] >= positions[k]
                prob_searches_given_draw *= prob_search_not_buy(
                    m, xβγ_j, xβκ_j, ξ, wt_k, wt_k, dE, dV)
            # unsearched & discovered before 
            elseif positions[j] < positions[k]
                prob_searches_given_draw *= prob_not_search(m, w_k, xβγ_j + ξ, dV)
            # unsearched & at same time or after 
            elseif j != k && positions[j] >= positions[k]
                prob_searches_given_draw *= prob_not_search(m, wt_k, xβγ_j + ξ, dV)
            end
            if prob_searches_given_draw == 0
                prob_searches_given_draw = T(ALMOST_ZERO_NUMERICAL)
                break
            end
        end
        LL += prob_searches_given_draw * prob_draws_in_bounds
    end
    return lik_return_stable(LL/n_draws, return_log) 
end

"""
	function prob_search_not_buy(m::Union{SD, WM}, xβ_list::T, xβ_full::T, ξ::T, lb::T, ub::T, dE::Normal, dV::Normal)  where T 

Compute probability of searching without buying. This probability is given by P(xβ_list + ξ + ν_j >= lb ∩ xβ_full + ν_j + ε_j < ub). 
"""
@inline function prob_search_not_buy(m::Union{SD, WM}, xβ_list::T, xβ_full::T, ξ::T, lb::T, ub::T,
        dE::Normal{R1}, dV::Normal{R2}) where {T <: Real, R1 <: Real, R2 <: Real}
    σe = std(dE)
    σv = std(dV)

    a = σe > 0 ? (ub - xβ_full) / σe : T(MAX_NUMERICAL)
    b = σe > 0 ? -σv / σe : -T(MAX_NUMERICAL)
    Y = (lb - ξ - xβ_list) / σv

    P = cdf_n(a / sqrt(1 + b^2)) -
        bvncdf(a / sqrt(1 + b^2), Y, -b / sqrt(1 + b^2))

    return P::T
end

"""
	function prob_not_search(m::Union{SD, WM}, u, z_j, dV)

Compute probaility of not searching alternaitve, given chosen option has utility `u`. 
"""
@inline function prob_not_search(m::Union{SD, WM}, u::T, z_j::T, dV) where {T}
    return cdf(dV, u - z_j)
end

##  Demand functions based on partitioned probability space 
# function calculate_demand(m::Union{SD, WM}, d::DataSD, i, j, n_draws; kwargs...)

#     mc = construct_mapping_characteristics(m, d; kwargs...)

#     # Different functions depending on whether product is outside option or not
#     demand_j = if d.product_ids[i][j] == 0
#         return calculate_demand_outside_option(m, d, i, mc, n_draws; kwargs...)
#     else
#         return calculate_demand_product(m, d, i, j, mc, n_draws; kwargs...)
#     end

#     return demand_j
# end

# function calculate_demand_outside_option(m::SD{T}, d::DataSD, i, mapping_characteristics, n; kwargs...) where {T}

#     # note: unpacking things here and passing into function saves a lot of allocations
#     @unpack β, ξ, dE, dV, dU0 = m
#     @unpack γ, κ = m.information_structure

#     # Extract zdfun. If already applied as keyword, can save compilation time.
#     zd_h = get(kwargs, :zd_h, zeros(T, length(d.positions[i])))
#     if zd_h[1] == 0
#         zdfun = get(kwargs, :zdfun, get_functional_form(m.zdfun))
#         for h in eachindex(zd_h)
#             zd_h[h] = zdfun(m.Ξ, m.ρ, d.positions[i][h])
#         end
#     end

#     xβγ = get(kwargs, :xβγ, zeros(T, length(d.product_ids[i])))
#     xβκ = get(kwargs, :xβκ, zeros(T, length(d.product_ids[i])))
    
#     # Reset 
#     xβγ .= typemin(T) 
#     xβκ .= typemin(T) 


#     return calculate_demand_outside_option(m, d, i, n, β, γ, κ, ξ, zd_h, dE, dV, dU0,
#         mapping_characteristics, xβγ, xβκ; kwargs...)
# end


# function calculate_demand_outside_option(m::SD{T}, d::DataSD, i, n,
#     β, γ, κ, ξ, zd_h, dE, dV, dU0, mapping_characteristics, xβγ, xβκ; kwargs...) where T <: Real

#     n_products = length(d.product_ids[i])
#     positions = @views d.positions[i]
#     with_outside_option_dummy = d.product_ids[i][1] == 0
#     ixb = mapping_characteristics[1][i]
#     ixg = mapping_characteristics[2][i]
#     ixk = mapping_characteristics[3][i]

#     demand = zero(T)

#     β0 = with_outside_option_dummy ? β[end] : zero(T)

#     demand = zero(T)

#     rng = get_rng(kwargs)

#     for dd in 1:n, h in 2:n_products
#         # If not last product in same position or last product, skip 
#         if h < n_products && positions[h] == positions[h + 1]
#             continue
#         end

#         # Set lower bound for truncation based on position
#         lb::T = if h < n_products # not yet last position 
#             zd_h[h + 1] - β0
#         else # no lower bound if last position 
#             -T(MAX_NUMERICAL)
#         end

#         # Set upper bound for truncation based on position
#         ub::T = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
#             T(MAX_NUMERICAL)
#         else
#             zd_h[h] - β0 # Max accounts for case where nA0 < nd 
#         end

#         # Get probability of u0 in bounds and draw for u0
#         prob_u0_in_bounds = trunc_cdf(dU0, lb, ub)
#         u0_draw = rand_trunc(rng, dU0, lb, ub) + β0

#         # Initialize for probability
#         prob_buy_u0 = one(T)

#         for j in 2:h
#             if xβγ[j] == typemin(T) # fill only if not already filled 
#                 xβ_j, xγ_j, xκ_j = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ, 
#                     ixb, ixg, ixk) 
#                 xβγ[j] = xβ_j + xγ_j
#                 xβκ[j] = xβ_j + xκ_j
#             end

#             xβγ_j = xβγ[j]
#             xβκ_j = xβκ[j]
#             prob_buy_u0 *= prob_not_buy(m, xβγ_j, xβκ_j, ξ, u0_draw, dE, dV)
#         end

#         demand += prob_buy_u0 * prob_u0_in_bounds
#     end

#     conditional_on_search = get(kwargs, :conditional_on_search, false)
#     if conditional_on_search
#         throw(ArgumentError("Conditional on search not implemented for demand calculation of outside option."))
#     end

#     return demand / n
# end

# function calculate_demand_product(m::SD{T}, d::DataSD, i, k, mapping_characteristics, n; kwargs...) where {T}

#     # note: unpacking things here and passing into function saves a lot of allocations
#     @unpack β, ξ, dE, dV, dU0 = m
#     @unpack γ, κ = m.information_structure

#     # Extract zdfun. If already applied as keyword, can save compilation time.
#     zd_h = get(kwargs, :zd_h, zeros(T, length(d.positions[i])))
#     if zd_h[1] == 0
#         zdfun = get(kwargs, :zdfun, get_functional_form(m.zdfun))
#         for h in eachindex(zd_h)
#             zd_h[h] = zdfun(m.Ξ, m.ρ, d.positions[i][h])
#         end
#     end

#     if zd_h[k] <= -T(MAX_NUMERICAL)
#         return zero(T)
#     end

#     xβγ = get(kwargs, :xβγ, zeros(T, length(d.product_ids[i])))
#     xβκ = get(kwargs, :xβκ, zeros(T, length(d.product_ids[i])))

#     # Reset 
#     xβγ .= typemin(T) 
#     xβκ .= typemin(T) 

#     return calculate_demand_product(m, d, i, k, n, β, γ, κ,  ξ, zd_h, dE, dV, dU0,
#         mapping_characteristics, xβγ, xβκ; kwargs...)
# end

# function calculate_demand_product(m::SD{T}, d::DataSD, i, k, n,
#         β::Vector{T}, γ::Vector{T}, κ::Vector{T}, ξ::T, zd_h::Vector{T}, dE::Distribution,
#         dV::Distribution, dU0::Distribution, mapping_characteristics,
#         xβγ, xβκ;  kwargs...) where {T <: Real}
        
#     n_products = length(d.product_ids[i])
#     positions = @views d.positions[i]
#     with_outside_option_dummy = d.product_ids[i][1] == 0
#     ixb = mapping_characteristics[1][i]
#     ixg = mapping_characteristics[2][i]
#     ixk = mapping_characteristics[3][i]

#     demand = zero(T)

#     # xβ of purchased 
#     xβ_k, xγ_k, xκ_k = @views construct_util_parts(d.product_characteristics, i, k, β, γ, 
#         κ, ixb, ixg, ixk)
#     xβγ_k = xβ_k + xγ_k
#     xβκ_k = xβ_k + xκ_k
#     rng = get_rng(kwargs)

#     max_n_products =  get(kwargs, :max_n_products, n_products)

#     for dd in 1:n, h in 1:max_n_products, ddd in 1:2


#         if h > n_products || h < k 
#             rand(rng) ; rand(rng) # keeps seed fixed
#             continue
#         end

#         # If not last product in same position or last product, skip 
#         if h < n_products && positions[h] == positions[h + 1]
#             continue
#         end

#         # Reset for each draw
#         prob_purchase_k = one(T)

#         # Set lower bound for truncation based on position 
#         lb::T = if h < n_products # not yet last position 
#             zd_h[h + 1] 
#         else # no lower bound if last position 
#             -T(MAX_NUMERICAL)
#         end

#         # Set upper bound for truncation based on position
#         ub::T = if positions[h] == positions[k] # first no upper bound if last click in initial awareness set (position 0)
#             T(MAX_NUMERICAL)
#         else
#             zd_h[h]  # Max accounts for case where nA0 < nd 
#         end

#         # Fill 
#         if ddd == 1 # e + xβ_k_detail < ξ -> ̃w_k = u_k 
#             e = rand_trunc(rng, dE, -one(T) * MAX_NUMERICAL, ξ + xγ_k - xκ_k)
#             v = rand_trunc(rng, dV, lb - xβκ_k - e, ub - xβκ_k - e) # truncate v so that u_k in bounds
#             u_k = xβκ_k + e + v
#             z_k = xβγ_k + v + ξ 
#             prob_draws_in_bounds = trunc_cdf(dE, -one(T) * MAX_NUMERICAL, ξ + xγ_k - xκ_k) *
#                                    trunc_cdf(dV, lb - xβκ_k - e, ub - xβκ_k - e)
#         else # e + xβ_k_detail >= ξ  -> ̃w_k = z_k
#             e = rand_trunc(rng, dE, ξ + xγ_k - xκ_k, one(T) * MAX_NUMERICAL)
#             v = rand_trunc(rng, dV, lb - ξ - xβγ_k, ub - ξ - xβγ_k) # truncate v so that z_k in bounds 
#             u_k = xβκ_k + e + v
#             z_k = xβγ_k + v + ξ 
#             prob_draws_in_bounds = trunc_cdf(dE, ξ + xγ_k - xκ_k, one(T) * MAX_NUMERICAL) *
#                                    trunc_cdf(dV, lb - ξ - xβγ_k, ub - ξ - xβγ_k)
#         end

#         # Get values only if last click not in initial awareness set 
#         wt_k = min(u_k, z_k)
#         w_k = positions[h] == 0 ? wt_k : min(wt_k, zd_h[h])

#         if with_outside_option_dummy
#             prob_purchase_k *= cdf(dU0, w_k - β[end])
#         end

#         # P(not buy j) for other products
#         for j in (with_outside_option_dummy + 1):h
#             if j == k
#                 continue
#             end
#             if xβγ[j] == typemin(T) # fill only if not already filled 
#                 xβ_j, xγ_j, xκ_j = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ, 
#                     ixb, ixg, ixk) 
#                 xβγ[j] = xβ_j + xγ_j
#                 xβκ[j] = xβ_j + xκ_j
#             end

#             xβγ_j = xβγ[j]
#             xβκ_j = xβκ[j]

#             if positions[j] < positions[k]
#                 prob_purchase_k *= prob_not_buy(m, xβγ_j, xβκ_j, ξ, w_k, dE, dV)
#             else
#                 prob_purchase_k *= prob_not_buy(m, xβγ_j, xβκ_j, ξ, wt_k, dE, dV)
#             end
#         end

#         demand += prob_purchase_k * prob_draws_in_bounds
#     end

#     conditional_on_search = get(kwargs, :conditional_on_search, false)
#     if conditional_on_search
#         demand = demand / lik_no_searches(m, zd_h, ξ, β, γ, κ, dV, dU0, mapping_characteristics, d, i, n, true, rng, false, (xβγ, xβκ))
#     end

#     return demand / n
# end

"""
	function prob_no_buy(m::Union{SD, WM}, xβ::T, ξ::T, u::T, dE::Normal, dV::Normal)  where T 

Compute probability of not buying given chosen utility has utility `u`. This probability is given by P(xβ + v_j + min(ξ, xβ_list + ε_j) <= u)
"""

@inline function prob_not_buy(m::Union{SD, WM}, xβ_list::T, xβ_full::T,  ξ::T, u::T,
        dE::Normal, dV::Normal) where {T}
    σe = std(dE)
    σv = std(dV)

    a = (u - xβ_full) / σe
    b = -σv / σe
    Y = (u - ξ - xβ_list) / σv

    P = cdf(dV, u - ξ - xβ_list) + cdf(Normal(), a / sqrt(1 + b^2)) -
        bvncdf(a / sqrt(1 + b^2), Y, -b / sqrt(1 + b^2))
    return P
end

# ######################################################################
# ## P(click but not buy )
# ######################################################################

# @inline function cdf_ZWU(m::Union{SD3,WM},xb::T,xi::T,K::T,
# 							dE::Normal,dV::Normal)  where T 

# 	σe = std(dE) 
# 	σv = std(dV)

# 	a = σe > 0 ? (K-xb)/σe : one(T)*10000000
# 	b = σe > 0 ? -σv/σe : -one(T)*10000000
# 	Y = (K - xi - xb) / σv

# 	P = cdf_n(a/sqrt(1+b^2)) - 
# 				bvncdf(a/sqrt(1+b^2), Y, -b/sqrt(1+b^2)) 
# 	# Note: Second part is bivariate normal cdf from bivariate.jl
# 	return P::T
# end

# @inline function cdf_ZWU(m::Union{SD3,WM},xb::T,xi::T,lb::T,ub::T,
# 							dE::Normal,dV::Normal) where T 

# 	σe = std(dE) 
# 	σv = std(dV)

# 	a = σe > 0 ? (ub-xb)/σe : one(T)*10000000
# 	b = σe > 0 ? -σv/σe : -one(T)*10000000
# 	Y = (lb -xi -xb) / σv

# 	P = cdf_n(a/sqrt(1+b^2)) - 
# 			bvncdf(a/sqrt(1+b^2), Y, -b/sqrt(1+b^2))

# 	return P::T
# end

# ######################################################################
# ## P( not buy ) (with or without click)
# ######################################################################
# @inline function cdf_ZWnb(m::Union{SD3,WM},xb::T,xi::T,z::T,
# 								dE::Normal,dV::Normal) where T 
# 	σe = std(dE) 
# 	σv = std(dV)

# 	a = (z-xb)/σe
# 	b = -σv/σe
# 	Y = (z - xi - xb) / σv

# 	P = cdf(dV,z-xi-xb) + cdf(Normal(),a/sqrt(1+b^2)) - StructSearch.bvncdf(a/sqrt(1+b^2),Y,-b/sqrt(1+b^2)) 
# 	return P 
# end
