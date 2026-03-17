"""
Abstract type for the *Weitzman* (WM) model. This type is a base type for all models that are subtypes of the WM model.
"""
abstract type WMModel <: SDModel end

"""
    WM{T} <: WMModel

Weitzman model with the following parameterization:

- uᵢⱼ = xⱼβ + xⱼκ + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ(h) = xⱼβ + xⱼγ + ξ(h) + νᵢⱼ
- uᵢ₀ = x₀'β + ηᵢ, ηᵢ ~ dU0
- ξ(h) = zsfun(ξ, ρ, h)

The specification of `xⱼβ`, `xⱼκ`, and `xⱼγ` is determined by `information_structure`.

# Fields
- `β::Vector{T}`: Preference weights.
- `ξ::T`: Baseline search value.
- `ρ::Union{T, Vector{T}}`: Parameters governing decrease of ξ across positions.
- `dE::Distribution`: Distribution of εᵢⱼ.
- `dV::Distribution`: Distribution of νᵢⱼ.
- `dU0::Distribution`: Distribution of ηᵢ.
- `zsfun::String`: Functional form f(ξ, ρ, h) for the search value at position h. Available options: `""` (constant), `"linear"`, `"log"`, `"exp"`, `"linear-k"`, `"log-k"` (where `k` is an integer).
- `information_structure::InformationStructureSpecification{T}`: Information structure specification. See `InformationStructureSpecification`.
- `cs::Union{Vector{Vector{T}}, Nothing}`: Search costs per product, populated by `calculate_costs!`. `nothing` until computed.
- `heterogeneity::HeterogeneitySpecification`: Heterogeneity specification. Defaults to homogeneous model.
"""
@with_kw mutable struct WM{T} <: WMModel where {T <: Real}
    β::Vector{T}
    ξ::T
    ρ::Union{T, Vector{T}}
    dE::Distribution
    dV::Distribution
    dU0::Distribution
    zsfun::String

    # Parameters with defaults
    information_structure::InformationStructureSpecification{T} = InformationStructureSpecification(length(β))
    heterogeneity::HeterogeneitySpecification{T} = HeterogeneitySpecification()
    cs::Union{Vector{Vector{T}}, Nothing} = nothing

    @assert length(information_structure.γ) == length(β) "Length of γ must be equal to length of β."
    @assert length(information_structure.κ) == length(β) "Length of κ must be equal to length of β."

end
function ==(m1::WM, m2::WM)
    return isequal(m1.β, m2.β) &&
           isequal(m1.ρ, m2.ρ) &&
           isequal(m1.ξ, m2.ξ) &&
           isequal(m1.dE, m2.dE) &&
           isequal(m1.dV, m2.dV) &&
           isequal(m1.dU0, m2.dU0) &&
           isequal(m1.zsfun, m2.zsfun) &&
           isequal(m1.cs, m2.cs) &&
            isequal(m1.information_structure, m2.information_structure) &&
            isequal(m1.heterogeneity, m2.heterogeneity)
end

function WM(β, ξ, ρ, dE, dV, dU0, zsfun,
    information_structure, heterogeneity, cs)
    T = eltype(ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    return WM(convert(Vector{T}, β), ξ, ρ, dE, dV, dU0, zsfun,
        information_structure, heterogeneity, cs)
end

# this one for some reason is necessary as Parameters.jl does not seem to enforce T for β
function WM(β::Vector{Any}, ξ, ρ, dE, dV, dU0, zsfun; cs = nothing,
        information_structure = InformationStructureSpecification(),
        heterogeneity = HeterogeneitySpecification())
    T = eltype(ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    return WM(convert(Vector{T}, β), ξ, ρ, dE, dV, dU0, zsfun,
        information_structure, heterogeneity, cs)
end

function WM(β, ξ, ρ, dE, dV, dU0, zsfun; cs = nothing,
        information_structure = InformationStructureSpecification(),
        heterogeneity = HeterogeneitySpecification())
    T = eltype(ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    return WM(convert(Vector{T}, β), ξ, ρ, dE, dV, dU0, zsfun,
        cs, information_structure, heterogeneity)
end
function convert_cs(c, T)
    if isnothing(c)
        return nothing
    elseif c isa Vector
        return convert(Vector{T}, c)
    elseif c isa Matrix where R
        return convert(Matrix{T}, c)
    else
        throw(ArgumentError("cs must be either nothing, a scalar or a vector of type T."))
    end
end


# Functions from SDCore to WM
function SDCore(m::WM)
    # Update heterogeneity specification to map ρ into ξρ
    h = deepcopy(m.heterogeneity)
    if :ρ ∈ keys(h.parameters_with_observed_heterogeneity)
        h.parameters_with_observed_heterogeneity[:ξρ] = h.parameters_with_observed_heterogeneity[:ρ]
        delete!(h.parameters_with_observed_heterogeneity, :ρ)
    end

    cd = isnothing(m.cs) ? nothing : fill(0.0, length(m.cs))

    m_new =  SDCore(; β = m.β, Ξ = Inf, ρ = [-1e-100], ξ = m.ξ, ξρ = m.ρ, cd = cd,
        cs = m.cs, dE = m.dE, dV = m.dV, dU0 = m.dU0,
        dW = Normal(0, 0), zdfun = "", zsfun = m.zsfun,
        information_structure = m.information_structure,
        heterogeneity = h)
    update_heterogeneity_specification!(m_new)

    return m_new
end

function calculate_welfare(
        model::WM, data::DataSD, n_sim; method = "effective_values", kwargs...)
    calculate_welfare(SDCore(model), data, n_sim; method = method, kwargs...)
end

function calculate_revenues(m::WM, d::DataSD, kprice, n_draws; kwargs...)
    return calculate_revenues(SDCore(m), d::DataSD, kprice, n_draws; kwargs...)
end

function calculate_costs!(m::WM, d, n_draws...;
        force_recompute = true,
        kwargs...)
    m1 = SDCore(m)
    calculate_costs!(m1, d, n_draws...; force_recompute, kwargs...)

    m.cs = m1.cs

    return nothing
end

function generate_data(m::WM, n_consumers, n_sessions_per_consumer; kwargs...)
    generate_data(SDCore(m), n_consumers, n_sessions_per_consumer; kwargs...)
end
generate_data(m::WM, data::DataSD; kwargs...) = generate_data(SDCore(m), data; kwargs...)

function calculate_fit_measures(m::WM, data::DataSD, n_sim; kwargs...)
    calculate_fit_measures(SDCore(m), data, n_sim; kwargs...)
end

# Estimation
function prepare_arguments_likelihood(m::WM, e::Estimator, d::DataSD; kwargs...)

    # Get functional forms
    zsfun = get_functional_form(m.zsfun)

    # Get maximum number of products
    data_arguments = prepare_data_arguments_likelihood(d)

    # Keep fixed seed: either random or provided by kwargs
    rng = get_rng(kwargs)
    seed = get(kwargs, :seed, rand(rng, 1:(10^9)))

    # Mapping characteristics to parameters
    mapping_characteristics = construct_mapping_characteristics(m, d; kwargs...)

    # Draws for unobserved heterogeneity
    ni_pts_whts = prepare_draws_unobserved_heterogeneity(m,
        e.numerical_integration_method_heterogeneity, d, rng)

    return data_arguments..., nothing, zsfun, rng, seed,
        mapping_characteristics, ni_pts_whts
end

# Vectorize parameters
function vectorize_parameters(m::WM; kwargs...)
    # Default estimate all parameters
    θ = if !haskey(kwargs, :fixed_parameters)
        θ = vcat(m.β[vcat(m.information_structure.indices_characteristics_β_union, end)],
            m.information_structure.γ[m.information_structure.indices_characteristics_γ_union],
            m.information_structure.κ[m.information_structure.indices_characteristics_κ_union],
            m.ξ, m.ρ)
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
            if :ξ ∉ fixed_parameters
                θ = vcat(θ, m.ξ)
            end
            if :ρ ∉ fixed_parameters
                θ = vcat(θ, m.ρ)
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

function construct_model_from_pars(θ::Vector{T}, m::WM; kwargs...) where {T <: Real}

    # Extract parameters from vector, some may be fixed through kwargs
    β, γ, κ, _, _, ξ, ρ, ind_last_par = extract_parameters(m, θ; kwargs...)
    ψ, U, ind_last_par = extract_heterogeneity_parameters(m, θ, ind_last_par; kwargs...)
    dE, dV, dU0, _, ind_last_par = extract_distributions(m, θ, ind_last_par; kwargs...)

    information_structure = deepcopy(m.information_structure)
    information_structure.γ = γ
    information_structure.κ = κ

    heterogeneity = construct_new_heterogeneity_specification(m, ψ, U)


    # Construct model from parameters
    m_new = WM{T}(; β, ξ, ρ, dE, dV, dU0, zsfun = m.zsfun,
        information_structure, heterogeneity)

    if get(kwargs, :return_ind_last_par, false)
        return m_new, ind_last_par
    else
        return m_new
    end
end

function extract_parameters(m::WM, θ::Vector{T}; kwargs...) where {T <: Real}
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

        ξ = θ[ind_current]
        ind_current += 1
        ρ = θ[ind_current:(ind_current + n_ρ - 1)]
        ind_current += n_ρ
        return β, γ, κ, nothing, nothing, ξ, ρ, ind_current
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

    γ = T.(m.information_structure.γ)
    if :γ ∉ fixed_parameters
        n_γ = length(m.information_structure.γ)
        for i in 1:n_γ
            γ[i] = θ[ind_current]
            ind_current += 1
        end
    end

    κ = T.(m.information_structure.κ)
    if :κ ∉ fixed_parameters
        n_κ = length(m.information_structure.κ)
        for i in 1:n_κ
            κ[i] = θ[ind_current]
            ind_current += 1
        end
    end

    ξ = if :ξ ∈ fixed_parameters
        T(m.ξ)
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
    return β, γ, κ, nothing, nothing, ξ, ρ, ind_current
end

function lik_no_searches(m::WM, zd, ξj::Vector{T}, β::Vector{T}, γ, κ,
        dV, dU0,
        mapping_characteristics, d::DataSD, i::Int, n_draws, complement, rng,
        return_log, cache) where {T <: Real}

    n_products = length(d.product_ids[i])
    ixb = mapping_characteristics[1][i]
    ixg = mapping_characteristics[2][i]
    ixk = mapping_characteristics[3][i]

    xβγ, _ = unpack_cache(cache, β)
    xβγ .= typemin(T) # reset

    LL = zero(T)

    for dd in 1:n_draws

        # Unbounded draw for u0
        u0_draw = rand(rng, dU0) + β[end] * d.product_characteristics[i][1, end]

        # Initialize for probability
        prob_no_search_given_draw = one(T)

        # Loop over products up to one discovered
        for j in 2:n_products
            if xβγ[j] == typemin(T) # fill only if not already filled
                xβ_j, xγ_j, _ = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ,
                    ixb, ixg, ixk; no_κ = true)
                xβγ[j] = xβ_j + xγ_j
            end

            zj = xβγ[j] + ξj[j]
            prob_no_search_given_draw *= prob_not_search(m, u0_draw, zj, dV)

            if prob_no_search_given_draw == 0
                prob_no_search_given_draw = T(ALMOST_ZERO_NUMERICAL)  # adding this helps AD
                break
            end
        end
        if complement
            LL += (1 - prob_no_search_given_draw)
        else
            LL += prob_no_search_given_draw
        end
    end

    return lik_return_stable(LL/n_draws, return_log)
end

function lik_search_no_purchase(m::WM, zd, ξj::Vector{T}, β::Vector{T}, γ, κ,
        dE, dV, dU0,
        mapping_characteristics, d::DataSD, i::Int, n_draws, rng,
        return_log, cache) where {T <: Real}
    n_products = length(d.product_ids[i])
    consideration_set = @views d.consideration_sets[i]
    ixb = mapping_characteristics[1][i]
    ixg = mapping_characteristics[2][i]
    ixk = mapping_characteristics[3][i]

    xβγ, xβκ = unpack_cache(cache, β)
    xβγ .= typemin(T) # reset
    xβκ .= typemin(T) # reset


    LL = zero(T)

    for dd in 1:n_draws

        # Unbounded draw for u0
        u0_draw = rand(rng, dU0) + β[end] * d.product_characteristics[i][1, end]

        prob_searches_given_draw = one(T)

        for j in 2:n_products
            if consideration_set[j] # searched item
                if xβγ[j] == typemin(T) # fill only if not already filled
                    xβ_j, xγ_j, xκ_j = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ,
                        ixb, ixg, ixk)
                    xβγ[j] = xβ_j + xγ_j
                    xβκ[j] = xβ_j + xκ_j
                end

                prob_searches_given_draw *= prob_search_not_buy(
                    m, xβγ[j], xβκ[j], ξj[j], u0_draw, u0_draw, dE, dV)
            else
                if xβγ[j] == typemin(T) # fill only if not already filled
                    xβ_j, xγ_j, _ = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ,
                        ixb, ixg, ixk; no_κ = true)
                    xβγ[j] = xβ_j + xγ_j
                end

                prob_searches_given_draw *= prob_not_search(m, u0_draw, xβγ[j] + ξj[j], dV)
            end

            if prob_searches_given_draw == 0
                prob_searches_given_draw = T(ALMOST_ZERO_NUMERICAL)  # adding this helps AD
                break
            end
        end

        LL += prob_searches_given_draw
    end
    return lik_return_stable(LL/n_draws, return_log)
end

function lik_purchase(m::WM, zd, ξj::Vector{T}, β::Vector{T}, γ, κ,
        dE, dV, dU0,
        mapping_characteristics, d::DataSD, i::Int, n_draws, rng,
        return_log, cache) where {T <: Real}
    n_products = length(d.product_ids[i])
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

    # xb of purchased
    xβ_k, xγ_k, xκ_k = @views construct_util_parts(d.product_characteristics, i, k, β, γ, κ,
                    ixb, ixg, ixk)
    xβγ_k = xβ_k + xγ_k
    xβκ_k = xβ_k + xκ_k

    for dd in 1:n_draws, ddd in 1:2

        # Reset for each draw
        prob_searches_given_draw = one(T)

        # Fill
        if ddd == 1 # e + xβ_k_detail  < ξ
            e = rand_trunc(rng, dE, -one(T) * MAX_NUMERICAL, ξj[k] + xγ_k - xκ_k)
            v = rand(rng, dV)
            u_k = xβκ_k + e + v
            z_k = xβγ_k + v + ξj[k] # this way v draw still stored in u_k
            prob_draw_in_bounds = trunc_cdf(dE, -one(T) * MAX_NUMERICAL, ξj[k] + xγ_k - xκ_k)
        else # e + xβ_k_detail  >= ξ
            e = rand_trunc(rng, dE, ξj[k] + xγ_k - xκ_k, one(T) * MAX_NUMERICAL)
            v = rand(rng, dV)
            u_k = xβκ_k + e + v
            z_k = xβγ_k + v + ξj[k] # this way v draw still stored in u_k
            prob_draw_in_bounds = trunc_cdf(dE, ξj[k] + xγ_k - xκ_k, one(T) * MAX_NUMERICAL)
        end

        # Get tilde w_k
        wt_k = min(u_k, z_k)

        # Get probability outside option P(u0 < min{wj,zd(j-1)})
        if with_outside_option
            prob_searches_given_draw *= cdf(dU0, wt_k - β[end] * d.product_characteristics[i][1, end])
        end

        # Probabilities for other products
        for j in 2:n_products
            if xβγ[j] == typemin(T) # fill only if not already filled
                xβ_j, xγ_j, xκ_j = @views construct_util_parts(d.product_characteristics, i, j, β, γ, κ,
                    ixb, ixg, ixk)
                xβγ[j] = xβ_j + xγ_j
                xβκ[j] = xβ_j + xκ_j
            end

            xβγ_j = xβγ[j]
            xβκ_j = xβκ[j]

            # searched
            if consideration_set[j] && j != k
                prob_searches_given_draw *= prob_search_not_buy(
                    m, xβγ_j, xβκ_j, ξj[j], wt_k, wt_k, dE, dV)
            # unsearched
            elseif j != k
                prob_searches_given_draw *= prob_not_search(m, wt_k, xβγ_j + ξj[j], dV)
            end
            if prob_searches_given_draw == 0
                prob_searches_given_draw = T(1e-100)
                break
            end
        end
        LL += prob_searches_given_draw * prob_draw_in_bounds
    end
    return lik_return_stable(LL/n_draws, return_log)
end


