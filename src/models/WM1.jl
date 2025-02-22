"""
Abstract type for the *Weitzman* (WM) model. This type is a base type for all models that are subtypes of the WM model. 
"""
abstract type WM <: SD end

"""
*Weitzman model* `WM1{T} <: WM` with the following parameterization: 
- uᵢⱼ = xⱼ'β + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ = xⱼ'β + ξ(h) + νᵢⱼ 
- uᵢ₀ = x₀'β + ηᵢ , ηᵢ ~ dU0
- ξ(h) = zsfun(ξ, ρ, pos)


# Fields:  
- `β::Vector{T}`: vector of preference weights. 
- `cs`: search costs. Initialized as nothing by to avoid computational cost. Can be updated through `calculate_costs!(m, data; kwargs...)`. 
- `ξ::T`: baseline ξ.
- `ρ::Vector{T}`: parameters governing decrease of ξ across positions.
- `dE::Distribution`: distribution of εᵢⱼ.
- `dV::Distribution`: distribution of νᵢⱼ.
- `dU0::Distribution`: distribution of ηᵢ. 
- `zsfun::String`: select functional form f(ξ, ρ, h) that determines the search value in position h. 
- `unobserved_heterogeneity::Dict`: dictionary of unobserved heterogeneity parameters and options. Currently not used. 
"""
@with_kw mutable struct WM1{T} <: WM where {T <: Real}
    β::Vector{T}
    ξ::T
    ρ::Vector{T}
    cs::Union{T, Nothing} = nothing
    cs_h::Union{Vector{T}, Nothing} = nothing
    dE::Distribution
    dV::Distribution
    dU0::Distribution
    zsfun::String
    unobserved_heterogeneity::Dict = Dict()
end

# Functions from SDCore to WM1 
function SDCore(m::WM1)
    SDCore(; β = m.β, Ξ = Inf, ρ = [-1e-100], ξ = m.ξ, ξρ = m.ρ, cd = 0.0,
        cs = m.cs, cs_h = m.cs_h, dE = m.dE, dV = m.dV, dU0 = m.dU0,
        dW = Normal(0, 0), zdfun = "linear", zsfun = m.zsfun,
        unobserved_heterogeneity = m.unobserved_heterogeneity)
end

function calculate_welfare(
        model::WM1, data::DataSD, n_sim; method = "effective_values", kwargs...)
    calculate_welfare(SDCore(model), data, n_sim; method = method, kwargs...)
end

function calculate_costs!(m::WM1, d;
        force_recompute = true,
        cd_kwargs...)
    m1 = SDCore(m)
    calculate_costs!(m1, d, 1; force_recompute, cd_kwargs...)

    m.cs = m1.cs
    m.cs_h = m1.cs_h

    return nothing
end

function calculate_costs!(m::WM1, d::DataSD, h::Int;
        force_recompute = true,
        cd_kwargs...)
    calculate_costs!(m, d; force_recompute, cd_kwargs...)
    return nothing
end

function generate_data(m::WM1, n_consumers, n_sessions_per_consumer; kwargs...)
    generate_data(SDCore(m), n_consumers, n_sessions_per_consumer; kwargs...)
end
generate_data(m::WM1, data::DataSD; kwargs...) = generate_data(SDCore(m), data; kwargs...)

function evaluate_fit(m::WM1, data::DataSD, n_sim; kwargs...)
    evaluate_fit(SDCore(m), data, n_sim; kwargs...)
end

# Estimation 
function prepare_arguments_likelihood(m::WM1, estimator::Estimator, d::DataSD; kwargs...)

    # Get functional forms 
    zsfun = get_functional_form(m.zsfun)

    # Get maximum number of products
    data_arguments = prepare_data_arguments_likelihood(d)

    # Keep fixed seed: either random or provided by kwargs 
    seed = get(kwargs, :seed, rand(1:(10^9)))

    return data_arguments..., nothing, zsfun, seed
end

# Vectorize parameters 
function vectorize_parameters(m::WM1; kwargs...)
    # Default estimate all parameters 
    θ = if !haskey(kwargs, :fixed_parameters)
        θ = vcat(m.β, m.ξ, m.ρ)
    else
        fixed_parameters = get(kwargs, :fixed_parameters, nothing)
        if !isnothing(fixed_parameters)
            θ = eltype(m.β)[]
            if !fixed_parameters[1]
                θ = vcat(θ, m.β)
            end
            if !fixed_parameters[2]
                θ = vcat(θ, m.ξ)
            end
            if !fixed_parameters[3]
                θ = vcat(θ, m.ρ)
            end
        end
        θ
    end

    θ = add_distribution_parameters(m, θ, kwargs)

    return θ
end

function construct_model_from_pars(θ::Vector{T}, m::WM1; kwargs...) where {T <: Real}

    # Extract parameters from vector, some may be fixed through kwargs 
    β, _, _, ξ, ρ, ind_last_par = extract_parameters(m, θ; kwargs...)
    dE, dV, dU0 = extract_distributions(m, θ, ind_last_par; kwargs...)

    # Construct model from parameters 
    m_new = WM1{T}(; β, ξ, ρ, dE, dV, dU0, zsfun = m.zsfun)

    return m_new
end

function extract_parameters(m::WM1, θ::Vector{T}; kwargs...) where {T <: Real}
    n_beta = length(m.β)
    n_ρ = length(m.ρ)

    # track where in parameter vector we are and move it. 
    ind_current = 1

    # Default: estimate all parameters
    if !haskey(kwargs, :fixed_parameters)
        β = θ[1:n_beta]
        ind_current += n_beta
        ξ = θ[ind_current]
        ind_current += 1
        ρ = θ[ind_current:(ind_current + n_ρ - 1)]
        ind_current += n_ρ
        return β, nothing, nothing, ξ, ρ, ind_current
    end

    # If keyword supplied, don't estimate parameters indicated in fixed_parameters
    fixed_parameters = get(kwargs, :fixed_parameters, nothing)
    β = if fixed_parameters[1]
        T.(m.β)
    else
        ind_current += n_beta
        θ[1:n_beta]
    end
    ξ = if fixed_parameters[2]
        T(m.ξ)
    else
        ind_current += 1
        θ[ind_current - 1]
    end
    ρ = if fixed_parameters[4]
        T.(m.ρ)
    else
        ind_current += n_ρ
        θ[ind_current:(ind_current + n_ρ - 1 - 1)]
    end
    return β, nothing, nothing, ξ, ρ, ind_current
end

function ll_no_searches(m::WM1, zd, ξj::Vector{T}, β::Vector{T}, dV, dU0,
        d::DataSD, i::Int, n_draws, complement) where {T <: Real}
    n_products = length(d.product_ids[i])
    with_outside_option_dummy = d.product_ids[i][1] == 0

    LL = zero(T)

    for dd in 1:n_draws

        # Unbounded draw for u0 
        u0_draw = rand(dU0) + (with_outside_option_dummy ? β[end] : zero(T))

        # Initialize for probability
        prob_no_search_given_draw = one(T)

        # Loop over products up to one discovered 
        for j in 2:n_products
            zs_j = @views d.product_characteristics[i][j, :]' * β + ξj[j]
            prob_no_search_given_draw *= prob_not_search(m, u0_draw, zs_j, dV)

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

    # println(log(LL / n_draws))
    return log(max(T(ALMOST_ZERO_NUMERICAL), LL / n_draws))
end

function ll_search_no_purchase(m::WM1, zd, ξj::Vector{T}, β::Vector{T}, dE, dV,
        dU0, d::DataSD, i::Int, n_draws) where {T <: Real}
    n_products = length(d.product_ids[i])
    with_outside_option_dummy = d.product_ids[i][1] == 0
    consideration_set = @views d.consideration_sets[i]

    LL = zero(T)

    for dd in 1:n_draws

        # Unbounded draw for u0 
        u0_draw = rand(dU0) + (with_outside_option_dummy ? β[end] : zero(T))

        prob_searches_given_draw = one(T)

        for j in 2:n_products
            xβ = @views d.product_characteristics[i][j, :]' * β

            if consideration_set[j] # searched item 
                prob_searches_given_draw *= prob_search_not_buy(
                    m, xβ, ξj[j], u0_draw, u0_draw, dE, dV)
            else
                prob_searches_given_draw *= prob_not_search(m, u0_draw, xβ + ξj[j], dV)
            end

            if prob_searches_given_draw == 0
                prob_searches_given_draw = T(1e-100)
                break
            end
        end

        LL += prob_searches_given_draw
    end
    return log(max(T(ALMOST_ZERO_NUMERICAL), LL / n_draws))
end

function ll_purchase(m::WM1, zd, ξj::Vector{T}, β::Vector{T}, dE, dV,
        dU0, d::DataSD, i::Int, n_draws) where {T <: Real}
    n_products = length(d.product_ids[i])
    with_outside_option_dummy = d.product_ids[i][1] == 0
    consideration_set = @views d.consideration_sets[i]

    LL = zero(T)

    k = d.purchase_indices[i] # index of purchased product

    for dd in 1:n_draws, ddd in 1:2

        # Reset for each draw
        prob_searches_given_draw = one(T)

        # xb of purchased 
        xβ_k = @views d.product_characteristics[i][k, :]' * β

        # Fill 
        if ddd == 1 # e < ξ
            e = rand_trunc(dE, -one(T) * MAX_NUMERICAL, ξj[k])
            u_k = xβ_k + e + rand(dV)
            z_k = u_k - e + ξj[k] # this way v draw still stored in z_k
            prob_draw_in_bounds = trunc_cdf(dE, -one(T) * MAX_NUMERICAL, ξj[k])
        else # e >= ξ
            e = rand_trunc(dE, ξj[k], one(T) * MAX_NUMERICAL)
            z_k = xβ_k + ξj[k] + rand(dV)
            u_k = z_k - ξj[k] + e # this way v draw still stored in z_k 
            prob_draw_in_bounds = trunc_cdf(dE, ξj[k], one(T) * MAX_NUMERICAL)
        end

        # Get tilde w_k
        wt_k = min(u_k, z_k)

        # Get probability outside option P(u0 < min{wj,zd(j-1)})
        if with_outside_option_dummy
            prob_searches_given_draw *= cdf(dU0, wt_k - β[end])
        else
            prob_searches_given_draw *= cdf(dU0, wt_k)
        end

        # Probabilities for other products
        for j in 2:n_products
            xβ_j = @views d.product_characteristics[i][j, :]' * β

            # searched 
            if consideration_set[j] && j != k
                prob_searches_given_draw *= prob_search_not_buy(
                    m, xβ_j, ξj[j], wt_k, wt_k, dE, dV)
                # unsearched 
            elseif j != k
                prob_searches_given_draw *= cdf(dV, wt_k - ξj[j] - xβ_j)
            end
            if prob_searches_given_draw == 0
                prob_searches_given_draw = T(1e-100)
                break
            end
        end
        LL += prob_searches_given_draw * prob_draw_in_bounds
    end
    return log(max(T(ALMOST_ZERO_NUMERICAL), LL / n_draws))
end

function calculate_demand_outside_option(m::WM1, d::DataSD, i, n; kwargs...)
    T = eltype(m.β)

    positions = @views d.positions[i]
    product_characteristics = @views d.product_characteristics[i]
    n_products = length(d.product_ids[i])

    # Extract zs. If already applied as keyword, can save compilation time.
    ξj = get(kwargs, :ξj, zeros(T, length(positions)))
    if ξj[1] == 0
        zsfun = get_functional_form(m.zsfun)
        for h in eachindex(ξj)
            ξj[h] = zsfun(m.ξ, m.ρ, positions[h])
        end
    end

    demand = zero(T)

    for dd in 1:n

        # Get probability of u0 in bounds and draw for u0
        u0_draw = rand(m.dU0) + m.β[end]

        # Initialize for probability
        prob_buy_u0 = one(T)

        for j in 2:n_products
            xβ = @views product_characteristics[j, :]' * m.β
            prob_buy_u0 *= prob_not_buy(m, xβ, ξj[j], u0_draw, m.dE, m.dV)
        end

        demand += prob_buy_u0 
    end

    conditional_on_search = get(kwargs, :conditional_on_search, false)
    if conditional_on_search
        throw(ArgumentError("Conditional on search not implemented for demand calculation of outside option."))
    end

    return demand / n
end

function calculate_demand_product(m::WM1{T}, d::DataSD, i, k, n; kwargs...) where {T}

    # note: unpacking things here and passing into function saves a lot of allocations
    @unpack β, ξ, dE, dV, dU0 = m

    # Extract zs. If already applied as keyword, can save compilation time.
    ξj = get(kwargs, :ξj, zeros(T, length(d.positions[i])))
    if ξj[1] == 0
        zsfun = get_functional_form(m.zsfun)
        for h in eachindex(ξj)
            ξj[h] = zsfun(ξ, m.ρ, d.positions[i][h])
        end
    end

    return calculate_demand_product(m, d, i, k, n, β, ξj, dE, dV, dU0; kwargs...)
end


function calculate_demand_product(m::WM1{T}, d::DataSD, i, k, n,
        β::Vector{T}, ξj::Vector{T}, dE::Distribution,
        dV::Distribution, dU0::Distribution; kwargs...) where {T}
    n_products = length(d.product_ids[i])
    product_characteristics = @views d.product_characteristics[i]
    with_outside_option_dummy = d.product_ids[i][1] == 0

    demand = zero(T)

    # xb of purchased (same across draws)
    xβ_k = @views product_characteristics[k, :]' * β

    ξk = ξj[k]

    for dd in 1:n, ddd in 1:2

        # Reset for each draw
        prob_purchase_k = one(T)

        # Fill 
        if ddd == 1 # e < ξ
            e = rand_trunc(dE, -one(T) * MAX_NUMERICAL, ξk)
            u_k = xβ_k + e + rand(dV)
            z_k = u_k - e + ξk  # this way v draw still stored in z_k
            prob_draws_in_bounds = trunc_cdf(dE, -one(T) * MAX_NUMERICAL, ξk) 
        else # e >= ξ
            e = rand_trunc(dE, ξk, one(T) * MAX_NUMERICAL)
            z_k = xβ_k + ξk + rand(dV)
            u_k = z_k - ξk + e # this way v draw still stored in z_k 
            prob_draws_in_bounds = trunc_cdf(dE, ξk, one(T) * MAX_NUMERICAL) 
        end

        # Get values only if last click not in initial awareness set 
        w_k = min(u_k, z_k)

        if with_outside_option_dummy
            prob_purchase_k *= cdf(dU0, w_k - β[end])
        end

        # P(not buy j) for other products
        for j in (with_outside_option_dummy + 1):n_products
            if j == k
                continue
            end
            xβ_j = @views product_characteristics[j, :]' * β
            prob_purchase_k *= prob_not_buy(m, xβ_j, ξj[j], w_k, dE, dV)
        end

        demand += prob_purchase_k * prob_draws_in_bounds
    end

    conditional_on_search = get(kwargs, :conditional_on_search, false)
    if conditional_on_search
        demand = demand / exp(ll_no_searches(m, nothing, ξj, m.β, m.dV, m.dU0, d, 1, n, true))
    end

    return demand / n
end

