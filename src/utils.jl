# Generate products 

"""
    generate_products(n_sessions, distribution::Distribution;
        n_products = 1_000_000,
        n_products_per_session = 30,
        outside_option = true,
        kwargs...)

Generate product IDs and their characteristics for `n_sessions` sessions. Each session has `n_products_per_session` products randomly sampled from `n_products` available in total. By default, `n_products` is large and 30 products are sampled per session. 
    
Product characteristics for each product are sampled from the specified `distribution`, which needs to be multivariate (e.g., `MvNormal`) when using multiple characteristics. If `outside_option=true`, then an outside option is included as the first product for each session with product ID 0. For this product, the last characteristic is set to 1.0 and the remaining product characteristics are set to 0.0.
    
Returns product IDs and product characteristics as two separate vectors. Each element is a session, with a vector of product IDs or matrix of characteristics for all products available to the session. 
"""
function generate_products(n_sessions, distribution::Distribution;
        n_products = 1_000_000,
        n_products_per_session = 30,
        outside_option = true,
        kwargs...)

    # Set seed 
    rng = get_rng(kwargs)

    # Product ids, 0 will be outside option 
    pid = collect(1:n_products)

    # Draw product characteristics for each product  
    product_characteristics = rand(rng, distribution, n_products)
    if size(product_characteristics, 2) > 1
        product_characteristics = product_characteristics'
    end

    # Draw products per session randomly from set of products for each session 
    product_ids = if outside_option
        [vcat(0, rand(rng, pid, n_products_per_session)) for i in 1:n_sessions]
    else
        [rand(rng, pid, n_products_per_session) for i in 1:n_sessions]
    end

    # Gather characteristics for each product
    product_characteristics = if outside_option
        [hcat(
             vcat(zeros(1, size(product_characteristics, 2)),
                 product_characteristics[pids[2:end], :]),
             vcat(1.0, zeros(length(pids) - 1))) for pids in product_ids]
    else
        [product_characteristics[pids, :] for pids in product_ids]
    end

    return product_ids, product_characteristics
end

function expand_products!(data; kwargs...) 

    rng = get_rng(kwargs) # set seed 

    n_prod_in_session = length.(data.product_ids)
    n_prod_max = maximum(n_prod_in_session)

    session_ids_with_enough_products = [findall(x -> x >= h, n_prod_in_session) for h in 1:n_prod_max]

    for i in eachindex(data) 
        n_prod_i = n_prod_in_session[i]
        if n_prod_i < n_prod_max # expand if not full 
            data.positions[i] = data.positions[session_ids_with_enough_products[end][1]] # last one has full position
            n_added_products = n_prod_max - n_prod_i
            
            for h in n_prod_i+1:n_prod_max
                sid = rand(rng, session_ids_with_enough_products[h]) # session from which to take product 
                push!(data.product_ids[i], data.product_ids[sid][h]) 
                data.product_characteristics[i] = vcat(data.product_characteristics[i], data.product_characteristics[sid][h, :]')
                push!(data.consideration_sets[i], fill(false, n_added_products)...)
            end
        end
    end

    return nothing
end


# Get chunk size
function get_chunks(n::Int)
    chunk_size = max(1, n ÷ (TASKS_PER_THREAD * Threads.nthreads()))
    data_chunks = Iterators.partition(1:n, chunk_size)
    return chunk_size, data_chunks
end

# Functional form definitions 
function get_functional_form(s::String)
    if s == ""
        (Ξ, ρ, pos) -> Ξ[1]
    elseif s == "linear"
        (Ξ, ρ, pos) -> Ξ[1] + ρ[1] * pos
    elseif s == "log"
        (Ξ, ρ, pos) -> Ξ[1] + ρ[1] * _LP[pos + 1]
    elseif s == "exp"
        (Ξ, ρ, pos) -> Ξ[1] + ρ[1] * _EP[pos + 1]

    elseif length(s) >= 7 && s[1:7] == "linear-" 
        k = parse(Int, s[8:end])
        function (Ξ, ρ, pos)
            if pos == 0
                return Ξ[1]
            else
                return Ξ[1] + @views sum(ρ[i] for i in eachindex(ρ)[1:min(pos, k)]) + (pos > k ? ρ[k+1] * (pos - k) : zero(Ξ))
            end
        end

    elseif s[1:3] == "log" && s[4] == '-'
        k = parse(Int, s[5:end])
        function (Ξ, ρ, pos)
            if pos == 0
                return Ξ[1]
            else
                return Ξ[1] + @views sum(ρ[i] for i in eachindex(ρ)[1:min(pos, k)]) + (pos > k ? ρ[k+1] * _LP[pos - k + 1] : zero(Ξ))
            end
        end
    else
        error("zdfun not correctly specified. Currently only 'linear', 'log', 'exp', '', or 'log-k' and 'linear-k' are supported.")
    end
end

# Set seed 
function get_rng(kwargs)

    rng = get(kwargs, :rng, Random.default_rng())

    # Set seed for rng if given 
    seed = get(kwargs, :seed, nothing) # if no seed set, just draw one
    if !isnothing(seed)
        Random.seed!(rng, seed)
    end
    return rng
end

# Take or generate draw
@inline function take_or_generate_draw!(draws, rng, distribution::Distribution, j)
    if isnothing(draws)
        return rand(rng, distribution)::Float64
    else
        return draws[j]
    end
end

function get_precomputed_draws(kwargs)
    draws_u0 = get(kwargs, :draws_u0, nothing)
    draws_e = get(kwargs, :draws_e, nothing)
    draws_v = get(kwargs, :draws_v, nothing)
    draws_w = get(kwargs, :draws_w, nothing)

    return (draws_u0, draws_e, draws_v, draws_w)
end

function get_precomputed_draws_indexed(kwargs, s)
    draws_u0, draws_e, draws_v, draws_w = get_precomputed_draws(kwargs)
    draws_η = get(kwargs, :draws_η, nothing)

    draws_u0 = isnothing(draws_u0) ? nothing : @views draws_u0[s]
    draws_e = isnothing(draws_e) ? nothing : @views draws_e[s]
    draws_v = isnothing(draws_v) ? nothing : @views draws_v[s]
    draws_w = isnothing(draws_w) ? nothing : @views draws_w[s]
    draws_η = isnothing(draws_η) ? nothing : @views draws_η[s]

    return (draws_u0, draws_e, draws_v, draws_w, draws_η)
end

@inline function replace_typemin_draws!(draws, rng, dist::Distribution)
    @inbounds for i in eachindex(draws)
        if isequal(draws[i], typemin(eltype(draws)))
            draws[i] = rand(rng, dist)
        end
    end
end

function take_or_generate_consumer_shocks(m, d, kwargs) 

    if !has_unobserved_heterogeneity(m) 
        return nothing
    end

    draws_η = get(kwargs, :draws_η, nothing)
    rng = get_rng(kwargs)
    
    if isnothing(draws_η) 
        draws_η = generate_consumer_shocks(m, d, rng)
    end

    return draws_η
end

function generate_consumer_shocks(m, d, rng)

    if !has_unobserved_heterogeneity(m) 
        return nothing
    end

    n_cons = d.consumer_ids[end]
    _, chunks = get_chunks(n_cons) 

    draws_η = zeros(Float64, n_cons, size(m.heterogeneity.distribution.Σ, 1))

    tasks = map(chunks) do chunk
        Threads.@spawn begin

            for i in chunk 
                # Draws for unobserved heterogeneity 
                r = rand(rng, m.heterogeneity.distribution) 
                draws_η[i, :] .= r
            end
        end
    end
    fetch.(tasks)
    return draws_η
end

function lik_return_stable(lik::T, return_log) where T <: Real

    if return_log 
        return log(max(T(ALMOST_ZERO_NUMERICAL), lik)) 
    else
        return lik
    end
end

"""
    add_product_fe!(model::SDModel, data::DataSD, n_min::Int, location::String)

Add product fixed effects for all products that observed at least `n_min` times in the data. Fixed effects can shift either the search value, the hidden part of utility, or both. This is specified through `location`, which is set to eitehr `"search"`, `"hidden"`, or `"both"`. 
"""
function add_product_fe!(model::SDModel, data::DataSD, n_min::Int, location::String)

    product_ids_with_fe = find_products_appearing_min_n_times(data, n_min) 

    add_product_fe_data!(data, product_ids_with_fe) 

    add_product_fe_model!(model, data, location) 

    return nothing 

end

function add_product_fe_data!(data, product_ids_with_fe) 

    sort!(product_ids_with_fe) # sort product ids with fixed effects (yields faster search later on)

    # Expand characteristics and add product indicator 
    n_fe = length(product_ids_with_fe)
    for i in eachindex(data)
        product_ids = data.product_ids[i]
        product_indicator = zeros(length(product_ids), n_fe) 
    
        for (j, id) in enumerate(product_ids) 
            index_in_product_fe = searchsorted(product_ids_with_fe, id) 
    
            # Skip if no FE for this product (e.g., outside option)
            if length(index_in_product_fe) < 1 
                continue
            end
            product_indicator[j, index_in_product_fe[1]] = 1 
        end
    
    
        data.product_characteristics[i] = 
            hcat(data.product_characteristics[i][:, 1:end-1], 
                 product_indicator,
                 data.product_characteristics[i][:, end], # moves outside option indicator to end 
                 )
    end
    println("Number of product fixed effects added: ", n_fe)

    return product_ids_with_fe 
end

function find_products_appearing_min_n_times(data, n_min)

    product_ids = sort(vcat(data.product_ids...))
    counts = countmap(product_ids )
    counts = sort(collect(counts), by=x->x[2], rev=true) # sort by counts
    product_ids_with_fe = [pid for (pid, count) in counts if count >= n_min] # only keep products with more than n_min observatoins 
    product_ids_with_fe = product_ids_with_fe[2:end] # first is outside option, dropping this one 

    sort!(product_ids_with_fe) # sort product ids with fixed effects (yields faster search later on)

    return product_ids_with_fe
end



function add_product_fe_model!(m::SDModel, d::DataSD, location) 
    n_coef = length(m.β) - 1 # -1 for outside option
    n_fe = size(d.product_characteristics[1], 2) - n_coef - 1 
    β = vcat(m.β[1:n_coef], zeros(n_fe), m.β[end])
    m.β = β

    γ = vcat(m.information_structure.γ[1:n_coef], zeros(n_fe), m.information_structure.γ[end])
    m.information_structure.γ = γ

    κ = vcat(m.information_structure.κ[1:n_coef], zeros(n_fe), m.information_structure.κ[end])
    m.information_structure.κ = κ

    if location == "both"

        add_fe_γ!(m, n_coef, n_fe) 
        add_fe_κ!(m, n_coef, n_fe) 

    elseif location == "search" 
        add_fe_γ!(m, n_coef, n_fe) 

    elseif location == "hidden" 
        add_fe_κ!(m, n_coef, n_fe) 
    else
        throw(ArgumentError("Location must be either 'both', 'search' or 'hidden'."))
    end
end

function add_fe_γ!(m, n_coef, n_fe) 
    m.information_structure.indices_characteristics_γ_union = vcat(
        collect(m.information_structure.indices_characteristics_γ_union), 
        collect(n_coef+1:n_coef + n_fe)
    )

    if typeof(m.information_structure.indices_characteristics_γ_union) <: UnitRange || 
        typeof(m.information_structure.indices_characteristics_γ_union) <: Vector{Int}

        m.information_structure.indices_characteristics_γ_individual= m.information_structure.indices_characteristics_γ_union
    else
        si =  m.information_structure.indices_characteristics_γ_individual
        for i in eachindex(si) 
            si[i] = vcat(collect(si[i]), collect(n_coef+1:n_coef + n_fe))
        end
    end
end

function add_fe_κ!(m, n_coef, n_fe) 
    m.information_structure.indices_characteristics_κ_union = vcat(
        collect(m.information_structure.indices_characteristics_κ_union), 
        collect(n_coef+1:n_coef + n_fe)
    )

    if typeof(m.information_structure.indices_characteristics_κ_union) <: UnitRange || 
        typeof(m.information_structure.indices_characteristics_κ_union) <: Vector{Int}

        m.information_structure.indices_characteristics_κ_individual= m.information_structure.indices_characteristics_κ_union
    else
        si =  m.information_structure.indices_characteristics_κ_individual
        for i in eachindex(si) 
            si[i] = vcat(collect(si[i]), collect(n_coef+1:n_coef + n_fe))
        end
    end
end


##################################################################################
# Computations for bivariate normal 

# Source: https://github.com/JuliaStats/StatsFuns.jl/blob/60dc61c63ba74d0da0e1000a4b3b785a6badacf5/src/tvpack.jl
# (open pull request on StatsFuns.jl)
# Removed unnecessary parts and adapted to accept Reals (necessary for AD)

# Written in Julia by Andreas Noack Jensen
# January 2013
#
# Translation of Fortran file tvpack.f authored by
#
# Alan Genz
# Department of Mathematics
# Washington State University
# Pullman, WA 99164-3113
# Email : alangenz@wsu.edu
#
# Original source available from
# http://www.math.wsu.edu/faculty/genz/software/fort77/tvpack.f

# This function is based on the method described by
#     Drezner, Z and G.O. Wesolowsky, (1989),
#     On the computation of the bivariate normal integral,
#     Journal of Statist. Comput. Simul. 35, pp. 101-107,
# 		with major modifications for double precision, and for |R| close to 1.

@inline normcdf(x) = Distributions.normcdf(x)

const bvncdf_w_array = [0.1713244923791705e+00 0.4717533638651177e-01 0.1761400713915212e-01;
                        0.3607615730481384e+00 0.1069393259953183e+00 0.4060142980038694e-01;
                        0.4679139345726904e+00 0.1600783285433464e+00 0.6267204833410906e-01;
                        0.0 0.2031674267230659e+00 0.8327674157670475e-01;
                        0.0 0.2334925365383547e+00 0.1019301198172404e+00;
                        0.0 0.2491470458134029e+00 0.1181945319615184e+00;
                        0.0 0.0 0.1316886384491766e+00;
                        0.0 0.0 0.1420961093183821e+00;
                        0.0 0.0 0.1491729864726037e+00;
                        0.0 0.0 0.1527533871307259e+00]

const bvncdf_x_array = [-0.9324695142031522e+00 -0.9815606342467191e+00 -0.9931285991850949e+00;
                        -0.6612093864662647e+00 -0.9041172563704750e+00 -0.9639719272779138e+00;
                        -0.2386191860831970e+00 -0.7699026741943050e+00 -0.9122344282513259e+00;
                        0.0 -0.5873179542866171e+00 -0.8391169718222188e+00;
                        0.0 -0.3678314989981802e+00 -0.7463319064601508e+00;
                        0.0 -0.1252334085114692e+00 -0.6360536807265150e+00;
                        0.0 0.0 -0.5108670019508271e+00;
                        0.0 0.0 -0.3737060887154196e+00;
                        0.0 0.0 -0.2277858511416451e+00;
                        0.0 0.0 -0.7652652113349733e-01]

function bvnuppercdf(dh::T, dk::T, r::Tf)::T where {T <: Real, Tf}
    # Fast path for out-of-domain r
    @inbounds if r < -one(T) || r > one(T)
        throw(DomainError("r must be ∈ [-1,1]"))
    end

    # Fast path for infinite arguments
    if !isfinite(dh) || !isfinite(dk)
        if dh == typemax(T) || dk == typemax(T)
            return zero(T)
        elseif dh == typemin(T)
            return normcdf(-dk)
        elseif dk == typemin(T)
            return normcdf(-dh)
        else
            return zero(T)
        end
    end

    absr = abs(r)
    # Precompute constants and select quadrature order
    if absr < 0.3
        ng = 1; lg = 3
    elseif absr < 0.75
        ng = 2; lg = 6
    else
        ng = 3; lg = 10
    end

    h = dh
    k = dk
    hk = h * k
    bvn = zero(T)

    if absr < 0.925
        if absr > 0
            hs = (h * h + k * k) * 0.5
            asr = asin(r)
            @inbounds for i in 1:lg
                x = bvncdf_x_array[i, ng]
                w = bvncdf_w_array[i, ng]
                for j in (-1, 1)
                    sn = sin(asr * (j * x + 1.0) * 0.5)
                    bvn += w * exp((sn * hk - hs) / (1.0 - sn * sn))
                end
            end
            bvn *= asr / (4.0pi)
        end
        bvn += normcdf(-h) * normcdf(-k)
    else
        if r < 0
            k = -k
            hk = -hk
        end
        if absr < 1
            as = (1.0 - r) * (1.0 + r)
            a = sqrt(as)
            bs = (h - k)^2
            c = (4.0 - hk) * 0.125
            d = (12.0 - hk) * 0.0625
            asr = -(bs / as + hk) * 0.5
            if asr > -100
                bvn = a * exp(asr) *
                      (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 +
                       c * d * as * as / 5.0)
            end 
            if -hk < 100
                b = sqrt(bs)
                bvn -= exp(-hk * 0.5) * sqrt(2.0pi) * normcdf(-b / a) * b *
                       (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
            end
            a2 = a / 2.0
            @inbounds for i in 1:lg
                x = bvncdf_x_array[i, ng]
                w = bvncdf_w_array[i, ng]
                for j in (-1, 1)
                    xs = (a2 * (j * x + 1.0))^2
                    rs = sqrt(1.0 - xs)
                    asr = -(bs / xs + hk) * 0.5
                    if asr > -100
                        bvn += a2 * w * exp(asr) *
                               (exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs -
                                (1.0 + c * xs * (1.0 + d * xs)))
                    end
                end
            end
            bvn /= -2.0pi
        end
        if r > 0
            bvn += normcdf(-max(h, k))
        else
            bvn = -bvn
            if k > h
                bvn += normcdf(k) - normcdf(h)
            end
        end
    end
    return bvn
end

@inline bvncdf(dh, dk, r) = bvnuppercdf(-dh, -dk, r)

##################################################################################
# Truncation of random variables -> use own functions for normal to avoid rejection sampling 

# Normal distribution: helper functions 
@inline cdf_n(z::T) where {T <: Real} = Distributions.erfc(-z * invsqrt2) / 2

@inline function invcdf_n(z::T) where {T <: Real}
    # Note: For z very close to 1 or zero, will return Inf
    # Fixing bounds helps calculate gradient (and introduces little inaccuracy)
    if z >= 1 - eps(T)
        return sqrt2 * erfinv(1 - eps(T))
    elseif z <= eps(T)
        return sqrt2 * erfinv(2 * eps(T) - 1)
    else
        return sqrt2 * erfinv(2 * z - 1)
    end
end

"""
	rand_trunc(d::Normal, lb::T, ub::T) where T

Take a random draw from a truncated normal distribution. 
"""
@inline function rand_trunc(rng, d::Normal, lb::T, ub::T) where {T}
    @assert d.μ == 0
    if lb >= ub
        rand(rng) # just drawing a number keeps seed fixed 
        return lb
    end
    # Differ whether in upper or lower tail. By symmetry same, but works more accurately. Further tests an improvements welcome. 
    if lb > 0 && ub > 0
        l = cdf(d, -ub)
        u = cdf(d, -lb)
        if l == u > 0
            u += eps(u)
        end
        r = l + rand(rng) * (u - l)
        q = quantile(d, max(min(r, 1 - 1e-16), 1e-300)) # imposing bounds to avoid numerical issues in gradient leading to nan when inf from quantile
        return max(min(-q, ub), lb)
    else
        l = cdf(d, lb)
        u = cdf(d, ub)
        if l == u > 0
            u += eps(u)
        end

        r = l + rand(rng) * (u - l)
        q = quantile(d, max(min(r, 1 - 1e-16), 1e-300)) # imposing bounds to avoid numerical issues in gradient leading to nan when inf from quantile
        return max(min(q, ub), lb)
    end
end

# @inline function rand_trunc(d::Gumbel,lb::T,ub::T) where T
# 	if lb >= ub 
# 		return lb 
# 	end
# 	if ub < -1e6 || lb > 1e6  || abs(ub-lb) < 1e-8
# 		return (ub-lb) / 2 
# 	end

# 	tp = trunc_cdf(d,lb,ub)
# 	r = if tp > sqrt(eps(T)) 
# 			c1 = lb < -1e6 ? zero(T) : cdf(d,lb) 
# 			rtcdf = rand(T)*trunc_cdf(d,lb,ub)
# 			t = isnan(c1) ? rtcdf : rtcdf + c1
# 			# note: cdf(d,lb) gives nan if lb too small because of gumbel cdf
# 			# so accounting for c1 helps AD 
# 			d.μ - d.θ*log(-log(t))
# 		else 
# 			loglcdf = max(logcdf(d,lb),-1e100)
# 			logtp = log(tp) 
# 			invlogcdf(d, Distributions.logaddexp(loglcdf, logtp - randexp(T)))
# 		end
# 	r = max(min(r,ub),lb)
# 	return r::T	
# end

"""
	trunc_cdf(d::Uniform, lb::T, ub::T) where T

Calculate the cumulative distribution function of a truncated uniform distribution.
"""
@inline function trunc_cdf(d::Uniform, lb::T, ub::T) where {T <: Real}
    if ub <= minimum(d) || lb >= maximum(d)
        return zero(T)
    end
    return cdf(d, ub) - cdf(d, lb)
end

@inline function rand_trunc(rng, d::Uniform, lb::T, ub::T) where {T <: Real}
    if ub <= minimum(d) || lb >= maximum(d)
        rand(rng) # just drawing a number keeps seed fixed 
        return zero(T)
    end
    if lb >= ub || isnan(ub)
        rand(rng) # just drawing a number keeps seed fixed 
        return lb
    elseif isnan(lb)
        rand(rng) # just drawing a number keeps seed fixed 
        return ub
    end
    return rand(rng, truncated(d, lb, ub))
end

"""
	function rand_trunc(d::Distribution, lb::T, ub::T) where T

Calculate the cumulative distribution function of a generic truncated distribution.
"""
@inline function rand_trunc(rng, d::Distribution, lb::T, ub::T) where {T}
    if ub <= minimum(d) || lb >= maximum(d)
        rand(rng)
        return zero(T)
    end
    if lb >= ub
        rand(rng)
        return lb
    end

    tp = trunc_cdf(d, lb, ub)
    loglcdf = max(logcdf(d, lb), -1e100)

    r = if tp > sqrt(eps(T))
        c1 = lb < -1e12 ? zero(T) : cdf(d, lb)
        rtcdf = rand(rng, T) * tp
        t = isnan(c1) ? rtcdf : rtcdf + c1
        # note: cdf(d,lb) gives nan if lb too small because of gumbel cdf
        # so accounting for c1 helps AD 
        quantile(d, min(exp(loglcdf) + t, 1))
    else
        invlogcdf(d, Distributions.logaddexp(loglcdf, log(tp) - randexp(rng)))
    end
    r = max(min(r, ub), lb)
    return r::T
end

@inline function trunc_cdf(d, lb::T, ub::T) where {T <: Real}
    # Do calculation in log-scale to avoid numerical issues
    # Note: follows truncate.jl in Distributions package 
    # introduces additional max(..) to help AD avoid NaN
    if ub <= minimum(d) || lb >= maximum(d) || lb >= ub
        return zero(T)
    end

    x = max(-1e100, logcdf(d, lb))
    y = max(-1e100, logcdf(d, ub))

    tp = exp(Distributions.logsubexp(x, y))
    return max(1e-300, tp)
end

@inline function transform_to_consecutive_ids(ids) 

    unique_vals = sort(unique(ids))

    return [searchsortedfirst(unique_vals, id) for id in sort(ids)]
end
