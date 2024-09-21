"""
*Search and Discovery* SD1 model with the following parameterization: 
- uᵢⱼ = xⱼ'β + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ = xⱼ'β + ξ + νᵢⱼ 
- uᵢ₀ = x₀'β + η , η_i ~ dU0
- Ξ(h) = zdfun(Ξ, ρ, pos) with ρ ≤ 0


# Fields:  
- `β::Vector{T}`: vector of preference weights. 
- `cs`: search costs. Initialized as nothing by to avoid computational cost. Can be updated through `calculate_costs!(m, data; kwargs...)`. 
- `cd`: discovery costs. Initialized in the same way as `cs`, and is also added in `calculate_costs!(m, data; kwargs...)`.
- `Ξ::T`: baseline Ξ.
- `ρ::Vector{T}`: parameters governing decrease of Ξ across positions.
- `ξ::T`: baseline ξ.
- `dE::Distribution`: distribution of ε_{ij}.
- `dV::Distribution`: distribution of ν_{ij}.
- `dU0::Distribution`: distribution of u_{i0}. 
- `zdfun::String`: select functional form f(Ξ, ρ, h) that determines the discovery value in position h. 
- `unobserved_heterogeneity::Dict`: dictionary of unobserved heterogeneity parameters and options. Currently not used. 
"""

@with_kw mutable struct SD1{T} <: SD where T <: Real
	β::Vector{T} 
	cs::Union{T, Nothing}	= nothing 
	cd::Union{T, Nothing}	= nothing 
	Ξ::T
	ρ::Vector{T}
	ξ::T
	dE::Distribution
	dV::Distribution
	dU0::Distribution
	zdfun::String 
	unobserved_heterogeneity::Dict = Dict()

	@assert ρ[1] <= 0 "ρ[1] must be less or equal to zero for weakly decreasing discovery value across positions."
end 

# Functions from SDCore to SD1
SDCore(m::SD1) = SDCore(; β = m.β, Ξ = m.Ξ, ρ = m.ρ, ξ = m.ξ, ξρ = [0.0], cs = m.cs, cd = m.cd, 
							dE = m.dE, dV = m.dV, dU0 = m.dU0, dW = Normal(0, 0), zdfun = m.zdfun, zsfun = "linear", unobserved_heterogeneity = m.unobserved_heterogeneity)

calculate_welfare(model::SD1, data::DataSD, n_sim 	; method = "effective_values", kwargs...) = calculate_welfare(SDCore(model), data, n_sim; method = method, kwargs...)

function calculate_costs!(m::SD1, d, n_draws_cd; 
							force_recompute = true,
							cd_kwargs...)

	m1 = SDCore(m) 
	calculate_costs!(m1, d, n_draws_cd; force_recompute, cd_kwargs...) 

	m.cs = m1.cs
	m.cd = m1.cd 

	return nothing 
end

generate_data(m::SD1, n_consumers, n_products; kwargs...) = generate_data(SDCore(m), n_consumers, n_products; kwargs...)
generate_data(m::SD1, data::DataSD; kwargs...) = generate_data(SDCore(m), data; kwargs...)

evaluate_fit(m::SD1, data::DataSD, n_sim; kwargs...) = evaluate_fit(SDCore(m), data, n_sim; kwargs...)


# Estimation 
function prepare_arguments_likelihood(m::SD1, estimator::Estimator, d::DataSD; kwargs...) 
	
	# Get functional forms 
	zdfun = get_functional_form(m.zdfun)
	
	# get data arguments 
	data_arguments = prepare_data_arguments_likelihood(d) 

	# Keep fixed seed: either random or provided by kwargs 
	seed = get(kwargs, :seed, rand(1:10^9))

    return data_arguments..., zdfun, nothing, seed 
end




# Vectorize parameters 
function vectorize_parameters(m::SD1; kwargs...)
	# Default estimate all parameters 
	θ = if !haskey(kwargs, :fixed_parameters)
			θ = vcat(m.β, m.Ξ, m.ρ, m.ξ) 
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
			end
			θ
		end

	θ = add_distribution_parameters(m, θ, kwargs)

	return θ
end

function construct_model_from_pars(θ::Vector{T}, m::SD1; kwargs...) where T <: Real

	# Extract parameters from vector, some may be fixed through kwargs 
	β, Ξ, ρ, ξ,	_, ind_last_par  = extract_parameters(m, θ; kwargs...)
	dE, dV, dU0 = extract_distributions(m, θ, ind_last_par; kwargs...)

	# Construct model from parameters 
	m_new = SD1{T}(; β, Ξ, ρ, ξ, dE, dV, dU0, zdfun = m.zdfun)

    return m_new 
end

function extract_parameters(m::SD1, θ::Vector{T}; kwargs...) where T <: Real

	n_beta = length(m.β)
	n_ρ = length(m.ρ)

	# track where in parameter vector we are and move it. 
	ind_current = 1 

	# Default: estimate all parameters
	if !haskey(kwargs, :fixed_parameters)
		β = θ[1:n_beta] ; ind_current += n_beta 
		Ξ = θ[ind_current] ; ind_current += 1
		ρ = θ[ind_current:ind_current + n_ρ - 1] ; ind_current += n_ρ
		ξ = θ[ind_current] ; ind_current += 1
		return β, Ξ, ρ, ξ, nothing, ind_current
	end

	# If keyword supplied, don't estimate parameters indicated in fixed_parameters
	fixed_parameters = get(kwargs, :fixed_parameters, nothing)
	β = if fixed_parameters[1];  T.(m.β) ; else ; ind_current += n_beta; θ[1:n_beta] ; end
	ξ = if fixed_parameters[2];  T(m.ξ) ; else ; ind_current += 1 ; θ[ind_current - 1] ; end
	Ξ = if fixed_parameters[3];  T(m.Ξ) ; else ; ind_current += 1 ; θ[ind_current - 1] ;end
	ρ = if fixed_parameters[4];  T.(m.ρ) ; else ; ind_current += n_ρ ;  θ[ind_current:ind_current + n_ρ - 1 - 1] ; end
	return β, Ξ, ρ, ξ, nothing, ind_current
end


function ll_no_searches(m::SD1, zd_h::Vector{T}, ξ::T, β::Vector{T}, dV, dU0, d::DataSD, i::Int, n_draws, complement) where T <: Real
	min_position_discover = complement ? searchsortedfirst(d.positions[i], 1) - 1 : d.min_discover_indices[i] # if complement 
	n_products = length(d.product_ids[i])
	positions = @views d.positions[i]
	with_outside_option_dummy = d.product_ids[i][1] == 0

	LL = zero(T)
	
	for dd in 1:n_draws, h in min_position_discover:n_products

		# If not last product in same position or last product, skip 
		if h < n_products && positions[h] == positions[h+1]
			continue
		end

		# Set lower bound for truncation based on position 
		lb::T = if h < n_products # not yet last position 
				zd_h[h + 1] - (with_outside_option_dummy ? β[end] : zero(T))
			else # no lower bound if last position 
				- T(MAX_NUMERICAL)
			end

		# Set upper bound for truncation based on position
		ub::T = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
				T(MAX_NUMERICAL)
			else 
				zd_h[h]  - (with_outside_option_dummy ? β[end] : zero(T)) # Max accounts for case where nA0 < nd 
			end

		# Get probability of u0 in bounds and draw for u0 
		prob_u0_in_bounds = trunc_cdf(dU0, lb, ub) 
		u0_draw = rand_trunc(dU0, lb, ub) + (with_outside_option_dummy ? β[end] : zero(T))
		
		# Initialize for probability
		prob_no_search_given_draw = one(T)

		# Loop over products up to one discovered 
		for j in 2:h 
			zs_j = @views d.product_characteristics[i][j, :]' * β + ξ
			prob_no_search_given_draw *= prob_not_search(m, u0_draw, zs_j, dV)

			if prob_no_search_given_draw == 0
				prob_no_search_given_draw = T(1e-100)  # adding this helps AD 
				break
			end
		end
		if complement 
			LL += (1 - prob_no_search_given_draw) * prob_u0_in_bounds
		else
			LL += prob_no_search_given_draw * prob_u0_in_bounds
		end
	end

	# println(log(LL / n_draws))
	return log(max(T(ALMOST_ZERO_NUMERICAL), LL / n_draws))
end

function ll_search_no_purchase(m::SD1, zd_h::Vector{T}, ξ::T, β::Vector{T}, dE, dV, dU0, d::DataSD, i::Int, n_draws) where T <: Real

	min_position_discover = d.min_discover_indices[i] 
	n_products = length(d.product_ids[i])
	positions = @views d.positions[i]
	with_outside_option_dummy = d.product_ids[i][1] == 0
	consideration_set = @views d.consideration_sets[i]

	LL = zero(T)

	for dd in 1:n_draws, h in min_position_discover:n_products

		# If not last product in same position or last product, skip 
		if h < n_products && positions[h] == positions[h+1]
			continue
		end
		# Set lower bound for truncation based on position 
		lb::T = if h < n_products # not yet last position 
				zd_h[h + 1] - (with_outside_option_dummy ? β[end] : zero(T))
			else # no lower bound if last position 
				- T(MAX_NUMERICAL)
			end

		# Set upper bound for truncation based on position
		ub::T = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
				T(MAX_NUMERICAL)
			else 
				zd_h[h]  - (with_outside_option_dummy ? β[end] : zero(T)) # Max accounts for case where nA0 < nd 
			end

		# Get probability of u0 in bounds and draw for u0 
		prob_u0_in_bounds = trunc_cdf(dU0, lb, ub) 
		u0_draw = rand_trunc(dU0, lb, ub) + (with_outside_option_dummy ? β[end] : zero(T))

		prob_searches_given_draw = one(T) 
		
		for j in 2:h 
			xβ = @views d.product_characteristics[i][j, :]' * β 

			if consideration_set[j] # searched item 
				prob_searches_given_draw *= prob_search_not_buy(m, xβ, ξ, u0_draw, u0_draw, dE, dV)
			else
				prob_searches_given_draw *= prob_not_search(m, u0_draw, xβ + ξ, dV)
			end
			
			if prob_searches_given_draw == 0 
				prob_searches_given_draw = T(1e-100) 
				break
			end
		end

		LL += prob_searches_given_draw * prob_u0_in_bounds
	end
	return log(max(T(ALMOST_ZERO_NUMERICAL), LL / n_draws))
end

function ll_purchase(m::SD1, zd_h::Vector{T}, ξ::T, β::Vector{T}, dE, dV, dU0, d::DataSD, i::Int, n_draws) where T <: Real

	min_position_discover = d.min_discover_indices[i] 
	n_products = length(d.product_ids[i])
	positions = @views d.positions[i]
	with_outside_option_dummy = d.product_ids[i][1] == 0
	consideration_set = @views d.consideration_sets[i]

	LL = zero(T)

	k = d.purchase_indices[i] # index of purchased product

	for dd in 1:n_draws, h in min_position_discover:n_products, ddd in 1:2

		# If not last product in same position or last product, skip 
		if h < n_products && positions[h] == positions[h+1]
			continue
		end
		
		# Reset for each draw
		prob_searches_given_draw = one(T) 

		# xb of purchased 
		xβ_k = @views d.product_characteristics[i][k, :]' * β 

		# Set lower bound for truncation based on position 
		lb::T = if h < n_products # not yet last position 
				zd_h[h + 1] - xβ_k
			else # no lower bound if last position 
				- T(MAX_NUMERICAL)
			end

		# Set upper bound for truncation based on position
		ub::T = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
				T(MAX_NUMERICAL)
			elseif positions[h] == positions[min_position_discover] == positions[k] # if position last click same as where purchased -> no upper bound 
				T(MAX_NUMERICAL)
			else 
				zd_h[h] - xβ_k  # Max accounts for case where nA0 < nd 
			end

		# Fill 
		if ddd == 1 # e < ξ
				e =  rand_trunc(dE, -one(T)*MAX_NUMERICAL, ξ)
				u_k = xβ_k + e + rand_trunc(dV, lb - e, ub - e)
				z_k = u_k - e + ξ # this way v draw still stored in z_k
				prob_draws_in_bounds = trunc_cdf(dE, -one(T)*MAX_NUMERICAL, ξ) * 
								trunc_cdf(dV, lb - e, ub - e)
		else # e >= ξ
				e = rand_trunc(dE, ξ, one(T)*MAX_NUMERICAL)
				z_k = xβ_k + ξ + rand_trunc(dV, lb - ξ, ub - ξ)
				u_k = z_k - ξ + e # this way v draw still stored in z_k 
				prob_draws_in_bounds = trunc_cdf(dE, ξ, one(T)*MAX_NUMERICAL) * 
								trunc_cdf(dV, lb - ξ, ub - ξ)
		end		
		
		# Get values only if last click not in initial awareness set 
		wt_k = min(u_k, z_k) 
		w_k = positions[h] == 0 ? wt_k : min(wt_k, zd_h[h])

		# Get probability outside option P(u0 < min{wj,zd(j-1)})
		if with_outside_option_dummy
			prob_searches_given_draw *= cdf(dU0, w_k - β[end])
		end
		
		# Probabilities for other products
		for j in 1 + with_outside_option_dummy:h

			xβ_j = @views d.product_characteristics[i][j, :]' * β

			# searched, and discovered before the one that was purchased
			if consideration_set[j] && positions[j] < positions[k]
				prob_searches_given_draw *= prob_search_not_buy(m, xβ_j, ξ, w_k, w_k, dE, dV) 
			# searched, and discovered after the one that was purchased
			elseif consideration_set[j] && j != k && positions[j] >= positions[k]
				prob_searches_given_draw *= prob_search_not_buy(m, xβ_j, ξ, wt_k, wt_k, dE, dV) 
			# unsearched & discovered before 
			elseif positions[j] < positions[k]
				prob_searches_given_draw *= prob_not_search(m, w_k, ξ + xβ_j, dV)
			# unsearched & at same time or after 
			elseif j != k  && positions[j] >= positions[k]
				prob_searches_given_draw *= prob_not_search(m, wt_k, ξ + xβ_j, dV)
			end
			if prob_searches_given_draw == 0 
				prob_searches_given_draw = T(ALMOST_ZERO_NUMERICAL) 
				break 
			end
		end	
		LL += prob_searches_given_draw * prob_draws_in_bounds
	end
	return log(max(T(ALMOST_ZERO_NUMERICAL), LL / n_draws))
end

"""
	function prob_search_not_buy(m::Union{SD1, WM1}, xβ::T, ξ::T, lb::T, ub::T, dE::Normal, dV::Normal)  where T 

Compute probability of searching without buying. This probability is given by P(xβ + ξ + ν_j >= lb ∩ xβ + ξ + ν_j + ε_j < ub). 
"""
@inline function prob_search_not_buy(m::Union{SD1, WM1}, xβ::T, ξ::T, lb::T, ub::T,
							dE::Normal{R1}, dV::Normal{R2})  where {T <: Real, R1 <: Real, R2 <: Real} 

	σe = std(dE) 
	σv = std(dV)
					
	a = σe > 0 ? (ub - xβ)/σe : T(MAX_NUMERICAL)
	b = σe > 0 ? -σv/σe : -T(MAX_NUMERICAL)
	Y = (lb -ξ -xβ) / σv

	P = cdf_n(a / sqrt(1 + b^2)) - 
			bvncdf(a / sqrt(1 + b^2), Y, -b/sqrt(1 + b^2))
	
	return P::T
end

"""
	function prob_not_search(m::Union{SD1, WM1}, u, z_j, dV)

Compute probaility of not searching alternaitve, given chosen option has utility `u`. 
"""
@inline function prob_not_search(m::Union{SD1, WM1}, u::T, z_j::T, dV) where T 
	return cdf(dV, u - z_j)
end


## Demand functions
function calculate_demand(m::SD1, d::DataSD, j, n_draws; kwargs...) 

	set_seed(kwargs) 

	if length(d) > 1
		throw(ArgumentError("Trying to predict product demand for multiple sessions. Use loop over sessions."))
	end

	# Different functions depending on whether product is outside option or not
	demand_j = if d.product_ids[1][j] == 0 
					return calculate_demand_outside_option(m, d, n_draws; kwargs...)
				else
					return calculate_demand_product(m, d, j, n_draws; kwargs...)
				end

	return demand_j
end

function calculate_demand_outside_option(m::SD1, d::DataSD, n; kwargs...) 
	T = eltype(m.β)

	# Extract zdfun. If already applied as keyword, can save compilation time.
	zd_h = get(kwargs, :zd_h, nothing)
	if isnothing(zd_h)
		zdfun = get(kwargs, :zdfun, get_functional_form(m.zdfun))
		zd_h = [zdfun(m.Ξ, m.ρ, h) for h in d.positions[1]]
	end 

	positions = @views d.positions[1]
	n_products = length(positions)

	demand = zero(T)

	for dd in 1:n, h in 2:n_products
		# If not last product in same position or last product, skip 
		if h < n_products && positions[h] == positions[h+1]
			continue
		end

		# Set lower bound for truncation based on position
		lb::T = if h < n_products # not yet last position 
				zd_h[h + 1] - m.β[end]
			else # no lower bound if last position 
				- T(MAX_NUMERICAL)
			end

		# Set upper bound for truncation based on position
		ub::T = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
				T(MAX_NUMERICAL)
			else 
				zd_h[h] - m.β[end]  # Max accounts for case where nA0 < nd 
			end

		# Get probability of u0 in bounds and draw for u0
		prob_u0_in_bounds = trunc_cdf(m.dU0, lb, ub)
		u0_draw = rand_trunc(m.dU0, lb, ub) + m.β[end]

		# Initialize for probability
		prob_buy_u0 = one(T)

		for j in 2:h
			xβ = @views d.product_characteristics[1][j, :]' * m.β
			prob_buy_u0 *= prob_not_buy(m, xβ, m.ξ, u0_draw, m.dE, m.dV)
		end

		demand += prob_buy_u0 * prob_u0_in_bounds
	end

	conditional_on_search = get(kwargs, :conditional_on_search, false)
	if conditional_on_search
		throw(ArgumentError("Conditional on search not implemented for demand calculation of outside option."))
	end
	
	return demand / n 
end

function calculate_demand_product(m::SD1, d::DataSD, k, n; kwargs...) 
	T = eltype(m.β)

	i = 1 
	n_products = length(d.product_ids[i])
	positions = @views d.positions[i]
	with_outside_option_dummy = d.product_ids[i][1] == 0

	# Extract zdfun. If already applied as keyword, can save compilation time.
	zd_h = get(kwargs, :zd_h, nothing)
	if isnothing(zd_h)
		zdfun = get(kwargs, :zdfun, get_functional_form(m.zdfun))
		zd_h = [zdfun(m.Ξ, m.ρ, h) for h in d.positions[1]]
	end 
	
	β, ξ, dE, dV, dU0 = m.β, m.ξ, m.dE, m.dV, m.dU0

	demand = zero(T) 

	for dd in 1:n, h in 1:n_products, ddd in 1:2
 		# If not last product in same position or last product, skip 
		 if h < n_products && positions[h] == positions[h+1]
			continue
		end

		# Reset for each draw
		prob_purchase_k = one(T) 

		# xb of purchased 
		xβ_k = @views d.product_characteristics[i][k, :]' * β 

		# Set lower bound for truncation based on position 
		lb::T = if h < n_products # not yet last position 
			zd_h[h + 1] - xβ_k
		else # no lower bound if last position 
			- T(MAX_NUMERICAL)
		end

		# Set upper bound for truncation based on position
		ub::T = if positions[h] == positions[k] # first no upper bound if last click in initial awareness set (position 0)
				T(MAX_NUMERICAL)
			else 
				zd_h[h] - xβ_k  # Max accounts for case where nA0 < nd 
			end

		# Fill 
		if ddd == 1 # e < ξ
			e =  rand_trunc(dE, -one(T)*MAX_NUMERICAL, ξ)
			u_k = xβ_k + e + rand_trunc(dV, lb - e, ub - e)
			z_k = u_k - e + ξ # this way v draw still stored in z_k
			prob_draws_in_bounds = trunc_cdf(dE, -one(T)*MAX_NUMERICAL, ξ) * 
							trunc_cdf(dV, lb - e, ub - e)
		else # e >= ξ
				e = rand_trunc(dE, ξ, one(T)*MAX_NUMERICAL)
				z_k = xβ_k + ξ + rand_trunc(dV, lb - ξ, ub - ξ)
				u_k = z_k - ξ + e # this way v draw still stored in z_k 
				prob_draws_in_bounds = trunc_cdf(dE, ξ, one(T)*MAX_NUMERICAL) * 
								trunc_cdf(dV, lb - ξ, ub - ξ)
		end		
		if h < k # no need to calculate, but still loop over to have fixed seed independent of which position product is shown 
			continue
		end

		# Get values only if last click not in initial awareness set 
		wt_k = min(u_k, z_k) 
		w_k = positions[h] == 0 ? wt_k : min(wt_k, zd_h[h])
		
		if with_outside_option_dummy
			prob_purchase_k *= cdf(dU0, w_k - β[end])
		end
	
		# P(not buy j) for other products
		for j in with_outside_option_dummy + 1:h
			if j == k
				continue
			end
			xβ_j = @views d.product_characteristics[i][j, :]' * β 
			if positions[j] < positions[k]
				prob_purchase_k *= prob_not_buy(m, xβ_j, ξ, w_k, dE, dV)
			else
				prob_purchase_k *= prob_not_buy(m, xβ_j, ξ, wt_k, dE, dV)
			end
		end

		demand += prob_purchase_k * prob_draws_in_bounds

	end

	conditional_on_search = get(kwargs, :conditional_on_search, false)
	if conditional_on_search
		demand = demand / exp(ll_no_searches(m, zd_h, m.ξ, m.β, m.dV, m.dU0, d, 1, n, true))
	end
	

	return demand / n
end
			



	
"""
	function prob_no_buy(m::Union{SD1, WM1}, xβ::T, ξ::T, u::T, dE::Normal, dV::Normal)  where T 

Compute probability of not buying given chosen utility has utility `u`. This probability is given by P(xβ + v_j + min(ξ, ε_j) <= u)
"""

@inline function prob_not_buy(m::Union{SD1, WM1}, xβ::T, ξ::T, u::T,
								dE::Normal, dV::Normal) where T 
	σe = std(dE) 
	σv = std(dV)

	a = (u - xβ) / σe
	b = - σv / σe
	Y = (u - ξ - xβ) / σv

	P = cdf(dV, u - ξ - xβ) + cdf(Normal(), a / sqrt(1 + b^2)) - bvncdf(a / sqrt(1 + b^2), Y, -b / sqrt(1 + b^2))
	return P 
end

# ######################################################################
# ## P(click but not buy )
# ######################################################################

# @inline function cdf_ZWU(m::Union{SD3,WM1},xb::T,xi::T,K::T,
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

# @inline function cdf_ZWU(m::Union{SD3,WM1},xb::T,xi::T,lb::T,ub::T,
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
# @inline function cdf_ZWnb(m::Union{SD3,WM1},xb::T,xi::T,z::T,
# 								dE::Normal,dV::Normal) where T 
# 	σe = std(dE) 
# 	σv = std(dV)

# 	a = (z-xb)/σe
# 	b = -σv/σe
# 	Y = (z - xi - xb) / σv

# 	P = cdf(dV,z-xi-xb) + cdf(Normal(),a/sqrt(1+b^2)) - StructSearch.bvncdf(a/sqrt(1+b^2),Y,-b/sqrt(1+b^2)) 
# 	return P 
# end