"""
*Search and Discovery* SD1 model with the following parameterization: 
- uᵢⱼ = xⱼ'β + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ = xⱼ'β + ξ(hᵢⱼ) + νᵢⱼ 
- uᵢ₀ = x₀'β + η , η_i ~ dU0
- Ξ(h) = zdfun(Ξ, ρ, pos) with ρ ≤ 0
- ξ(h) = zsfun(ξ, ξρ, pos)


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
- `zsfun::String`: select functional form f(ξ, ξρ, h) that determines the search value in position h.
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


# Estimation 
function prepare_arguments_likelihood(m::M, estimator::Estimator, d::DataSD) where M <: SD1	
	
	# Get functional forms 
	zdfun = get_functional_form(m.zdfun)
	zsfun = nothing 

	# Get maximum number of products
	max_n_products = maximum(length.(d.product_ids))
	
    return max_n_products, zdfun, zsfun 
end

# Vectorize parameters 

function vectorize_parameters(m::SD1; kwargs...)
	# Default estimate all parameters 
	θ = if !haskey(kwargs, :fixed_parameters)
			θ = vcat(m.β, m.ξ, m.Ξ, m.ρ) 
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
					θ = vcat(θ, m.Ξ)
				end
				if !fixed_parameters[4]
					θ = vcat(θ, m.ρ)
				end
			end
			θ
		end
	
	# Default: estimate variance of ε, keep others fixed
	if !haskey(kwargs, :distribution_options)
		θ = vcat(θ, params(m.dE)[end]) 
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

function loglikelihood(θ::Vector{T}, model::M, estimator::SmoothMLE, data::DataSD, args...; kwargs...) where {M <: SD1, T <: Real}
	
	# Extract arguments 
	max_n_products, zdfun  = args  

	# Extract parameters implied by θ 
	β, ξ, Ξ, ρ, ind_last_par  = extract_parameters(model, θ; kwargs...)
	dE, dV, dU0 = extract_distributions(model, θ, ind_last_par; kwargs...)

	if ρ[1] > 0 
		return -T(1e100)
	end
	
	# Pre-compute search and discovery values across positions -> same for all consumers 
	zd_h = [zdfun(Ξ, ρ, data.positions[1][h]) for h in 1:max_n_products]

	if get(kwargs, :debug_print, false)
		println("β = $β")
		println("ξ = $ξ")
		println("Ξ = $Ξ")
		println("ρ = $ρ")
		println("dE = $dE")
		println("dV = $dV")
		println("dU0 = $dU0")
		println("zd_h[1:5] = $zd_h[1:5]")
	end

	# Set seed for random number generation
	set_seed(kwargs)

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
				if data.search_paths[i][1] == 0 	# Case 1: no clicks (implies also no purchase)
					L += ll_no_searches(model, zd_h, β, ξ, dV, dU0, data, i, n_draws, false) 
				elseif data.purchase_indices[i] == 1 # Case 2: Some clicks but no purchase 
					L += ll_search_no_purchase(model, zd_h, β, ξ, dE, dV, dU0, data, i, n_draws) 
				else 	# Case 3: Purchase a product 
					L += ll_purchase(model, zd_h, β, ξ, dE, dV, dU0, data, i, n_draws) 
				end
				
			end

			return L # Return likelihood for chunk
		end
	end

	LL1 = sum(fetch.(tasks)) 

	LL2 = 	if estimator.conditional_on_search  
				
				tasks = map(data_chunks) do chunk 
					Threads.@spawn begin 

						local L = zero(T)

						for i in chunk  # Iterate over consumers in chunk 
							L += ll_no_searches(m, zd_h, β, ξ, dV, dU0, data, i, n_draws, true) 
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

function construct_model_from_pars(θ::Vector{T}, m::SD1; kwargs...) where T <: Real

	# Extract parameters from vector, some may be fixed through kwargs 
	β, ξ, Ξ, ρ, ind_last_par  = extract_parameters(m, θ; kwargs...)
	dE, dV, dU0 = extract_distributions(m, θ, ind_last_par; kwargs...)

	# Construct model from parameters 
	m_new = SD1{T}(; β, ξ, Ξ, ρ, dE, dV, dU0, zdfun = m.zdfun)

    return m_new 
end

function extract_parameters(m::M, θ::Vector{T}; kwargs...) where {M <: SD1, T <: Real}

	n_beta = length(m.β)
	n_ρ = length(m.ρ)

	# track where in parameter vector we are and move it. 
	ind_current = 1 

	# Default: estimate all parameters
	if !haskey(kwargs, :fixed_parameters)
		β = θ[1:n_beta] ; ind_current += n_beta 
		ξ = θ[ind_current] ; ind_current += 1
		Ξ = θ[ind_current] ; ind_current += 1
		ρ = θ[ind_current:ind_current + n_ρ - 1] ; ind_current += n_ρ
		return β, ξ, Ξ, ρ, ind_current
	end

	# If keyword supplied, don't estimate parameters indicated in fixed_parameters
	fixed_parameters = get(kwargs, :fixed_parameters, nothing)
	β = !fixed_parameters[1] ? θ[1:n_beta] : m.β;  ind_current += n_beta 
	ξ = !fixed_parameters[2] ? θ[ind_current] : m.ξ; ind_current += 1
	Ξ = !fixed_parameters[3] ? θ[ind_current] : m.Ξ;  ind_current += 1
	ρ = !fixed_parameters[4] ? θ[ind_current:ind_current+n_ρ] : m.ρ ; ind_current += n_ρ + 1 

	return (β, ξ, Ξ, ρ), ind_current
end

"""
Construct shock distributions using variances in vector θ. Starts from index c. 
"""
function extract_distributions(m::M, θ::Vector{T}, c; kwargs...) where {M <: Union{SD1},T <: Real}

	# Default: estimate variance of ε, keep others fixed 
	if !haskey(kwargs, :distribution_options)
		dE = eval(nameof(typeof(m.dE)))(params(m.dE)[1:end-1]...,abs(θ[end])) # convoluted way to allow for distributions other than Normal 
		dV = m.dV
		dU0 = m.dU0
		return dE, dV, dU0
	end 

	estimation_shock_distributions = get(kwargs, :distribution_options, nothing)

	# Extract distributions
	dE = 	if estimation_shock_distributions[1]
				c += 1 
				eval(nameof(typeof(m.dE)))(params(m.dE)[1:end-1]...,abs(θ[c]))
			else 	
				m.dE
			end

	dV =	if estimation_shock_distributions[2]
				c += 1 
				eval(nameof(typeof(m.dV)))(params(m.dV)[1:end-1]...,abs(θ[c]))
			else 	
				m.dV
			end

	dU0 = 	if estimation_shock_distributions[3] && estimation_shock_distributions[1]
				dE 
			elseif estimation_shock_distributions[3] && estimation_shock_distributions[2]
				dV
			elseif estimation_shock_distributions[4] 
				np = length(params(m.dU0))
				c += 1
				eval(nameof(typeof(m.dU0)))(params(m.dU0)[1:end-np+1]...,abs.(θ[c:end])...)
			else 	
				m.dU0
			end

	return dE, dV, dU0
end

function ll_no_searches(m::SD1, zd_h::Vector{T}, β::Vector{T}, ξ::T, dV, dU0, d::DataSD, i::Int, n_draws, complement) where T <: Real
	min_position_discover = d.min_discover_indices[i] 
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
		lb = if h < n_products # not yet last position 
				zd_h[h] - (with_outside_option_dummy ? β[end] : zero(T))
			else # no lower bound if last position 
				- T(MAX_NUMERICAL)
			end

		# Set upper bound for truncation based on position
		ub = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
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
			LL += (1-prob_no_search_given_draw) * prob_u0_in_bounds
		else
			LL += prob_no_search_given_draw * prob_u0_in_bounds
		end
	end

	# println(log(LL / n_draws))
	return log(max(T(ALMOST_ZERO_NUMERICAL), LL / n_draws))
end

function ll_search_no_purchase(m::SD1, zd_h::Vector{T}, β::Vector{T}, ξ::T, dE, dV, dU0, d::DataSD, i::Int, n_draws) where T <: Real

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
				zd_h[h] - (with_outside_option_dummy ? β[end] : zero(T))
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

function ll_purchase(m::SD1, zd_h::Vector{T}, β::Vector{T}, ξ::T, dE, dV, dU0, d::DataSD, i::Int, n_draws) where T <: Real

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
		lb = if h < n_products # not yet last position 
				zd_h[h] - xβ_k
			else # no lower bound if last position 
				- T(MAX_NUMERICAL)
			end

		# Set upper bound for truncation based on position
		ub = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
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
		else
			prob_searches_given_draw *= cdf(dU0, w_k)
		end
		
		# Probabilities for other products
		# Unsearched: P(zs <= min{wj,zd(j-1)})
		# Searched: P(zs > min{wj,zd(j-1)} ∩ u < min{wj,zd(j-1)} )

		for j in 2:h

			xβ_j = @views d.product_characteristics[i][j, :]' * β

			# searched, and discovered before the one that was purchased
			if consideration_set[j] && positions[j] < positions[k]
				prob_searches_given_draw *= prob_search_not_buy(m, xβ_j, ξ, w_k, w_k, dE, dV) 
			# searched, and discovered after the one that was purchased
			elseif consideration_set[j] && j != k &&	positions[j] >= positions[k]
				prob_searches_given_draw *= prob_search_not_buy(m, xβ_j, ξ, wt_k, wt_k, dE, dV) 
			# unsearched & discovered before 
			elseif positions[j] < positions[k]
				prob_searches_given_draw *= cdf(dV, w_k - ξ - xβ_j)
			# unsearched & at same time or after 
			elseif j != k  && positions[j] >= positions[k]
				prob_searches_given_draw *= cdf(dV, wt_k - ξ - xβ_j)
			end
			if prob_searches_given_draw == 0 
				prob_searches_given_draw = T(1e-100) 
				break 
			end
		end	
		LL += prob_searches_given_draw*prob_draws_in_bounds
	end
	return log(max(T(ALMOST_ZERO_NUMERICAL), LL / n_draws))
end

"""
	function prob_search_not_buy(m::SD1, xβ::T, ξ::T, lb::T, ub::T, dE::Normal, dV::Normal)  where T 

Compute probability of searching without buying. This probability is given by P(xβ + ξ + ν_j >= lb ∩ xβ + ξ + ν_j + ε_j < ub). 
"""
@inline function prob_search_not_buy(m::SD1, xβ::T, ξ::T, lb::T, ub::T,
							dE::Normal{R1}, dV::Normal{R2})  where {T <: Real, R1 <: Real, R2 <: Real} 

	σe = std(dE) 
	σv = std(dV)
					
	a = σe > 0 ? (ub - xβ)/σe : one(T)*10000000
	b = σe > 0 ? -σv/σe : -one(T)*10000000
	Y = (lb -ξ -xβ) / σv

	P = cdf_n(a / sqrt(1 + b^2)) - 
			bvncdf(a / sqrt(1 + b^2), Y, -b/sqrt(1 + b^2))
	
	return P::T
end

"""
	function prob_not_search(u, z_j, dV)
Compute probaility of not searching alternaitve, given chosen option has utility u. 
"""
@inline function prob_not_search(m::SD1, u::T, z_j::T, dV) where T 
	return cdf(dV, u - z_j)
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