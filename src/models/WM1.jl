"""
*Weitzman* WM1 model with the following parameterization: 
- uᵢⱼ = xⱼ'β + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ = xⱼ'β + ξ(h) + νᵢⱼ 
- uᵢ₀ = x₀'β + η , η_i ~ dU0
- ξ(h) = zsfun(ξ, ρ, pos)


# Fields:  
- `β::Vector{T}`: vector of preference weights. 
- `cs`: search costs. Initialized as nothing by to avoid computational cost. Can be updated through `calculate_costs!(m, data; kwargs...)`. 
- `ξ::T`: baseline ξ.
- `ρ::Vector{T}`: parameters governing decrease of ξ across positions.
- `dE::Distribution`: distribution of ε_{ij}.
- `dV::Distribution`: distribution of ν_{ij}.
- `dU0::Distribution`: distribution of u_{i0}. 
- `zsfun::String`: select functional form f(ξ, ρ, h) that determines the search value in position h. 
- `unobserved_heterogeneity::Dict`: dictionary of unobserved heterogeneity parameters and options. Currently not used. 
"""

@with_kw mutable struct WM1{T} <: SD where T <: Real
	β::Vector{T} 
	ξ::T
	ρ::Vector{T}
	cs::Union{T, Nothing}	= nothing 
	cs_h::Union{Vector{T}, Nothing}	= nothing 
	dE::Distribution
	dV::Distribution
	dU0::Distribution
	zsfun::String 
	unobserved_heterogeneity::Dict = Dict()
end 

# Functions from SDCore to WM1 
SDCore(m::WM1) = SDCore(; β = m.β, Ξ = Inf, ρ = [-1e-100], ξ = m.ξ, ξρ = m.ρ, cd = 0.0, cs = m.cs, cs_h = m.cs_h, dE = m.dE, dV = m.dV, dU0 = m.dU0, dW = Normal(0, 0), zdfun = "linear", zsfun = m.zsfun,  unobserved_heterogeneity = m.unobserved_heterogeneity)

calculate_welfare(model::WM1, data::DataSD, n_sim; method = "effective_values", kwargs...) = calculate_welfare(SDCore(model), data, n_sim; method = method, kwargs...)

function calculate_costs!(m::WM1, d; 
	force_recompute = true,
	cd_kwargs...)

	m1 = SDCore(m) 
	calculate_costs!(m1, d, 1; force_recompute, cd_kwargs...) 

	m.cs = m1.cs
	m.cs_h = m1.cs_h

	return nothing 
end

generate_data(m::WM1, n_consumers, n_products; kwargs...) = generate_data(SDCore(m), n_consumers, n_products; kwargs...)
generate_data(m::WM1, data::DataSD; kwargs...) = generate_data(SDCore(m), data; kwargs...)

evaluate_fit(m::WM1, data::DataSD, n_sim; kwargs...) = evaluate_fit(SDCore(m), data, n_sim; kwargs...)



# Estimation 
function prepare_arguments_likelihood(m::WM1, estimator::Estimator, d::DataSD; kwargs...) 

	# Get functional forms 
	zsfun = get_functional_form(m.zsfun)

	# Get maximum number of products
	data_arguments = prepare_data_arguments_likelihood(d) 

	# Keep fixed seed: either random or provided by kwargs 
	seed = get(kwargs, :seed, rand(1:10^9))

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

function construct_model_from_pars(θ::Vector{T}, m::WM1; kwargs...) where T <: Real

	# Extract parameters from vector, some may be fixed through kwargs 
	β, _, _, ξ,	ρ, ind_last_par  = extract_parameters(m, θ; kwargs...)
	dE, dV, dU0 = extract_distributions(m, θ, ind_last_par; kwargs...)

	# Construct model from parameters 
	m_new = WM1{T}(; β, ξ, ρ, dE, dV, dU0, zsfun = m.zsfun)

    return m_new 
end

function extract_parameters(m::WM1, θ::Vector{T}; kwargs...) where T <: Real

	n_beta = length(m.β)
	n_ρ = length(m.ρ)

	# track where in parameter vector we are and move it. 
	ind_current = 1 

	# Default: estimate all parameters
	if !haskey(kwargs, :fixed_parameters)
		β = θ[1:n_beta] ; ind_current += n_beta 
		ξ = θ[ind_current] ; ind_current += 1
		ρ = θ[ind_current:ind_current + n_ρ - 1] ; ind_current += n_ρ
		return β, nothing, nothing, ξ, ρ, ind_current
	end

	# If keyword supplied, don't estimate parameters indicated in fixed_parameters
	fixed_parameters = get(kwargs, :fixed_parameters, nothing)
	β = if fixed_parameters[1];  T.(m.β) ; else ; ind_current += n_beta; θ[1:n_beta] ; end
	ξ = if fixed_parameters[2];  T(m.ξ) ; else ; ind_current += 1 ; θ[ind_current - 1] ; end
	ρ = if fixed_parameters[4];  T.(m.ρ) ; else ; ind_current += n_ρ ;  θ[ind_current:ind_current + n_ρ - 1 - 1] ; end
	return β, nothing, nothing, ξ, ρ, ind_current
end

function ll_no_searches(m::WM1, zd, ξj::Vector{T}, β::Vector{T}, dV, dU0, d::DataSD, i::Int, n_draws, complement) where T <: Real

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
				prob_no_search_given_draw = T(MAX_NUMERICAL)  # adding this helps AD 
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

function ll_search_no_purchase(m::WM1, zd, ξj::Vector{T}, β::Vector{T}, dE, dV, dU0, d::DataSD, i::Int, n_draws) where T <: Real

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
				prob_searches_given_draw *= prob_search_not_buy(m, xβ, ξj[j], u0_draw, u0_draw, dE, dV)
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

function ll_purchase(m::WM1, zd, ξj::Vector{T}, β::Vector{T}, dE, dV, dU0, d::DataSD, i::Int, n_draws) where T <: Real

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
				e =  rand_trunc(dE, -one(T)*MAX_NUMERICAL, ξj[k])
				u_k = xβ_k + e + rand(dV) 
				z_k = u_k - e + ξj[k] # this way v draw still stored in z_k
				prob_draw_in_bounds = trunc_cdf(dE, -one(T)*MAX_NUMERICAL, ξj[k]) 
		else # e >= ξ
				e = rand_trunc(dE, ξj[k], one(T)*MAX_NUMERICAL)
				z_k = xβ_k + ξj[k] + rand(dV) 
				u_k = z_k - ξj[k] + e # this way v draw still stored in z_k 
				prob_draw_in_bounds = trunc_cdf(dE, ξj[k], one(T)*MAX_NUMERICAL) 
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
				prob_searches_given_draw *= prob_search_not_buy(m, xβ_j, ξj[j], wt_k, wt_k, dE, dV) 
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