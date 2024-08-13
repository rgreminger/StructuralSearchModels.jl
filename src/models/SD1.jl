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
function prepare_arguments_likelihood(d::DataSD, m::SD1, estimator::Estimator; kwargs...)
	
	# Get functional forms 
	zdfun = get_functional_form(m.zdfun)
    zsfun = get_functional_form(m.zsfun)

    return zdfun, zsfun 
end

###############################################################################
# Likelihood wrapper functions across models 
###############################################################################
function loglikelihood(m::M, θ::Vector{T}, data::DataSD, estimator::Estimator, args...; kwargs...) where {M <: SD1, T <: Real}
	
	# Extract arguments 
	zdfun, zsfun = args  

	# Extract parameters implied by θ
	parameters  = extract_parameters(m, θ; kwargs...)
	dists = extract_distributions(m, θ; kwargs...)

	# Pre-compute search and discovery values across positions -> same for all consumers 
	zd_h = [zdfun(Ξ, ρ, h) for h in 1:max_n_products]
	ξ_h  = [zsfun(ξ, ξρ, h) for h in 1:max_n_products]

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
			local Li = zeros(T, n_draws_purchase, max_n_products) # use no. purchase draws since larger than nDraws. Rest of array will be filled with zeros. 
			local r = zeros(T,6) 
			local r1 = zeros(T,3)

			for i in chunk  # Iterate over consumers in chunk 
				r .= zero(T) 
				r1 .= zero(T)
				Li .= zero(T)

				# Do inner likelihood calculations based on pre-allocated arrays
				if nS[i] == 0 	# Case 1: no clicks (implies also no purchase)
					@views ll_no_searches!(m, Li, r, data, i, parameters...,) 
					L += calc_logsum(Li, n_draws)
				elseif purch[i] == 1 # Case 2: Some clicks but no purchase 
					@views ll_search_no_purchase!(m, Li, r, data, i, parameters...)
					L += calc_logsum(Li, n_draws)
				else 	# Case 3: Purchase a product 
					@views ll_purchase!(m, Li, r, data, i, parameters...)
					L += calc_logsum(Li, n_draws_purchase)
				end
				
			end

			return L # Return likelihood for chunk
		end
	end

	LL1 = sum(fetch.(tasks)) 

	LL2 = 	if conditional_on_search  
				n_initial = typeof(m) <: Weitzman ? maxprod : 1 + m.nA0
				
				tasks = map(data_chunks) do chunk 
					Threads.@spawn begin 

						local L = zero(T)
						local Li = zeros(T, n_draws_purchase, max_n_products) # use no. purchase draws since larger than nDraws. Rest of array will be filled with zeros. 
						local r = zeros(T,5) 

						for i in chunk  # Iterate over consumers in chunk 
							Li .= zero(T)
							r .= zero(T) 	
							@views 	ll_no_searches!(m, Li, r, data, i, parameters...; complement = true) 
							L += calc_logsum(Li,nDraws)
						end

						return L 
					end
				end
				
				sum(fetch.(tasks))
			else
				zero(T)
			end

	LL = LL1 - LL2 

	# prevent Inf values, helps AD
	if isinf(LL) || isnan(LL) || LL >= 0 
		return -T(1e100)
	else
		return LL
	end
end


# # Case 1: no clicks (implies also no purchase)
# function logliki1!(m::M,Li,r,
# 					β,ξ,zd::AbstractArray{T,2},chars,
# 					dists,minDiscover,
# 					ic,i,complement::Bool) where {M <: SearchDiscovery, T <: Real}

# 	_,iz2,izr,ddr = getranges(ic,i,minDiscover,m, false) 

# 	_logliki1!(m,Li,r,
# 					β,ξ,zd,chars,
# 					dists...,
# 					ic,iz2,ddr,izr,complement)  

# 	if last(izr) != iz2 
# 		_logliki1!(m,Li,r,
# 					β,ξ,zd,chars,
# 					dists...,
# 					ic,iz2,ddr,iz2,complement)  
# 	end
# end

# # Case 2: Some clicks but no purchase 
# function logliki2!(m::M,Li,r,
# 					β,ξ,zd,chars,searched,
# 					dists,minDiscover,
# 					ic,i) where {M <: SearchDiscovery} 

# 	_,iz2,izr,ddr = getranges(ic,i,minDiscover,m, false) 
	
# 	_logliki2!(m,Li,r,β,ξ,zd,chars,searched,
# 					dists...,
# 					ic,i,iz2,izr,ddr)
# 	if last(izr) != iz2 # Case where with last discovery fewer elements are discovered
# 		_logliki2!(m,Li,r,β,ξ,zd,chars,searched,
# 					dists...,
# 					ic,i,iz2,iz2,ddr)
# 	end
# end

# # Case 3: Purchase a product 
# function logliki3!(m::M,Li,r::AbstractArray{T,1},r1,
# 					β,ξ,zd::AbstractArray{T,2},chars,searched,purch,
# 					dists,minDiscover,maxClick,
# 					ic,i) where {M <: SearchDiscovery, T <: Real} 

# 	iz1,iz2,izr,ddr = getranges(ic,i,minDiscover,m, true) 
# 	e = zeros(T,2)
						
# 	_logliki3!(m,Li,r,r1,
# 				β,ξ,zd,chars,searched,purch,
# 				dists...,maxClick,
# 				ic,i,iz1,iz2,izr,e,ddr)
# 	if last(izr) != iz2 # Case where with last discovery fewer elements are discovered
# 		_logliki3!(m,Li,r,r1,
# 					β,ξ,zd,chars,searched,purch,
# 					dists...,maxClick,
# 					ic,i,iz1,iz2,iz2,e,ddr)
# 	end
# end

