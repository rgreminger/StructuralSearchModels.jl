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
				β = !fixed_parameters[1] ? m.β : nothing
				ξ = !fixed_parameters[2] ? m.ξ : nothing
				Ξ = !fixed_parameters[3] ? m.Ξ : nothing
				ρ = !fixed_parameters[4] ? m.ρ : nothing
				θ = vcat(β, ξ, Ξ, ρ)

			end
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

###############################################################################
# Likelihood wrapper functions across models 
###############################################################################
function loglikelihood(θ::Vector{T}, model::M, estimator::SmoothMLE, data::DataSD, args...; kwargs...) where {M <: SD1, T <: Real}
	
	# Extract arguments 
	max_n_products, zdfun  = args  

	# Extract parameters implied by θ 
	m_hat = construct_model_from_pars(θ, model; kwargs...)

	# Pre-compute search and discovery values across positions -> same for all consumers 
	zd_h = [zdfun(m_hat.Ξ, m_hat.ρ, h) for h in 1:max_n_products]

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
					L += ll_no_searches(m_hat, zd_h, data, i, n_draws, false) 
				elseif data.purchase_indices[i] == 1 # Case 2: Some clicks but no purchase 
					# @views ll_search_no_purchase!(m, Li, r, data, i, parameters...)
					# L += calc_logsum(Li, n_draws)
				else 	# Case 3: Purchase a product 
					# @views ll_purchase!(m, Li, r, data, i, parameters...)
					# L += calc_logsum(Li, n_draws_purchase)
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
							L += ll_no_searches(m_hat, zd_h, data, i, n_draws, true) 
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
		ρ = θ[ind_current:ind_current+n_ρ] ; ind_current += n_ρ
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

function ll_no_searches(m::SD1{T}, zd_h, d::DataSD, i::Int, n_draws, complement) where T <: Real

	min_position_discover = d.min_discover_indices[i] 
	n_products = length(d.product_ids[i])
	positions = @views d.positions[i]
	with_outside_option = d.product_ids[i][1] == 0

	LL = zero(T)
	
	for dd in 1:n_draws, h in min_position_discover:n_products

		# If not last product in same position or last product, skip 
		if h < n_products && positions[h] == positions[h+1]
			continue
		end

		# Set lower bound for truncation based on position 
		lb = if h < n_products # not yet last position 
				zd_h[h] - (with_outside_option ? m.β[end] : zero(T))
			else # no lower bound if last position 
				- MAX_NUMERICAL 
			end

		# Set upper bound for truncation based on position
		ub = if positions[h] == 0 # first no upper bound if last click in initial awareness set (position 0)
				MAX_NUMERICAL 
			else 
				zd_h[h]  - (with_outside_option ? m.β[end] : zero(T)) # Max accounts for case where nA0 < nd 
			end

		# Get probability of u0 in bounds and draw for u0 
		prob_u0_in_bounds = trunc_cdf(m.dU0, lb, ub) 
		u0_draw = rand_trunc(m.dU0, lb, ub) + (with_outside_option ? m.β[end] : zero(T))
		
		# Initialize for probability
		prob_no_search_given_draw = one(T)

		# Loop over products up to one discovered 
		for j in 2:h 
			zs_j = @views d.product_characteristics[i][j, :]' * m.β + m.ξ
			prob_no_search_given_draw *= cdf(m.dV, u0_draw - zs_j)

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
	return log(LL / n_draws)
end

# function _logliki2!(m::SD3,Li,r,
# 						β::AbstractArray{T,2},ξ,zd,chars,searched,
# 						dE, dV, dU0,
# 						ic,i,iz2,izr,ddr)  where T 

# 	activeLB,outDummy,nA0,nd = m.activeLB,m.outDummy,m.nA0,m.nd
# 	maxnum = T(1e100)

# 	for (idd,dd) in enumerate(ddr), (iizr,iz) in enumerate(izr)
# 		r[end] = one(T)	
# 		lb = if iz < iz2 || activeLB  
# 					zd[dd,iz] - (outDummy ? β[dd,end] : 0)
# 				else
# 					-maxnum 
# 				end

# 		ub = if iz <= 1 + nA0 # first no upper bound if last click in initial awareness set 
# 			maxnum 
# 			else 
# 			zd[dd,max(iz-nd,1)] - (outDummy ? β[dd,end] : 0) # Max accounts for case where nA0 < nd 
# 		end

# 		truncCdf = trunc_cdf(dU0,lb,ub) 

# 		r[1] = rand_trunc(dU0,lb,ub) + (outDummy ? β[dd,end] : 0)

# 		for h in 2:iz 
# 			r[2] = zero(T)
# 			for k in axes(β,2)
# 				r[2] += chars[ic[h],k] * β[dd,k] 
# 			end

# 			if searched[i][h]
# 				r[end] *= cdf_ZWU(m,r[2],ξ[dd,h],r[1],dE,dV)
# 			else
# 				r[end] *= cdf(dV,r[1] - ξ[dd,h] - r[2])
# 			end
			
# 			if r[end] == 0 
# 				r[end] = T(1e-100) 
# 				break
# 			end
# 		end
# 		Li[idd,iizr] = r[end] * truncCdf 
# 	end
# end

# function _logliki3!(m::SD3,Li,r::AbstractArray{T,1},r1,
# 						β,ξ,zd::AbstractArray{T,2},chars,searched,purch,
# 						dE,dV,dU0,
# 						maxClick,
# 						ic,i,iz1,iz2,izr,e,ddr) where T 

# 	activeLB,outDummy,nA0,nd = m.activeLB,m.outDummy,m.nA0,m.nd
# 	maxnum = T(1e100)

# 	@inbounds for (idd,dd) in enumerate(ddr), (iizr,iz) in enumerate(izr)
# 		r1 .= zero(T) 
# 		for ddd in 1:2
# 			r[end] = one(T)

# 			# xb of purchased 
# 			r[1] = zero(T)
# 			for k in axes(β,2)
# 				r[1] += chars[ic[purch],k] * β[dd,k] 
# 			end
# 			# Bounds for draws to fit into different zd 
# 			ub = 	if iz <= 1 + nA0  # Last click in initial awareness set -> initially no upper bound 
# 				maxnum 
# 			elseif iz == iz1 && checksamepos(purch,maxClick,nA0,nd) # if last clicked purchased -> initially no upper bound 
# 				maxnum 
# 			else
# 				zd[dd,iz-nd] - r[1]
# 			end

# 			lb = 	if iz < iz2 || activeLB 
# 						zd[dd,iz] - r[1]
# 					else
# 						-maxnum 
# 					end
			
# 			# Fill 
# 			# r[1] = wj 
# 			# r[2] = uj 
# 			if ddd == 1 # e < ξ
# 					e[1] =  rand_trunc(dE,-one(T)*maxnum,ξ[dd,purch])
# 					r[1] += e[1] + rand_trunc(dV,lb-e[1],ub-e[1])
# 					r[2] = r[1] 
# 					truncCdf = trunc_cdf(dE,-one(T)*maxnum,ξ[dd,purch]) * 
# 									trunc_cdf(dV,lb-e[1],ub-e[1])
# 			else # e >= ξ
# 					e[1] = rand_trunc(dE,ξ[dd,purch],one(T)*maxnum)
# 					r[1] += ξ[dd,purch] + 
# 								rand_trunc(dV,lb-ξ[dd,purch],ub-ξ[dd,purch])
# 					r[2] = r[1] - ξ[dd,purch] + e[1] 
# 					truncCdf = trunc_cdf(dE,ξ[dd,purch],one(T)*maxnum) * 
# 									trunc_cdf(dV,lb-ξ[dd,purch],ub-ξ[dd,purch])
# 			end		
			
# 			# Get values only if last click not in initial awareness set 
# 			r[3] = r[1] # min{wj,zd(j-1)}
# 			r[4] = r[2] # min{uj,zd(j-1)}
# 			if iz1 > 1 + nA0 
# 				r[3] = min(r[3],zd[dd,iz-nd])
# 				r[4] = min(r[4],zd[dd,iz-nd])
# 			end

# 			# Get probability outside option P(u0 < min{wj,zd(j-1)})
# 			if outDummy
# 				r[end] *= cdf(dU0,r[3]- β[dd,end])
# 			else
# 				r[end] *= cdf(dU0,r[3])
# 			end
			
# 			# Probabilities for other products
# 			# Unsearched: P(zs <= min{wj,zd(j-1)})
# 			# Searched: P(zs > min{wj,zd(j-1)} ∩ u < min{wj,zd(j-1)} )

# 			for h in 2:iz
# 				# Get xβ
# 				r[5] = zero(T)
# 				for k in axes(β,2)
# 					r[5] += chars[ic[h],k] * β[dd,k] 
# 				end
# 				# searched, and discovered before the one that was purchased
# 				if searched[i][h] && 
# 						get_pos(h,nA0,nd) < get_pos(purch,nA0,nd) 
# 					r[end] *= cdf_ZWU(m,r[5],ξ[dd,h],r[3],dE,dV) 
# 				# searched, and discovered after the one that was purchased
# 				elseif searched[i][h] && h!= purch &&
# 					get_pos(h,nA0,nd) >= get_pos(purch,nA0,nd) 
# 					r[end] *= cdf_ZWU(m,r[5],ξ[dd,h],r[1],dE,dV) 
# 				# unsearched & discovered before 
# 				elseif h != purch  && get_pos(h,nA0,nd) < get_pos(purch,nA0,nd)
# 					r[end] *= cdf(dV,r[3] - ξ[dd,h] - r[5])
# 				# unsearched & at same time or after 
# 				elseif h != purch  && get_pos(h,nA0,nd) >= get_pos(purch,nA0,nd)
# 					r[end] *= cdf(dV,r[1] - ξ[dd,h] - r[5])
# 				end
# 				if r[end] == 0 
# 					r[end] = T(1e-100) 
# 					break 
# 				end
# 			end	
# 			r1[ddd] = r[end]*truncCdf
# 		end
# 		Li[idd,iizr] = sum(@views r1) 
# 	end
# end


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