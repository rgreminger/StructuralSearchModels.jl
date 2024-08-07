abstract type SearchDiscovery end
"""
*Search and Discovery* (SD) core model. This model is a base model for all models that are subtypes of the Search and Discovery model. 

# Fields:  
- `β::Vector{T}`: vector of preference weights. 
- `Ξ::T`: baseline Ξ.
- `ρ::Vector{T}`: parameters governing decrease of Ξ across positions.
- `ξ::T`: baseline ξ.
- `ξρ::Vector{T}`: parameters governing decrease of ξ across positions.
- `n_d::Int64`: number of alternatives per position. 
- `n_A0::Int64`: number of alternatives in the initial awareness set. 
- `dE::Distribution`: distribution of ε_{ij}.
- `dV::Distribution`: distribution of ν_{ij}.
- `dU0::Distribution`: distribution of u_{i0}. 
- `dW::Distribution`: distribution of ω_{ij}.
- `zdfun::String`: select functional form f(h, Ξ, ρ) that determines the discovery value in position h. 
- `zsfun::String`: select functional form f(h, ξ, ξρ) that determines the search value in position h.
- `unobserved_heterogeneity::Dict`: dictionary of unobserved heterogeneity parameters and options. 
"""
@with_kw mutable struct SDCore{T} <: SearchDiscovery where T <: Real
	β::Vector{T} 
	Ξ::T
	ρ::Vector{T}
	ξ::T
	ξρ::Vector{T}
	dE::Distribution
	dV::Distribution
	dU0::Distribution
	dW::Distribution
	zdfun::String 
	zsfun::String
	unobserved_heterogeneity::Dict = Dict()
end 



""" 
*Data* type for the core Search and Discovery model. 

"""
@with_kw mutable struct DataSDCore{T} <: Data where T <: Real
	consumer_indices::Vector{UnitRange{Int}}			# which sessions belong to which consumer 
	product_ids::Vector{Vector{Int}}					# product ids for each session in order 
	product_characteristics::Vector{Matrix{T}}			# product characteristics matrix 
	positions::Vector{Vector{Int}}						# positions for each session in order
	search_paths::Union{Vector{Vector{Int}}, Nothing} 	= nothing # search paths for each session, can be nothing 
	consideration_sets::Vector{Vector{Bool}}			# consideration sets for each session, booleans whether searched or not 
	purchase_indices::Vector{Int} 						# which product within session is purchased 
	stop_indices::Vector{Int}							# which product within session is stopped a

	# Check that all vectors have the same length (number of sessions)
	@assert length(product_ids) == length(product_characteristics) == length(positions) == length(consideration_sets) == length(purchase_indices) == length(stop_indices)
	@assert isnothing(search_paths) || length(search_paths) == length(product_ids)
end


# Data generation 
function generate_data(m::SDCore, n_consumers, n_sessions_per_consumer, seed; 
						n_A0 = 1, n_d = 1, 
						products = generate_products(n_consumers*n_sessions_per_consumer), kwargs...) 

	n_sessions = n_consumers * n_sessions_per_consumer 
	
	# Set seed 
	Random.seed!(seed)

	# Unpack products
	product_ids, product_characteristics = products 

	# Create positions based on number of alternatives per position
	positions = [vcat(ones(Int64, 1 + n_A0), repeat(collect(Int64, 2:(length(product_ids[i]) - 1 - n_A0) / n_d), inner=n_d)) for i in 1:n_sessions]

	# Generate search paths 
	# paths, consideration_sets, indices_purchase, indices_stop = generate_search_paths(m, product_ids, product_characteristics, positions) 




	return product_ids, product_characteristics, positions
end

function generate_search_paths(m::SDCore, product_ids, product_characteristics, positions) 

	# Number of consumers and sessions
	max_products_per_session = maximum(length.(product_ids))
	n_sessions  = length(product_ids)

	# Extract other values 
	zdfun = get_functional_form(m.zdfun)
	zsfun = get_functional_form(m.zsfun)

	# Create empty vectors to store paths, searched, purchase and stop indices
	paths = [zeros(Int, max_products_per_session) for i in 1:n_sessions]
	consideration_sets = [fill(false, max_products_per_session) for i in 1:n_sessions]
	indices_purchase = zeros(Int, n_sessions)
	indices_stop = fill(max_products_per_session, n_sessions)

	# Define chunks for parallelization. Each chunk is a range of sessions for which a single task 
	# creates the search path.
	chunk_size, data_chunks = get_chunks(n_sessions)

	# Create and define tasks for each chunk
	tasks = map(data_chunks) do chunk 
		Threads.@spawn begin 
			# Create local variables that the thread can work with. Pre-allocation circumvenst allocations in the loop.
			local u 	= zeros(Float64, n_products_per_session)
			local zs 	= zeros(Float64, n_products_per_session)
			local e 	= zeros(Float64, n_products_per_session)
			local v 	= zeros(Float64, n_products_per_session)
			local w 	= zeros(Float64, n_products_per_session)
			local u0 	= zeros(Float64, n_products_per_session)

			# Loop over sessions in the chunk
			for i in chunk
				fill_path_i!(paths, consideration_sets, indices_purchase, indices_stop, m,
								i, product_ids, product_characteristics, positions, u, zs, e, v, w, u0, zdfun, zsfun)
			end

		end
	end
	
	# Execute tasks
	fetch.(tasks) 

	return paths, consideration_sets, indices_purchase, indices_stop
end


# function fill_path_i!(paths, consideration_sets, indices_purchase, indices_stop, m, 
# 	i, product_ids, product_characteristics, positions, u, zs, e, v, w, u0, zdfun, zsfun)

# 	# outside option utility 
# 	u[1] = rand(m.dU0) + rand(m.dV) + m.β[1]
# 	u[1] = u0[s[1],k]
# 	if m.outDummy
# 		u[s[1]] += b[end]
# 	end

# 	# max utility and search value , initially outside option 
# 	max_u = u[s[1]] 
# 	max_zs = zs[s[1]]  # = -Inf 

# 	# search and purchase indicators (where max is) 
# 	ind_p = s[1]
# 	ind_s = s[1]

# 	# index and position 
# 	pos = 0 
# 	ns = 0 # number of searches 

# 	# Discovery value (varies with position)
# 	ρ = isempty(ρ) ? [0.0] : ρ
# 	zd = zdfun(Ξ[1],ρ,0)

# 	# Index tracking product for reservation value 
# 	ij = 0  

# 	ls = length(s) # number of products for consumer

# 	# Reservation values in initial awareness set
# 	for j in s[2:min(1+m.nA0,ls)]  # 1 is outside
# 		ij += 1 
# 		zs[j] =  zsfun(ξ[1],ξρ,ij-1) + v[j,k] + w[j,k]
# 		for h in eachindex(b)
# 			zs[j] += chars[j,h] * b[h]
# 		end
# 		if zs[j] > max_zs 
# 			max_zs = zs[j]
# 			ind_s = j 
# 		end
# 	end
# 	# Index tracking disocveries (how far discovered )
# 	i = 1+m.nA0
# 	# Loop through discovering more products and searching 
# 	while true 
# 		if max_u < zd && max_zs < zd && i < ls # discover more 
# 			pos += 1 
# 			for j in s[i+1:min(i+m.nd,ls)]
# 				ij += 1 
# 				zs[j] =  zsfun(ξ[1],ξρ,ij-1) + v[j,k] + w[j,k]
# 				for h in eachindex(b)
# 					zs[j] += chars[j,h] * b[h]
# 				end
# 				if zs[j] > max_zs 
# 					max_zs = zs[j]
# 					ind_s = j
# 				end
# 			end
# 			i += m.nd

# 			# Note: if position-specific mean not provided for all 
# 			# 		positions, uses last one provided for all subsequent 
# 			zd =  zdfun(Ξ[1],ρ,pos ) 
# 		elseif ( max_u >= zd || i>=length(s) ) && max_u > max_zs # stop and buy 
# 			purch[iCons] = ind_p - s[1] + 1 
# 			stop[iCons] = pos 
# 			path[ns+1,iCons] = 0 
# 			break 
# 		else # search 
# 			ns += 1 
# 			path[ns,iCons] = ind_s - s[1] + 1 
# 			u[ind_s] =  e[ind_s,k] + w[ind_s,k]
# 			for h in eachindex(b) 
# 				u[ind_s] += chars[ind_s,h] * b[h] 
# 			end	
# 			if u[ind_s] > max_u 
# 				max_u = u[ind_s]
# 				ind_p = ind_s 
# 			end
# 			zs[ind_s] = -Inf 
# 			max_zs,ind_s = StructSearch.findmax_range(zs,s[1:min(i,ls)])
# 		end
# 	end
# 	return nothing
# end

