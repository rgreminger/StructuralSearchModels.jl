abstract type SearchDiscovery end
"""
*Search and Discovery* (SD) core model. This model is a base model for all models that are subtypes of the Search and Discovery model. 

# Fields:  
- `β::Vector{T}`: vector of preference weights. 
- `Ξ::T`: baseline Ξ.
- `ρ::Vector{T}`: parameters governing decrease of Ξ across positions.
- `ξ::T`: baseline ξ.
- `ξρ::Vector{T}`: parameters governing decrease of ξ across positions.
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
*Data* type for the core Search and Discovery model. Indexing is based on sessions. 

# Fields:
- `consumer_ids::Vector{Int}`: consumer id for each session 
- `product_ids::Vector{Vector{Int}}`: product ids for each session in order
- `product_characteristics::Vector{Matrix{T}}`: product characteristics matrix
- `positions::Vector{Vector{Int}}`: positions for each session in order
- `search_paths::Union{Vector{Vector{Int}}, Nothing}`: search paths for each session, can be nothing
- `consideration_sets::Vector{Vector{Bool}}`: consideration sets for each session, booleans whether searched or not
- `purchase_indices::Vector{Int}`: which product within session is purchased
- `stop_indices::Vector{Int}`: which product within session is stopped a
"""
@with_kw mutable struct DataSDCore{T} <: Data where T <: Real
	consumer_ids::Vector{Int}			
	product_ids::Vector{Vector{Int}}					
	product_characteristics::Vector{Matrix{T}}			
	positions::Vector{Vector{Int}}						
	search_paths::Union{Vector{Vector{Int}}, Nothing} 	= nothing
	consideration_sets::Vector{Vector{Bool}}			
	purchase_indices::Vector{Int} 						
	stop_indices::Vector{Int}							

	# Check that all vectors have the same length (number of sessions)
	@assert length(product_ids) == length(product_characteristics) == length(positions) == length(consideration_sets) == length(purchase_indices) == length(stop_indices)
	@assert isnothing(search_paths) || length(search_paths) == length(product_ids)
end

# Define base functions for working with the core data
function length(d::DataSDCore) 
	return length(d.product_ids)
end
function getindex(d::DataSDCore, elements...) 
	i = vcat(elements...)
	
	return DataSDCore(d.consumer_ids[i], d.product_ids[i], d.product_characteristics[i], d.positions[i], d.search_paths == nothing ? nothing : d.search_paths[i], d.consideration_sets[i], d.purchase_indices[i], d.stop_indices[i])
end


# Data generation 
function generate_data(m::SDCore, n_consumers, n_sessions_per_consumer, seed; 
						n_A0 = 1, n_d = 1, 
						products = generate_products(n_consumers*n_sessions_per_consumer), 
						kwargs...) 

	n_sessions = n_consumers * n_sessions_per_consumer 
	
	# Set seed 
	Random.seed!(seed)

	# Unpack products
	product_ids, product_characteristics = products 

	# Create positions based on number of alternatives per position
	positions = [vcat(zeros(Int64, 1 + n_A0), repeat(collect(Int64, 1:(length(product_ids[i]) - 1 - n_A0) / n_d), inner=n_d)) for i in 1:n_sessions]

	# Generate search paths 
	paths, consideration_sets, indices_purchase, indices_stop, utility_purchases = 
		generate_search_paths(m, product_ids, product_characteristics, positions; kwargs...) 

	# Create consumer indices mapping consumers into sessions 
	consumer_ids = repeat(1:n_consumers, n_sessions_per_consumer)

	# Create data object
	data = DataSDCore(consumer_ids, product_ids, product_characteristics, positions, paths, consideration_sets, indices_purchase, indices_stop)

	# Return together with purchase utilities 
	return data, utility_purchases
end

function generate_search_paths(m::SDCore, product_ids, product_characteristics, positions; kwargs...)

	# Extract keyword arguments 
	conditional_on_click = get(kwargs, :conditional_on_click, false)
	conditional_on_click_iter = get(kwargs, :conditional_on_click_iter, 100)

	# Number of consumers and sessions
	max_products_per_session = maximum(length.(product_ids))
	n_sessions  = length(product_ids)

	# Extract other values 
	zdfun = get_functional_form(m.zdfun)
	zsfun = get_functional_form(m.zsfun)

	# Create empty vectors to store outputs
	paths = [zeros(Int, max_products_per_session - 1) for i in 1:n_sessions]
	consideration_sets = [fill(false, max_products_per_session) for i in 1:n_sessions]
	indices_purchase = zeros(Int, n_sessions)
	indices_stop = fill(max_products_per_session, n_sessions)
	utility_purchases = zeros(Float64, n_sessions)


	# Define chunks for parallelization. Each chunk is a range of sessions for which a single task 
	# creates the search path.
	_, data_chunks = get_chunks(n_sessions)

	# Create and define tasks for each chunk
	tasks = map(data_chunks) do chunk 
		Threads.@spawn begin 
			# Create local variables that the thread can work with. Pre-allocation circumvenst allocations in the loop.
			local u 	= zeros(Float64, max_products_per_session)
			local zs 	= zeros(Float64, max_products_per_session)

			# Loop over sessions in the chunk
			for i in chunk

				# Reset 
				u .= typemin(Float64)
				zs .= typemin(Float64)

				fill_path_i!(paths, consideration_sets, indices_purchase, indices_stop, utility_purchases,
								m, i, 
								product_ids, product_characteristics, positions, 
								u, zs, zdfun, zsfun)

				# If conditional on click, iterate until have at least one 
				if conditional_on_click 
					iter = 1 
					while paths[i][1] == 0 && iter <= conditional_on_click_iter
						fill_path_i!(paths, consideration_sets, indices_purchase, indices_stop, utility_purchases, 
										m, i, 
										product_ids, product_characteristics, positions, 
										u, zs, zdfun, zsfun)
						iter += 1 
					end
				end
				
			end
		end
	end
	
	# Execute tasks
	fetch.(tasks) 

	return paths, consideration_sets, indices_purchase, indices_stop, utility_purchases
end


function fill_path_i!(paths, consideration_sets, indices_purchase, indices_stop, utility_purchases, 
						m, i, 
						product_ids, product_characteristics, positions, 
						u, zs, zdfun, zsfun; 
						debug_print = false)

	# draw outside option utility 
	u[1] = rand(m.dU0) + product_characteristics[i][1, end] * m.β[end]

	# Define variables tracking state during search 
	max_u = u[1] 	# current max utility 
	max_zs = zs[1]  # current max search value
	ind_p = 1 # index of product to purchase
	ind_s = 1 # index of product to search
	pos = 0 # current position 
	ns = 0 	# number of searches
	ij = 0  # Index tracking current product 
	n_prod = length(product_ids[i]) # number of products for consumer
	zd = zdfun(m.Ξ, m.ρ, pos) # discovery value for current position
	

	# Fill reservation values in initial awareness set
	for j in eachindex(u) 
		
		if j == 1 # 1st product is outside option and we skip
			continue
		end 
		
		if positions[i][j] > 0 # only for initial awareness set, indicated with position = 0. 
			break 
		end

		# Fill in reservation value for product j
		zs[j] =  zsfun(m.ξ, m.ξρ, pos) + rand(m.dV) + rand(m.dW) 
		for h in eachindex(m.β)
			zs[j] += product_characteristics[i][j, h] * m.β[h]
		end

		# Update max search value and index
		if zs[j] > max_zs 
			max_zs = zs[j]
			ind_s = j 
		end

		# Update index of current product 
		ij = j 
	end

	if i == 1 && debug_print 
		println("############################")
		println("ij = ", ij)
		println("n_prod = ", n_prod)
		println("Initial awareness set: ", zs)
		println("initial zd = ", zd)
		println("initial max_zs = ", max_zs)
		println("initial max_u = ", max_u)
	end


	# Loop through discovering more products and searching 
	while true 
	
		# discover more products
		if max_u < zd && max_zs < zd && pos < positions[i][end]
			pos += 1 

			if i == 1 && debug_print 
				println("############################")
				println("DISCOVERY ")
				println("ij = ", ij)
				println("n_prod = ", n_prod)

				println("Position ", pos)
				println("zd = ", zd)
				println("max_zs = ", max_zs)
				println("max_u = ", max_u)
			end
	

			# Update reservation values for products in next position 
			for j in ij+1:n_prod 
				# Reached next position 
				if positions[i][j] > pos 
					ij = j - 1 # current product is last in position
					break
				end

				# Update reservation value and max 
				zs[j] =  zsfun(m.ξ, m.ξρ, pos) + rand(m.dV) + rand(m.dW) 
				for h in eachindex(m.β)
					zs[j] += product_characteristics[i][j, h] * m.β[h]
				end
				if zs[j] > max_zs 
					max_zs = zs[j]
					ind_s = j
				end
			end

			# Update discovery value
			zd =  zdfun(m.Ξ, m.ρ, pos) 

		# Stop and buy 
		elseif ( max_u >= zd || pos == positions[i][end] ) && max_u > max_zs  
			if i == 1 && debug_print
				println("############################")
				println("PURCHASE  ")
				println("ij = ", ij)
				println("n_prod = ", n_prod)
				println("Position ", pos)
				println("zd = ", zd)
				println("max_zs = ", max_zs)
				println("max_u = ", max_u)
				println("ind_p = ", ind_p)
			end

			# Fill in purchase and stop indices 
			indices_purchase[i] = ind_p 
			indices_stop[i] = ij 
			utility_purchases[i] = max_u
			break 

		# search next product 
		else 
			if i == 1 && debug_print
				println("############################")
				println("SEARCH  ")
				println("Position ", pos)
				println("zd = ", zd)
				println("max_zs = ", max_zs)
				println("max_u = ", max_u)
				println("ind_s = ", ind_s)	
			end

			# Increase number of searches and fill in next search 
			ns += 1 
			paths[i][min(ns, end)] = ind_s 
			consideration_sets[i][ind_s] = true
			
			# Set search value to neg. infinity so that it is not searched again
			zs[ind_s] = typemin(eltype(zs))
			
			# Get utility of searched product 
			u[ind_s] =  rand(m.dE) + rand(m.dV) 
			for h in eachindex(m.β) 
				u[ind_s] += product_characteristics[i][ind_s, h] * m.β[h] 
			end	
			# Update max utility 
			if u[ind_s] > max_u 
				max_u = u[ind_s]
				ind_p = ind_s 
			end
			# Find next product to search. Note, all undiscovered and already-searched products have zs=-Inf, so that never chosen. 
			max_zs, ind_s = findmax(zs) 
		
		end
	end
	return nothing
end

