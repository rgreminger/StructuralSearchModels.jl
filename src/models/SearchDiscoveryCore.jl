abstract type SD end

"""
*Search and Discovery* (SD) core model. This model is a base model for all models that are subtypes of the Search and Discovery model. 

# Fields:  
- `β::Vector{T}`: vector of preference weights. 
- `cs`: search costs. Initialized as nothing by to avoid computational cost. Can be updated through `calculate_costs!(m, data; kwargs...)`. 
- `cd`: discovery costs. Initialized in the same way as `cs`, and is also added in `calculate_costs!(m, data; kwargs...)`.
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
@with_kw mutable struct SDCore{T} <: SD where T <: Real
	β::Vector{T} 
	cs::Union{T, Nothing}	= nothing 
	cd::Union{T, Nothing}	= nothing 
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

	@assert ρ[1] < 0 "ρ[1] must be negative for decreasing discovery value across positions."
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
@with_kw mutable struct DataSD{T} <: Data where T <: Real
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
function length(d::DataSD) 
	return length(d.product_ids)
end
function getindex(d::DataSD, elements...) 
	i = vcat(elements...)
	
	return DataSD(d.consumer_ids[i], d.product_ids[i], d.product_characteristics[i], d.positions[i], d.search_paths == nothing ? nothing : d.search_paths[i], d.consideration_sets[i], d.purchase_indices[i], d.stop_indices[i])
end

function eachindex(d::DataSD) 
	return eachindex(d.product_ids)
end

function sessions_with_clicks(d::DataSD) 
	has_click = x -> x[1] > 0
	return findall(has_click, d.search_paths)
end

function sessions_with_purchase(d::DataSD) 
	has_purchase = x -> x > 1
	return findall(has_purchase, d.purchase_indices)
end

# Data generation 
function generate_data(m::SDCore, n_consumers, n_sessions_per_consumer; 
						n_A0 = 1, n_d = 1, 
						products = generate_products(n_consumers*n_sessions_per_consumer), 
						kwargs...) 

	n_sessions = n_consumers * n_sessions_per_consumer 
	
	# Set seed (is stable across threads) 
	set_seed(kwargs)

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
	data = DataSD(consumer_ids, product_ids, product_characteristics, positions, paths, consideration_sets, indices_purchase, indices_stop)

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

function generate_data(m::SDCore, d::DataSD; kwargs...) 
	
	# Set seed (is stable across threads) 
	set_seed(kwargs)

	# Generate new paths
	paths, consideration_sets, indices_purchase, indices_stop, utility_purchases = 
		generate_search_paths(m, d.product_ids, d.product_characteristics, d.positions; kwargs...) 

	# Update and return data object
	data = DataSD(d.consumer_ids, d.product_ids, d.product_characteristics, d.positions, paths, consideration_sets, indices_purchase, indices_stop)

	return data, utility_purchases

end

# Cost computations 
"""
	caculate_costs!(m::SD, d, n_draws_cd, seed; force_recompute = true)

Calculate search and discovery costs for the Search and Discovery model `m` and data `d`. Uses `n_draws` to calculate the discovery costs using the distribution of characteristics in the data. If `force_recompute` is true, the costs are recomputed even if they are already present in the model.
	
"""
function calculate_costs!(m::SD, d, n_draws_cd; 
							force_recompute = true,
							cd_kwargs...)
	# Search costs 
	if isnothing(m.cs) || force_recompute
		m.cs = calculate_search_cost(m) 
	end
	# Discovery costs
	if isnothing(m.cd) || force_recompute
		m.cd = calculate_discovery_cost(m, d, n_draws_cd; cd_kwargs...)
	end
	return nothing 
end

""" 
	calculate_search_cost(m::SearchDiscovery)

Calculate search costs for the SD model given `ξ` and the distribution of ε `dE`. 
"""
function calculate_search_cost(m::SD)
	ξ = m.ξ
	F = m.dE

	return quadgk(e->(1-cdf(F ,e)), ξ, maximum(F))[1]
end


"""
	calculate_discovery_cost(m::SD, d::DataSD, n_draws; kwargs...)

"""

function calculate_discovery_cost(m::SD, d::DataSD, n_draws; kwargs...) 

	# Set seed (is stable across threads) 
	set_seed(kwargs)	

	# characteristics matrix without outside option 

	# Get distribution of expected utilities xβ across all products in data 
	chars = vcat([d.product_characteristics[i][d.product_ids[i] .> 0, :] for i in eachindex(d)]...) # excludes outside option 
	xβ = chars * m.β

	# Get discovery value at position where beliefs are correct
	zdfun = get_functional_form(m.zdfun)
	position_at_which_correct_beliefs = get(kwargs, :position_at_which_correct_beliefs, calculate_mean_position(d))
	zd = zdfun(m.Ξ, m.ρ, position_at_which_correct_beliefs)

	# Calculate discovery costs by sampling effective values using the empirical xβ distribution and taste shock assumptions 
	W = zeros(Float64, n_draws)
	fill_values_cd_compute!(W, m, xβ, zd) 

	# Return mean of the values 
	return mean(W) 

end

function fill_values_cd_compute!(W, m, xβ, zd::T) where T <: Real 

	_, data_chunks = get_chunks(length(W))

	# Create and define tasks for each chunk
	tasks = map(data_chunks) do chunk 
		Threads.@spawn begin 
			for i in chunk 
				e = rand(m.dE)
				v = rand(m.dV)
				w = rand(m.dW)
				xβi = rand(xβ)
				W[i] = max(zero(T), xβi + v + min(m.ξ + w, e) - zd ) 
			end
		end
	end
	
	fetch.(tasks)

	return nothing 
end

function calculate_mean_position(d)
	# Get positions without outside option 
	positions = 0:maximum(vcat(d.positions...))
	return round(Int, mean(positions[positions .> 0])) 
end

# Reservation values / inverse calculations for costs 
"""
	calculate_discovery_value(G::Normal, m::SD)

Calculate the discovery value zd given cs, cd, and ξ from SD model `m`. Assumes that pre-search values xβ + v + w follow normal distribution `G`.

"""

function calculate_discovery_value(G::Normal, m)

	ξ, cs, cd = m.ξ, m.cs, m.cd  

	zd = 	if integrate_cdfsingle(cd, ξ, cs, mean(G), std(G))-cd ≈ -cd  # case where no convergence 
				-cd
			elseif cd <= 0 || (std(G) > 1e9 && cd <= 1e8) 
				Inf
			else
				fzero(t -> integrate_cdfsingle(t, ξ, cs, mean(G), std(G)) - cd, cd)
			end
	return zd 
end

function integrate_cdfsingle(z,ξ,cs,μ,σ)
	a = z-μ ; b = σ
	f(x) = pdf(Normal(),x)
	F(x) = cdf(Normal(),x)

	return (1 - F((z - ξ - μ) / σ)) * (μ - z - cs) + σ * f((z - ξ - μ) / σ) +
		(z - μ) * bvncdf(a / sqrt(1 + b^2), (μ + ξ - z) / σ, -b / sqrt(1 + b^2)) -
		σ * (-b / sqrt(1 + b^2) * f(a / sqrt(1 + b^2)) * (1 - F((z - ξ - μ) / σ * sqrt(1 + b^2) - a * b / sqrt(1 + b^2))))
		+ F(a - b * (z - ξ - μ) / σ) * f((z - ξ - μ) / σ) +
		1.0 / sqrt(1 + b^2) * f(a / sqrt(1 + b^2)) * (1 - F((z - ξ - μ) / σ * sqrt(1 + b^2) - a * b / sqrt(1 + b^2)))

end


# Consumer welfare
"""
	calculate_welfare(m::SDCore, data::DataSD; 
										method = "effective_values",
										kwargs...)
Calculate consumer welfare for the Search and Discovery model `m` using the data `data` and `n_sim` simulation draws. `method` can be either `"simulate_paths"` or `"effective_values"`. Simulate paths will simulate search paths  for each consumer and calculate welfare based on these paths. Effective values will calculate welfare based on effective values. 

"""
function calculate_welfare(m::SDCore, data::DataSD, n_sim; 
										method = "effective_values",
										kwargs...)

	if method == "simulate_paths"
		return calculate_welfare_simpaths(m, data, n_sim; kwargs...)
	elseif method == "effective_values"
		return calculate_welfare_effective_values(m, data, n_sim, seed; kwargs...)
	else
		throw(ArgumentError("Method $method not recognized."))
	end
end

function calculate_welfare_simpaths(m::SDCore, data::DataSD, n_sim; kwargs_data_generation...)

	# Set seed 
	set_seed(kwargs_data_generation)

	# Iterate over simulation draws and calculate welfare measures for each 

	# Average across all consumers 
	utility_choice_avg = zeros(Float64, n_sim)
	search_costs_avg = zeros(Float64, n_sim)
	discovery_costs_avg = zeros(Float64, n_sim)
	n_ses = length(data)

	# Average conditional on click 
	utility_choice_conditional_on_click = zeros(Float64, n_sim)
	search_costs_conditional_on_click = zeros(Float64, n_sim)
	discovery_costs_conditional_on_click = zeros(Float64, n_sim)

	# Average conditional on purchase
	utility_choice_conditional_on_purchase = zeros(Float64, n_sim)
	search_costs_conditional_on_purchase = zeros(Float64, n_sim)
	discovery_costs_conditional_on_purchase = zeros(Float64, n_sim)
	

	for sim in 1:n_sim
		new_seed = rand(1:1000000)
		d_sim, utility_purchases = generate_data(m, data; seed = new_seed, kwargs_data_generation...) 

		search_costs, discovery_costs = calculate_costs_in_sessions(m, d_sim) # note: 0 draws for cd because requires that already calculated 
		utility_choice_avg[sim] = sum(utility_purchases) / n_ses 
		search_costs_avg[sim] = sum(search_costs) / n_ses
		discovery_costs_avg[sim] = sum(discovery_costs) / n_ses

		# Conditional on click
		i_ses_with_clicks = sessions_with_clicks(d_sim) 
		n_sessions_with_clicks = length(i_ses_with_clicks)

		search_costs, discovery_costs = calculate_costs_in_sessions(m, d_sim[i_ses_with_clicks])
		utility_choice_conditional_on_click[sim] = sum(utility_purchases[i_ses_with_clicks]) / n_sessions_with_clicks
		search_costs_conditional_on_click[sim] = sum(search_costs) / n_sessions_with_clicks
		discovery_costs_conditional_on_click[sim] = sum(discovery_costs) / n_sessions_with_clicks
		
		# Conditional on purchase
		i_ses_with_purchase = sessions_with_purchase(d_sim)
		n_sessions_with_purchase = length(i_ses_with_purchase)
		search_costs, discovery_costs = calculate_costs_in_sessions(m, d_sim[i_ses_with_purchase])

		utility_choice_conditional_on_purchase[sim] = sum(utility_purchases[i_ses_with_purchase]) / n_sessions_with_purchase
		search_costs_conditional_on_purchase[sim] = sum(search_costs) / n_sessions_with_purchase
		discovery_costs_conditional_on_purchase[sim] = sum(discovery_costs) / n_sessions_with_purchase
	end

	welfare_avg = utility_choice_avg - search_costs_avg - discovery_costs_avg
	welfare_conditional_on_click = utility_choice_conditional_on_click - search_costs_conditional_on_click - discovery_costs_conditional_on_click
	welfare_conditional_on_purchase = utility_choice_conditional_on_purchase - search_costs_conditional_on_purchase - discovery_costs_conditional_on_purchase

	# Show warning if no session with at least one click 
	if isempty(sessions_with_clicks(data))
		@warn "No sessions with at least one click. Conditional welfare is NaN."
	end

	# Show warning if no session with at least one purchase
	if isempty(sessions_with_purchase(data))
		@warn "No sessions with at least one purchase. Conditional welfare is NaN."
	end

	# Return averages across simulations 
	# 1. Average welfare, utility, search costs, discovery costs
	# 2. Average conditional on click
	# 3. Average conditional on purchase

	return mean.((welfare_avg, utility_choice_avg, search_costs_avg, discovery_costs_avg)), 
			mean.((welfare_conditional_on_click, utility_choice_conditional_on_click, search_costs_conditional_on_click, discovery_costs_conditional_on_click)), 
			mean.((welfare_conditional_on_purchase, utility_choice_conditional_on_purchase, search_costs_conditional_on_purchase, discovery_costs_conditional_on_purchase))
end

function calculate_costs_in_sessions(m::SDCore, d)

	# Assert costs are part of model
	if isnothing(m.cs) || isnothing(m.cd)
		throw(ArgumentError("Search and discovery costs not calculated. Run calculate_costs! first."))
	end
	
	# paid search costs are the number of searches multiplied with search costs
	search_costs = [m.cs * sum(d.consideration_sets[i]) for i in eachindex(d)] 

	# number of discovery is equal to the position of the last product discovered because 
	# position = 0 is initial awareness set. So if stops on position =1 is one discovery, and so on. 
	discovery_costs = [m.cd * d.positions[i][d.stop_indices[i]] for i in eachindex(d)] 

	return search_costs, discovery_costs	

end