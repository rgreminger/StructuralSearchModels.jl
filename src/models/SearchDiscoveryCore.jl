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
	n_d::Int64
	n_A0::Int64
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
	consumer_indices::Vector{UnitRange{Int}}	# which sessions belong to which consumer 
	product_ids::Vector{Vector{Int}}			# product ids for each session in order 
	product_characteristics::Vector{Matrix{T}}	# product characteristics matrix 
	positions::Vector{Vector{Int}}				# positions for each session in order
	search_paths::Vector{Vector{Int}} 			# search paths for each session 
	consideration_sets::Vector{Vector{Bool}}	# consideration sets for each session, booleans whether searched or not 
	purchase_indices::Vector{Int} 				# which product within session is purchased 
	stop_indices::Vector{Int}					# which product within session is stopped a

	@assert length(product_ids) == length(product_characteristics) == length(positions) == length(search_paths) == length(consideration_sets) == length(purchase_indices) == length(stop_indices)
end


# Data generation 
function generate_data(m::SDCore, n_consumers, n_sessions_per_consumer; 
						products = generate_products(n_consumers*n_sessions_per_consumer), kwargs...) 

	n_sessions = n_consumers * n_sessions_per_consumer 

	# Unpack products
	product_ids, product_characteristics = products 

	# Create positions 
	positions = [vcat(ones(Int64, 1 + m.n_A0), repeat(collect(Int64, 2:(length(product_ids[i]) - 1 - m.n_A0) / m.n_d), inner=m.n_d)) for i in 1:n_sessions]

	return product_ids, product_characteristics, positions
end