# Generate products 
function generate_products(n_sessions; 
    n_products = 1_000_000,
    n_products_per_session = 30,
    distribution = Normal(), 
    outside_option = true,
    seed = 890367) 

    # Set seed 
    Random.seed!(seed)

    # Product ids, 0 will be outside option 
    pid = collect(1:n_products) 

    # Draw product characteristics for each consumer 
    product_characteristics = rand(distribution, n_products, 1) 

    # Draw products per session randomly from set of products for each session 
    product_ids =   if outside_option
						[ vcat(0 , rand(pid, n_products_per_session)) for i in 1:n_sessions]
					else
						[ rand(pid, n_products_per_session) for i in 1:n_sessions]
					end

    # Gather characteristics for each product
    product_characteristics =   if outside_option
									[ hcat(vcat(1.0, zeros(length(pids)-1)), vcat(zeros(1, size(product_characteristics, 2)), product_characteristics[pids[2:end], :])) for pids in product_ids]
								else
									[ product_characteristics[pids, :] for pids in product_ids]
								end

    return product_ids, product_characteristics
end 


# Get chunk size
function get_chunks(n)
    chunk_size = max(1, n ÷ (TASKS_PER_THREAD * Threads.nthreads()))
    data_chunks = Iterators.partition(1:n, chunk_size) 
    return chunk_size, data_chunks
end

# Functional form definitions 
function get_functional_form(s::String)
	if s == "" 
		(Ξ, ρ, pos) -> Ξ 
	elseif s == "linear"
		(Ξ, ρ, pos) -> Ξ + ρ[1] * pos 
	elseif s == "log"
		(Ξ, ρ, pos) -> Ξ + ρ[1] * _LP[pos]
	elseif s == "exp"
		(Ξ, ρ, pos) -> Ξ + ρ[1] * _EP[pos]
	else
		error("zdfun not correctly specified")
	end
end
