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
    product_ids =   [ vcat(0 , rand(pid, n_products_per_session)) for i in 1:n_sessions]

    # Gather characteristics for each product
    product_characteristics =   [ product_characteristics[pids[2:end], :] for pids in product_ids]

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
		(Ξ,ρ,pos) -> Ξ 
	elseif s == "linear"
		(Ξ,ρ,pos) -> Ξ + ρ[1] * pos 
	elseif s == "fix" 
		(Ξ,ρ,pos) -> Ξ 
	elseif s == "log"
		(Ξ,ρ,pos) -> Ξ + ρ[1] * _lp[pos+1]
	elseif s == "logflex" 
		(Ξ,ρ,pos) -> Ξ + ρ[1] * log(-ρ[2]*pos+1)
	elseif s == "sqrt"
		(Ξ,ρ,pos) -> Ξ + ρ[1] * _sqp[pos+1]
	elseif s == "exp"
		(Ξ,ρ,pos) -> Ξ + ρ[1] * _ep[pos+1]
	elseif s[1:3] == "log" && length(s) > 3 && s[4] != 's'
		t = parse(Int,s[4:end])
		(Ξ,ρ,pos) -> Ξ + ρ[1] * _lp[min(t,pos)::Int+1]
	elseif s == "inverse"
		(Ξ,ρ,pos) -> Ξ - ρ[1] / (1 + pos)
	elseif s == "fquad"
		(Ξ,ρ,pos) -> Ξ - ρ[1] * (pos/ρ[2])^2
	elseif s == "fline"
		(Ξ,ρ,pos) -> Ξ - ρ[1] * pos + ρ[2] * pos^2
	elseif length(s) > 5 && s[1:5] == "cubic"
		t = parse(Float64,s[6:end])
		(Ξ,ρ,pos) -> Ξ + ρ[1] * (pos/5)^t
	elseif length(s) > 6 && s[1:7] == "logstep" && length(s) > 7
		t = parse(Int,s[8:end])
		(Ξ,ρ,pos) ->  Ξ + ρ[1] * _lp[floor(Int,pos/t)+1]
	elseif s == "steprates"
		(Ξ,ρ,pos) -> 	if pos == 0 
							Ξ - ρ[end]  
						elseif pos == 1 
							Ξ - ρ[end] + ρ[end-1]
						elseif pos == 2 
							Ξ - ρ[end] + ρ[end-1] + ρ[end-2]
						elseif pos <= 5 
							Ξ - ρ[end] + ρ[end-1] + ρ[end-2] + ρ[1] * _lp[pos+1]
						elseif 5 < pos 
							Ξ + ρ[1] * _lp[5+1] + ρ[2] * _lp[pos+1] 
						end					
	elseif length(s) == 4 && s[1:4] == "step"
		(Ξ,ρ,pos) -> 	if pos == 0 
							Ξ 
						elseif 0 < pos <= length(ρ) - 2 
							Ξ + sum(ρ[1:pos])
						else
							Ξ + sum(ρ) + ρ[end] * pos 
						end
	else
		error("zdfun not correctly specified")
	end
end
