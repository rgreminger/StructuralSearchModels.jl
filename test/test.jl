using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

m = SDCore( 
    β = [0.0, -1.0], 
    Ξ = -100.0, 
    ρ = [-100.0], 
    ξ = .0,
    ξρ = [0.0], 
    dE = Normal(0, 1), 
    dV = Normal(0, 0.0), 
    dU0 = Normal(0,1), 
    dW = Normal(0, 0) , 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 5000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1293, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data


d.search_paths[4] 

calculate_costs!(m, d, 100000) 

seed = 169


@time we = 
    calculate_welfare(m, data, 100; method = "effective_values", seed) ; 


@time wc = 
	calculate_welfare(m, data, 100; method = "simulate_paths", seed) ;
	

(we[3][1], wc[3][1]) 



## 
calculate_welfare(m, data[1], 1; method = "effective_values", seed) ; 


## 
calculate_costs!(m, d, 100000) 

# Get discovery value 
chars = vcat([d.product_characteristics[i][d.product_ids[i] .> 0, :] for i in eachindex(d)]...) # excludes outside option 
xβ = chars * m.β
μ = xβ
σ = std(xβ)

G = Normal(mean(xβ), sqrt(m.dV.σ^2 + var(xβ)) ) # products by default are drawn from normal distribution
Ξ = calculate_discovery_value(G, m) 

println("cd = $(m.cd)") 
println("Ξ = $(Ξ)")
println("m.Ξ = $(m.Ξ)")
