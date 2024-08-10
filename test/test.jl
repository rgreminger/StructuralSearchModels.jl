using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

m = SDCore( 
    β = [0.0, 2.0], 
    Ξ = 3., 
    ρ = [-0.2], 
    ξ = 3.0,
    ξρ = [0.0], 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(0, 0) , 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 5000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1293, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data

calculate_costs!(m, d, 100000) 


## 

seed = 11

calculate_welfare(m, data, 10; method = "effective_values", seed) ; 
calculate_welfare(m, data, 10; method = "simulate_paths", seed) ;

@time we = 
    calculate_welfare(m, data, 50; method = "effective_values", seed) ; 


@time wc = 
	calculate_welfare(m, data, 50; method = "simulate_paths", seed) ;
	


println("###################")
println("Avg welfare effective values = $(we[1][1])")   
println("Avg welfare simulate paths = $(wc[1][1])")
println("Welfare conditional on click = $(we[2][1])")
println("Welfare conditional on click = $(wc[2][1])")
println("Welfare conditional on purchase = $(we[3][1])")
println("Welfare conditional on purchase = $(wc[3][1])")



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
