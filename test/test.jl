using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

m = SDCore( 
    β = [0.0, .5], 
    Ξ = 5.0, 
    ρ = [-0.9], 
    ξ = -1.5,
    ξρ = [0.0], 
    dE = Normal(0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 0), 
    dW = Normal(0, 0) , 
    zdfun = "linear", 
    zsfun = "linear"
)

println("#########################################")
n_consumers = 5000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data

calculate_costs!(m, data, 100000) 

we = calculate_welfare(m, data, 100; method = "effective_values", seed = 1) ; 

wc = calculate_welfare(m, data, 100; method = "simulate_paths", seed = 1) ;

println("###################")
println("Avg welfare effective values = $(we[1][1])")   
println("Avg welfare simulate paths = $(wc[1][1])")
println("Welfare conditional on click = $(we[2][1])")
println("Welfare conditional on click = $(wc[2][1])")
println("Welfare conditional on purchase = $(we[3][1])")
println("Welfare conditional on purchase = $(wc[3][1])")

println("###################")
println("Discovery costs paid avg effective values = $(we[1][3])")
println("Discovery costs paid avg simulate paths = $(wc[1][4])")

## 

we = calculate_welfare(m, data, 500; method = "effective_values") ;
wc = calculate_welfare(m, data, 500; method = "simulate_paths") ;


println("###################")
println("Avg welfare effective values = $(we[1][1])")   
println("Avg welfare simulate paths = $(wc[1][1])")
println("Welfare conditional on click = $(we[2][1])")
println("Welfare conditional on click = $(wc[2][1])")
println("Welfare conditional on purchase = $(we[3][1])")
println("Welfare conditional on purchase = $(wc[3][1])")

println("###################")
println("Discovery costs paid avg effective values = $(we[1][3])")
println("Discovery costs paid avg simulate paths = $(wc[1][4])")
