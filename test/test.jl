using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

m = SDCore( 
    β = [0.0, 3.], 
    Ξ = 5., 
    ρ = [-0.5], 
    ξ = 5.0,
    ξρ = [0.0], 
    dE = Normal(0, 0.0), 
    dV = Normal(0, 5.0), 
    dU0 = Normal(0, 0), 
    dW = Normal(0, 0) , 
    zdfun = "linear", 
    zsfun = "linear"
)

n_consumers = 5000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 423, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data
calculate_costs!(m, data, 100000) 


we = calculate_welfare(m, data, 100; method = "effective_values") ; 

wc = calculate_welfare(m, data, 100; method = "simulate_paths") ;

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

calculate_welfare(m, data, 1; method = "effective_values") ;

calculate_welfare(m, data, 1; method = "simulate_paths") ;

