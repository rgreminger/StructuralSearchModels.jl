using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

m = SDCore( 
    β = [1.0, .5], 
    Ξ = 5.0, 
    ρ = [-0.7], 
    ξ = .5,
    ξρ = [0.0], 
    dE = Normal(0, 3.0), 
    dV = Normal(0, 3.0), 
    dU0 = Normal(0, 1), 
    dW = Normal(0, 0) , 
    zdfun = "log", 
    zsfun = "linear"
)
n_consumers = 20000
data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data

# click_stats, purchase_stats, fit_plot = evaluate_fit(m, d, 50) 
sum(d.min_discover_indices .> d.stop_indices)

i = 300 

## 
println(d.min_discover_indices[i])
println(d.stop_indices[i])

hcat(d.min_discover_indices, d.stop_indices)