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

click_stats, purchase_stats, fit_plot = evaluate_fit(m, d, 50) 
