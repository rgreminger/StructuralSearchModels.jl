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

println("#########################################")
n_consumers = 2000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data

click_stats, purchase_stats, b_click, b_purch = calcualte_fit_measures(m, d, 1000; return_bounds = true) 


stats = (click_stats, purchase_stats)
bounds = (b_click, b_purch)

plot_across_positions(stats, bounds) 