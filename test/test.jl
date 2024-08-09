using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

m = SDCore( 
    β = [0.0, 5.0], 
    Ξ = 1.0, 
    ρ = [-0.01], 
    ξ = 1.0, 
    ξρ = [0.0], 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(0, 0) , 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 20000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 2; seed = 123, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data



## 



## 

@time w = 
    calculate_welfare(m, data, 10; method = "simulate_paths", seed = 12873) 

## 
d1 = data[sessions_with_clicks(data)]
d2 = data[sessions_with_purchase(data)]