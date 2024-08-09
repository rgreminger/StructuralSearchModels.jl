using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools

m = SDCore( 
    β = [1.0, 5.0], 
    Ξ = 10.0, 
    ρ = [-1.0], 
    ξ = 5.5, 
    ξρ = [0.0], 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(), 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 5000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 2; seed = 123, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data
## 
    
calculate_costs!(m, data, 10090000; seed = 127) ; m.cd

## 

@time w = 
    calculate_welfare(m, data, 10; method = "simulate_paths", seed = 12873) 

## 
d1 = data[sessions_with_clicks(data)]
d2 = data[sessions_with_purchase(data)]