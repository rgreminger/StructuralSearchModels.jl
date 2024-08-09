using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools

m = SDCore( 
    β = [1.0, 100.0], 
    cs = 0.1, 
    cd = 0.05, 
    Ξ = 10.0, 
    ρ = [-1.0], 
    ξ = 0.5, 
    ξρ = [0.0], 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(), 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 100000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 2, 1; 
                conditional_on_click = false, conditional_on_click_iter = 100); 



avg, cc, componentwise_pdf = 
    calculate_welfare(m, data, 2, 1; method = "simulate_paths") 

## 
d1 = data[sessions_with_clicks(data)]
d2 = data[sessions_with_purchase(data)]