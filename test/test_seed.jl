using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie, Optimization
seed = 122 

function f() 
    
    seed = 9873
    m = SD1( 
        β = [0.0, 3.0], 
        Ξ = 4.5, 
        ρ = [-0.5], 
        ξ = 1.0,
        dE = Normal(0.0, 1.0), 
        dV = Normal(0, 1.0), 
        dU0 = Normal(0, 1), 
        zdfun = "log"        )
    n_consumers = 100
    data, utility_purchases = 
                    generate_data(m, n_consumers, 1; seed = seed,   
                    conditional_on_click = false) 

    ll = calculate_likelihood(m, SmoothMLE(), data; seed = seed)

    return data, ll 

end

data0, ll0 = f()
data1, ll1 = f()

isequal(data0, data1), isequal(ll0, ll1) 