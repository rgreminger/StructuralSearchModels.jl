using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

m = SDCore( 
    β = [1.0, 6.5], 
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
n_consumers = 5000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data
d0 = deepcopy(d) 

## 
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
isequal(data.search_paths, d0.search_paths) 

##

evaluate_fit(m, d, 50) 

## 
m_hat = SD1( 
    β = m.β,
    Ξ = m.Ξ,
    ρ = m.ρ,
    ξ = m.ξ,
    dE = m.dE,
    dV = m.dV,
    dU0 = m.dU0,
    zdfun = m.zdfun
)
e = SmoothMLE()

@time calculate_likelihood(m_hat, e, d; debug_print=true)

# NOTES:
# ll should be very clost to zero (-1e-8 or so) because it should be log(1) = 0
# ll should be much faster!  