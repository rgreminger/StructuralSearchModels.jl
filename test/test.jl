using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

m = SDCore( 
    β = [1.0, 2.0], 
    Ξ = 2.5, 
    ρ = [-0.3], 
    ξ = 1.0,
    ξρ = [0.0], 
    dE = Normal(0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1), 
    dW = Normal(0, 0) , 
    zdfun = "log", 
    zsfun = "linear"
)
n_consumers = 500
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data
d0 = deepcopy(d) 

evaluate_fit(m, d, 50) 

## 
fixed_parameters = [true, true, false, false]
distribution_options = fill(false,4)
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

@time calculate_likelihood(m_hat, e, d; debug_print=true, fixed_parameters, distribution_options)

## 
e = SmoothMLE(
    options_numerical_integration = (n_draws = 25, n_draws_purchases = 25),
    options_solver = (show_trace = false, show_every = 1)
    )
# e = SmoothMLE()
fixed_parameters = [false, false, false, false]
distribution_options = fill(false, 4)
@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, d, e; 
                    startvals = vectorize_parameters(m_hat; fixed_parameters, distribution_options), 
                    distribution_options, fixed_parameters) 

println("ll at estimates: ", likelihood_at_estimates)
hcat(estimates, vectorize_parameters(m_hat; fixed_parameters, distribution_options))
