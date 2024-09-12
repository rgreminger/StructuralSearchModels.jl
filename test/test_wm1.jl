using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

seed = 1212533 
m = SDCore( 
    β = [2.0, 3.5], 
    Ξ = 100000., 
    ρ = [-1e-100], 
    ξ = 2.0,
    ξρ = [-0.2], 
    dE = Normal(0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1), 
    dW = Normal(0, 0) , 
    zdfun = "linear", 
    zsfun = "linear"
)
n_consumers = 1000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data
d0 = deepcopy(d) 

evaluate_fit(m, d, 50) 

m_hat = WM1( 
    β = m.β,
    ρ = m.ξρ,
    ξ = m.ξ,
    dE = m.dE,
    dV = m.dV,
    dU0 = m.dU0,
    zsfun = m.zsfun
)
e = SmoothMLE()

distribution_options = fill(false, 4) 
distribution_options[1] = true

@time calculate_likelihood(m_hat, e, d; debug_print=true, distribution_options)

e = SmoothMLE(
    options_numerical_integration = (n_draws = 100, n_draws_purchases = 100),
    options_solver = (show_trace = false, show_every = 50)
    )


startvals = vectorize_parameters(m_hat; distribution_options) .* 0.5

@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, e, d; 
                                                                        startvals,
                                                                        distribution_options,
                                                                        print_solver_solution = true) 

hcat(estimates, vectorize_parameters(m_hat; distribution_options))

## 
@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, d, e) 

hcat(estimates, vectorize_parameters(m_hat))