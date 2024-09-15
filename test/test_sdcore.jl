using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie, Optimization
seed = 122 
m = SDCore( 
    β = [-0.05, 3.0], 
    Ξ = 4.5, 
    ρ = [-0.1], 
    ξ = 2.5,
    ξρ = [-0.2], 
    dE = Normal(1.0, 0.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1), 
    dW = Normal(0, 1) , 
    zdfun = "log", 
    zsfun = "linear"
)
n_consumers = 1000
@time d, _ = 
                generate_data(m, n_consumers, 1; seed, 
                # draws_e = fill(fill(2., 31), n_consumers),
                conditional_on_click = false, conditional_on_click_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,3)))
evaluate_fit(m, d, 1000) 

##



distribution_options = fill(false, 4) 
distribution_options[1] = false


e = SmoothMLE(
    options_numerical_integration = (n_draws = 25, n_draws_purchases = 25),
    options_solver = (show_trace = false, show_every = 1), 
    options_optimization = (algorithm = StructuralSearchModels.LBFGS(), differentiation = Optimization.AutoForwardDiff())
    )

m_hat = m 

@time calculate_likelihood(m_hat, e, d; distribution_options, debug_print = true)

## 
startvals = vectorize_parameters(m_hat; distribution_options) .* 0.8
@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, e, d; 
                                                                        startvals, 
                                                                        distribution_options) 

hcat(estimates, vectorize_parameters(m_hat; distribution_options))


## Standard errors 
m_hat = construct_model_from_pars(estimates, m_hat; distribution_options) 
se = calculate_standard_errors(m_hat, e, d)


## 
calculate_costs!(m_hat, d, 10000)
calculate_welfare(m_hat, d, 100) 