using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie, Optimization

m = SDCore( 
    β = [2.0, 4.5], 
    Ξ = 5.5, 
    ρ = [-0.5], 
    ξ = 3.0,
    ξρ = [0.0], 
    dE = Normal(0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1), 
    dW = Normal(0, 0) , 
    zdfun = "log", 
    zsfun = "linear"
)
n_consumers = 200
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed = 1, 
                conditional_on_click = false, conditional_on_click_iter = 100); 
d = data
d0 = deepcopy(d) 

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

distribution_options = fill(false, 4) 
distribution_options[1] = false

@time calculate_likelihood(m_hat, e, d; debug_print=true, distribution_options)


e = SmoothMLE(
    options_numerical_integration = (n_draws = 100, n_draws_purchases = 100),
    options_solver = (show_trace = true, show_every = 50), 
    options_optimization = (algorithm = StructuralSearchModels.NelderMead(), differentiation = Optimization.AutoForwardDiff())
    )

startvals = vectorize_parameters(m_hat; distribution_options) .* 0.8
@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, d, e; 
                                                                        startvals, 
                                                                        distribution_options) 

hcat(estimates, vectorize_parameters(m_hat; distribution_options))