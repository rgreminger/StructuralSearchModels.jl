using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie, Optimization
seed = 122 
m = SD1( 
    β = [-0.1, 2.5], 
    Ξ = 2.5, 
    ρ = [-0.2], 
    ξ = 0.5,
    dE = Normal(0.0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1.0), 
    zdfun = "log"
)
n_consumers = 5000
conditional_on_search = true
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                # draws_e = fill(fill(2., 31), n_consumers),
                conditional_on_search = conditional_on_search, conditional_on_search_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,3)))

evaluate_fit(m, data, 1000; conditional_on_search = conditional_on_search, seed) ; 


## 
m_hat = deepcopy(m)

e = SmoothMLE(
    conditional_on_search = conditional_on_search, 
    options_numerical_integration = (n_draws = 50, n_draws_purchases = 50),
    options_solver = (show_trace = true, show_every = 1)
)

distribution_options = fill(false, 4)

startvals = vectorize_parameters(m_hat; distribution_options) ./ 2 


@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, e, data; 
                                                                        startvals, 
                                                                        print_solver_solution = true, 
                                                                        distribution_options) 



s = calculate_standard_errors(m_hat, e, data; distribution_options)
hcat(estimates, vectorize_parameters(m_hat; distribution_options), s)


## Welfare 
calculate_costs!(m, data, 10000)
@time w1 = calculate_welfare(m, data, 10000; method = "effective_values", seed = 13) 
@time w2 = calculate_welfare(m, data, 10000; method = "simulate_paths", seed = 14)

w1[:average][:welfare], w2[:average][:welfare]


## 

a  = zeros(2) 

a[1, 1, 1]