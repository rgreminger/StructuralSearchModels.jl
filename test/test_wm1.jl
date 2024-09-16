using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

seed = 15
m = WM1( 
    β = [-0.1, 2.5], 
    ξ = 1.5,
    ρ = [-0.2], 
    dE = Normal(0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1.), 
    zsfun = "log"
)
n_consumers = 1000
conditional_on_search = true 
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                conditional_on_click = conditional_on_search, conditional_on_click_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,3)));
 

evaluate_fit(m, data, 100) 


m_hat = deepcopy(m)

e = SmoothMLE(;
    conditional_on_search,  
    options_numerical_integration = (n_draws = 100, n_draws_purchases = 100),
    options_solver = (show_trace = false, show_every = 1) 
    # options_optimization = (algorithm = StructuralSearchModels.NelderMead(), differentiation = Optimization.AutoForwardDiff())
    )


startvals = vectorize_parameters(m_hat) .* 0.5

@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, e, data;
                                                                        seed, 
                                                                        startvals) 

s = calculate_standard_errors(m_hat, e, data; seed)
hcat(estimates, vectorize_parameters(m_hat), s)


## 

calculate_likelihood(m_hat, e, data; debug_print = true )
## 
@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, d, e) 

hcat(estimates, vectorize_parameters(m_hat))

## 
function estimates_given_seed(seed, n_consumers) 
    data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                conditional_on_click = false, conditional_on_click_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,3)));

    estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, SmoothMLE(), data; startvals, distribution_options)  
    return estimates
end

n_consumers = 5000
n_sim = 10 
estimates_store = zeros(length(estimates), n_sim) 
for s in 1:n_sim 
    estimates = estimates_given_seed(s+ 20, n_consumers) 
    estimates_store[:, s] .= estimates  
end

hcat(mean(estimates_store, dims = 2), std(estimates_store, dims = 2), minimum(estimates_store, dims = 2), maximum(estimates_store, dims = 2))


## welfare
calculate_costs!(m_hat, d)
calculate_welfare(m_hat, d, 1000) 