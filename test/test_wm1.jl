using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

seed = 12125
m = SDCore( 
    β = [-0.05, 3.0], 
    Ξ = 100000., 
    ρ = [-1e-100], 
    ξ = 1.5,
    ξρ = [-0.2], 
    dE = Normal(0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1), 
    dW = Normal(0, 0) , 
    zdfun = "linear", 
    zsfun = "log"
)
n_consumers = 1000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                conditional_on_click = false, conditional_on_click_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,3)));
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
distribution_options[1] = distribution_options[3] = true

@time calculate_likelihood(m_hat, e, d; distribution_options)

e = SmoothMLE(
    options_numerical_integration = (n_draws = 100, n_draws_purchases = 100),
    options_solver = (show_trace = false, show_every = 50)
    )


startvals = vectorize_parameters(m_hat; distribution_options) .* 0.5

@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, e, d; 
                                                                        startvals,
                                                                        distribution_options) 

hcat(estimates, vectorize_parameters(m_hat; distribution_options))

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