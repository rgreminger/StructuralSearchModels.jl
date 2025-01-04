using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie

seed = 50234
m = WM1( 
    β = [-0.1, 4.5], 
    ξ = 1.5,
    ρ = [-0.2], 
    dE = Normal(0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1.), 
    zsfun = "log"
)
n_consumers = 2000
conditional_on_search = true 


@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                conditional_on_search, conditional_on_search_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,3)));
 

evaluate_fit(m, data, 500; conditional_on_search )


m_hat = deepcopy(m)

e = SmoothMLE(;
    conditional_on_search,  
    options_numerical_integration = (n_draws = 100, n_draws_purchases = 100),
    options_solver = (show_trace = true, show_every = 1) 
    # options_optimization = (algorithm = StructuralSearchModels.NelderMead(), differentiation = Optimization.AutoForwardDiff())
    )


distribution_options = [true, false, true, false]
startvals = vectorize_parameters(m_hat; distribution_options) .* 0.5
@time estimates, likelihood_at_estimates, result_solver = estimate_model(m_hat, e, data;
                                                                        distribution_options, 
                                                                        seed, 
                                                                        startvals) 
m_hat = construct_model_from_pars(estimates, m_hat; distribution_options)

s = calculate_standard_errors(m_hat, e, data; seed, distribution_options)

hcat(estimates, vectorize_parameters(m; distribution_options), s)


## 
function estimates_given_seed(seed, n_consumers) 
    data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                conditional_on_search = false, conditional_on_search_iter = 100,
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

############################################################################
## demand 

seed = 1236

n_consumers = 2000
conditional_on_search = false
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                conditional_on_search = conditional_on_search, conditional_on_search_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,.11)))


sum(data.search_paths[i][1] > 0 for i in 1:n_consumers) / n_consumers

## 
i = 1
j = 1
n_draws = 500
calculate_demand(m, data, i, j, n_draws; seed, conditional_on_search)
## 
j = 2
n_draws = 200
d0 = [calculate_demand(m, data, i, j, n_draws; conditional_on_search, seed = 123) for i in 1:1000]
    
println("Mean demand: ", mean(d0))
println("Demand data: ", count(data.purchase_indices .== j) / n_consumers)

## 
dem = [calculate_demand(m, data, i, j, n_draws; seed, conditional_on_search) for j in 1:31]
println(sum(dem))



