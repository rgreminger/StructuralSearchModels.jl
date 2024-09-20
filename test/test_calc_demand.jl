using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie, Optimization
seed = 1236
m = SD1( 
    β = [-0.1, 2.5], 
    Ξ = 3.5, 
    ρ = [-0.2], 
    ξ = 1.5,
    dE = Normal(0.0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Uniform(0, 1), 
    zdfun = "log"
)
n_consumers = 1000 
conditional_on_search = false
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                # draws_e = fill(fill(2., 31), n_consumers),
                conditional_on_click = conditional_on_search, conditional_on_click_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,3)))

@time evaluate_fit(m, data, 1000; conditional_on_click = conditional_on_search, 
percentile = 0.95, 
) ; 


## 

j = 1

calculate_demand(m, data[1], j, 10; seed )
## 
@time d0 = [calculate_demand(m, data[i], j, 30) for i in 1:1000]

println("Mean demand: ", mean(d0))
println("Demand data: ", count(data.purchase_indices .== 1) / n_consumers)


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
## Standard errors 
m_hat = construct_model_from_pars(estimates, m_hat; distribution_options) 

@time calculate_likelihood(m_hat, e, data; debug_print = true ) 

## 
calculate_costs!(m_hat, d, 10000)
calculate_welfare(m_hat, d, 100) 

## 

a  = zeros(2) 

a[1, 1, 1]