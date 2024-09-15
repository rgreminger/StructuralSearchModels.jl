using Revise 
using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie, Optimization
seed = 122 
m = SDCore( 
    β = [-0.05, 3.0], 
    Ξ = 4.5, 
    ρ = [-0.5], 
    ξ = 1.0,
    ξρ = [0.0], 
    dE = Normal(0.0, 1.0), 
    dV = Normal(0, 1.0), 
    dU0 = Normal(0, 1), 
    dW = Normal(0, 0) , 
    zdfun = "log", 
    zsfun = "linear"
)
n_consumers = 1000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 1; seed, 
                # draws_e = fill(fill(2., 31), n_consumers),
                conditional_on_click = false, conditional_on_click_iter = 100,
                products = generate_products(n_consumers; distribution = Normal(0,3)))

evaluate_fit(m, data, 1000)
## 

m.ξ = 1.0

@time draws = generate_draws_with_search(m, d, 100, 100) ;

draws_u0, draws_e, draws_v, draws_w = draws ; 

draws_u0 = draws_u0[1]; draws_e = draws_e[1]; draws_v = draws_v[1]; draws_w = draws_w[1]


##
d, _ = generate_data(m, n_consumers, 1; draws_u0, draws_e, draws_v, draws_w, 
                products = generate_products(n_consumers; seed = 78723, distribution = Normal(0,3)))


swc = sessions_with_clicks(d) 
n_wc = length(swc)

## 
d0 = deepcopy(d)

## 

## 

d = data

evaluate_fit(m, d, 50) 

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

@time calculate_likelihood(m_hat, e, d; distribution_options)

e = SmoothMLE(
    options_numerical_integration = (n_draws = 25, n_draws_purchases = 25),
    options_solver = (show_trace = false, show_every = 1), 
    options_optimization = (algorithm = StructuralSearchModels.LBFGS(), differentiation = Optimization.AutoForwardDiff())
    )

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

## 

a  = zeros(2) 

a[1, 1, 1]