using BenchmarkTools
using Distributions
using LinearAlgebra
using Random
using Revise
using StableRNGs
using StructuralSearchModels
using Test

m = WM(
    β = [-0.1, 0.1, 0.2, -0.1, 0.2, -0.2, 0.2, 0.3],
    # β = [-0.1, 0.1, 0.3],
    ρ = [-0.1],
    ξ = 1.0,
    dE = Normal(),
    dV = Normal(),
    dU0 = Normal(),
    zsfun = "log",
)

# Use stable RNG, guaranteeing draws stay the same across Julia versions
seed = 1


# Generate data
n_attributes = length(m.β)-1
n_consumers = 1000
d = generate_data(m, n_consumers, 1; seed,
    products = generate_products(n_consumers, MvNormal(zeros(n_attributes), I(n_attributes)); seed))

d.stop_indices = nothing

# add_product_fe!(m, d, 2, "both")

@time calculate_costs!(m, d, 1000; seed)


##
e = SMLE(;conditional_on_search = true)

calculate_likelihood(m, e, d; seed, show_timing =true )


##

e = SMLE(;
    numerical_integration_method = DefaultNI(n_draws=1000),
    options_solver = (show_trace = true, show_every = 5),
    conditional_on_search = true
)

@time m_hat, estimates, fval, solver_res, std_err = estimate(m, e, d; seed)
hcat(vectorize_parameters(m), vectorize_parameters(m_hat), estimates, std_err)

##

@time calculate_costs!(m_hat, d, 1000; seed)

hcat(m.cs[1], m_hat.cs[1])
