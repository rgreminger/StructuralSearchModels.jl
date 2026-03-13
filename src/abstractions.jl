"""
    Abstract model type. All models are specified as subtypes of this abstract type.
"""
abstract type Model end

"""
    Abstract search model type. All search models are specified as subtypes of this abstract type.
"""
abstract type SearchModel <: Model end

"""
Abstract type for the *Search and Discovery* (SD) model. This type is a base type for all models that are subtypes of the Search and Discovery model.
"""
abstract type SDModel<: SearchModel end

"""
    Abstract estimator type. All estimators are specified as subtypes of this abstract type.
"""
abstract type Estimator end

abstract type MLE <: Estimator end

"""
    Abstract data type. All data objects are specified as subtypes of this abstract type.
"""
abstract type Data end

"""
    Abstract type for numerical integration methods used.
"""
abstract type NIMethod end



##################################################################################
# Data generation and sampling

"""
    generate_data(model::Model, n_consumers, n_sessions_per_consumer;
                        n_A0 = 1, n_d = 1,
                        indices_list_characteristics = 1:length(m.β),
                        products = generate_products(n_consumers * n_sessions_per_consumer, MvNormal(I(length(m.β)-1))),
                        drop_undiscovered_products = false,
                        kwargs_path_generation...)

Generate and return a `DataSD` object for the model `model` with `n_consumers` consumers and `n_sessions_per_consumer` sessions per consumer. By default, this assumes that there is one alternative in the initial awareness set (`n_A0=1`), one alternative per position (`n_d=1`), and generates generic products using `generate_products`. Undiscovered products are not dropped by default. `kwargs_path_generation` are passed to the function generating the search paths.

# Returns
A `DataSD` object with all fields populated, including `consumer_ids`, `product_ids`,
`product_characteristics`, `positions`, `consideration_sets`, `purchase_indices`,
`min_discover_indices`, `search_paths`, and `stop_indices`.

# Example
```julia
using Distributions, StructuralSearchModels
m = SD(β = [-0.05, 3.0], Ξ = 3.5, ρ = [-0.1], ξ = 2.5,
       dE = Normal(), dV = Normal(), dU0 = Uniform(), zdfun = "log")
d = generate_data(m, 100, 1; seed = 1)
# DataSD with 100 sessions
```
"""
function generate_data(model::Model, n_consumers, n_sessions_per_consumer;
        n_A0 = 1, n_d = 1,
        indices_list_characteristics = 1:length(model.β),
        products = generate_products(n_consumers * n_sessions_per_consumer, MvNormal(I(length(model.β)-1))),
        drop_undiscovered_products = false,
        kwargs...)
end

"""
    generate_data(model::Model, data::Data;
                    products = generate_products(data::Data),
                    kwargs_path_generation...)

Generate and return a new `DataSD` object for the model `model` using the existing data object `data`. This allows simulating new search paths for the same consumers and products. If undiscovered products have been dropped, new products are sampled from `data` using `generate_products`. `kwargs_path_generation` are passed to the function generating the search paths.

# Returns
A `DataSD` object with the same consumers and products as `data` but with freshly simulated search paths.

# Example
```julia
# Re-simulate search paths for the same consumers and products
d_sim = generate_data(m_hat, d; seed = 2)
```
"""
function generate_data(model::Model, data::Data;
    drop_undiscovered_products = false,
    kwargs...)
end

##################################################################################
# Estimation

"""
    estimate(model::Model, estimator::Estimator, data::Data; kwargs...)

Estimate the `model` using `data` and `estimator`.

# Returns
A 5-tuple `(model_hat, estimates, likelihood_at_estimates, result_solver, std_errors)` where:
- `model_hat`: estimated model of the same type as `model`, with parameters set to the estimates
- `estimates`: `Vector{Float64}` of parameter estimates in the order defined by `vectorize_parameters`
- `likelihood_at_estimates`: `Float64` log-likelihood value at the estimates
- `result_solver`: raw solver result object from Optimization.jl
- `std_errors`: `Vector{Float64}` standard errors (given `estimator`), or `nothing` if the computation fails (e.g., Hessian not invertible) or when `compute_std_errors = false`. 

# Example
```julia
using Distributions, StructuralSearchModels
m = SD(β = [-0.05, 3.0], Ξ = 3.5, ρ = [-0.1], ξ = 2.5,
       dE = Normal(), dV = Normal(), dU0 = Uniform(), zdfun = "log")
d = generate_data(m, 1000, 1; seed = 1)
e = SMLE(200)
m_hat, estimates, likelihood_at_estimates, result_solver, std_errors = estimate(m, e, d; seed = 1)

# Compare estimates with true values and standard errors
true_params = vectorize_parameters(m)
hcat(estimates, true_params, std_errors)
```
"""
function estimate(model::Model, estimator::Estimator, data::Data; kwargs...) end

##################################################################################
# Post estimation

"""
    calculate_fit_measures(model::Model, data::Data, n_sim; kwargs...)

Compute fit measures of `model` for `data` using `n_sim` simulation draws. `kwargs` are passed to the data generation function.

# Returns
A `Dict` with keys `:click_stats_data`, `:click_stats_sim`, `:purchase_stats_data`, `:purchase_stats_sim`, `:stop_probabilities_data`, `:stop_probabilities_sim`, and `:bounds`. The `_data` keys contain the empirical moments; the `_sim` keys contain the corresponding simulated moments.

# Example
```julia
fit_measures = calculate_fit_measures(m_hat, d, 1000; seed = 1)

# Compare position-specific click probabilities
prob_sim  = fit_measures[:click_stats_sim][:click_probability_per_position]
prob_data = fit_measures[:click_stats_data][:click_probability_per_position]
hcat(prob_sim, prob_data)
```
"""
function calculate_fit_measures(model::Model, data::Data, n_sim; kwargs...) end

"""
    calculate_standard_errors(model::Model, estimator::Estimator, data::Data; kwargs...)

Calculate asymptotic standard errors for the estimated `model` using `data` and `estimator`. Standard errors are computed for the respective `estimator`. Pass the same `seed` used during estimation to ensure consistent simulation draws.

# Returns
A `Vector{Float64}` of standard errors in the same order as `vectorize_parameters`. Returns `nothing` (with a warning) if the Hessian is not invertible.

# Example
```julia
# Compute standard errors after estimation (use the same seed)
std_errors = calculate_standard_errors(m_hat, e, d; seed = 1)
hcat(vectorize_parameters(m_hat), std_errors)
```
"""
function calculate_standard_errors(
    model::Model, estimator::Estimator, data::Data; kwargs...) end

##################################################################################
# Predictions

# Welfare
"""
    calculate_welfare(model::Model, data::Data, n_sim; kwargs...)

Calculate consumer welfare for `model` using `data` and `n_sim` simulation draws. Requires that `calculate_costs!` has been called on `model` beforehand.

# Returns
A `Dict` with three top-level keys: `:average`, `:conditional_on_search`, and `:conditional_on_purchase`. Each contains a nested `Dict` with keys `:welfare`, `:utility`, `:search_costs`, and `:discovery_costs`, which are all averages across sessions in the respective group.

# Example
```julia
# First compute costs
calculate_costs!(m_hat, d, 100_000; seed = 1)

W = calculate_welfare(m_hat, d, 1000; seed = 1)
W[:average][:welfare]              # average welfare across all sessions
W[:conditional_on_purchase][:welfare]  # welfare for sessions with a purchase
```
"""
function calculate_welfare(model::Model, data::Data, n_sim; kwargs...) end

# Revenues
"""
    calculate_revenues(model::Model, data::Data, kprice, n_draws; kwargs...)

Calculate revenues for `model` using `data` and `n_draws` simulation draws. `kprice` is the column index in the product characteristics matrix used as the price attribute.

# Returns
A `Dict` with keys:
- `:revenues`: `Float64` — total revenues summed across all sessions
- `:demand`: `Float64` — total demand summed across all sessions
- `:revenues_individual`: `Vector{Float64}` — revenues per session
- `:demand_individual`: `Vector{Float64}` — demand per session

# Example
```julia
R = calculate_revenues(m_hat, d, 1, 50; seed = 1)
R[:revenues]  # total revenues
R[:demand]    # total demand
```
"""
function calculate_revenues(model::Model, data::Data, kprice, n_draws; kwargs...) end

##################################################################################
# Parameter handling

"""
    vectorize_parameters(model::Model; kwargs...)
Vectorize the parameters of the model `model`.
"""
function vectorize_parameters(model::Model; kwargs...) end

"""
    construct_model_from_pars(θ::Vector{T}, model::Model; kwargs...)
Construct a model from the parameter vector `pars` for the model `model`.
"""
function construct_model_from_pars(θ::Vector{T}, model::Model; kwargs...) where T <: Real end

##################################################################################
# Heterogeneity specification

abstract type AbstractSpecification end
abstract type AbstractHeterogeneitySpecification end
