# Tutorials and Examples

Below are several examples that showcase the functionality of the package and how it can be used.

## MC Simulation for the SD Model

A Monte Carlo simulation requires generating data for a model and estimating it. We now do this for the Search and Discovery Model.

First, we load the package. We also load the `Distributions` package because it specifies the `Distribution` type we're using here.
```@example 1
using StructuralSearchModels
using Distributions
```

Now, we create the model with defined parameter values.
```@example 1
# Define model
m = SD(
    β = [-0.05, 3.0], # preference parameters
    Ξ = 3.5,          # discovery value
    ρ = [-0.1],       # parameter governing decrease in discovery value across positions
    ξ = 2.5,          # search value
    dE = Normal(),    # Distribution of εᵢⱼ (shocks revealed from search)
    dV = Normal(),    # Distribution of νᵢⱼ (shocks revealed on list)
    dU0 = Uniform(),  # distribution of outside option shock η
    zdfun = "log"     # functional form specification
)
```

To generate sample data, we specify for how many consumers we want to generate data, and for how many sessions per consumer. We also specify a seed, which is not required but recommended. As a default, `generate_data` generates products with product characteristics drawn from a multivariate normal distribution and a last product characteristic which is a dummy for the outside option. In this case, this means that the last element of ``\beta`` is the preference parameter for the outside option. [Generating Products](@ref) provides an example of how products with more characteristics can be generated.

```@example 1
n_consumers = 1000
n_sessions_per_consumer = 1
seed = 1
d = generate_data(m, n_consumers, n_sessions_per_consumer; seed)
```

The data now is all in the data object `d`, which is a `DataSD` type. By default, the data contains all fields, including where consumers stopped scrolling and the search order. These fields can be dropped for the simulation by setting them to `nothing`. This is useful if you want to simulate data that does not include this information.

```@example 1
d.search_paths = nothing # hide
d.stop_indices = nothing
```

To estimate the model, we need to specify an estimator. The package currently includes one estimator: `SMLE`. It can be set up with the following command. See the [Estimation](estimation.md#options) section for more details on available options. By default, the `SMLE` estimator uses 100 simulation draws.

```@example 1
e = SMLE(200) # Simulated MLE estimator with 200 simulation draws
```

We now estimate the model using the `estimate` function. The function takes the model, the estimator, and the data as arguments. We also specify a seed for the RNG. The function returns the estimated model, the estimates, the likelihood at the estimates, the result of the optimization solver, and standard errors. Asymptotic standard errors in the `SMLE` are computed using the inverse Hessian.

```@example 1
seed = 1
m_hat, estimates, likelihood_at_estimates, result_solver,
    std_errors = estimate(m, e, d; seed)
m_hat
```

To compare the estimates with the true parameter values, we can get the parameter values of the model using the `vectorize_parameters` function.

```@example 1
true_params = vectorize_parameters(m)
est_params = vectorize_parameters(m_hat)
hcat(est_params, true_params, std_errors)
```

If we want to compute standard errors after estimation, the `calculate_standard_errors` function can be used. When doing so, it is important to set the `seed` argument to the same value as used for the estimation.

```@example 1
std_errors = calculate_standard_errors(m_hat, e, d; seed)
```

## MC Simulation for the WM Model

We can do the same for the WM model. The only difference is that we specify the model differently. The other functions are the same.

```@example 2
using StructuralSearchModels
using Distributions

# Define model
m = WM(
    β = [-0.1, 3.0], # preference parameters
    ρ = [-0.1],       # parameter governing decrease in discovery value across positions
    ξ = 1.5,          # search value
    dE = Normal(),    # Distribution of εᵢⱼ (shocks revealed from search)
    dV = Normal(),    # Distribution of νᵢⱼ (shocks revealed on list)
    dU0 = Uniform(),  # distribution of outside option shock η
    zsfun = "linear"     # functional form specification
)

n_consumers = 1000
n_sessions_per_consumer = 1
seed = 1
d = generate_data(m, n_consumers, n_sessions_per_consumer; seed)

e = SMLE(200) # Simulated MLE estimator with 200 simulation draws

seed = 1
m_hat, estimates, likelihood_at_estimates,
    result_solver, std_errors = estimate(m, e, d; seed) ;
m_hat
```

## Adding Search Over Vertical Attributes

To specify that vertical attributes are revealed when searching alternative, we additionally specify the `search_value_specification`. Below is an example where the first attribute in the `data.product_characteristics` is revealed on the list. This attribute enters utility through ``x_j^l\beta`` and also informs consumers beliefs about what will be revealed when searching through ``x_j^l\gamma``. The second characteristic is revealed on search and enters only ``x_j^d\kappa``. The following code sets up this model. Note that the other elements in the three parameter vectors are set to zero. Moreover, note that the last element in ``\beta`` is for the outside option.

```@example 1
m = SD(
    β = [-0.05, 0.0, 0.8],
    Ξ = 1.0,
    ρ = [-0.1],
    ξ = 1.0,
    dE = Normal(),
    dV = Normal(),
    dU0 = Normal(),
    zdfun = "linear",
    information_structure = InformationStructureSpecification(
        γ = [0.02, 0.0, 0.0],
        κ = [0.0, 0.1, 0.0],
        indices_characteristics_β_union = 1:1,
        indices_characteristics_γ_union= 1:0,
        indices_characteristics_κ_union = 2:2,
    )
)
```
## Computing Consumer Welfare and Revenues

To compute consumer welfare and revenues for an estimated model `m_hat` and data `d`, we can use the `calculate_welfare` and `calculate_revenues` functions.

First, we need to compute the costs using the `calculate_costs!` function. This function computes the search and discovery costs from the estimates for ``\xi`` and ``\Xi`` and adds them to the model in-place. Cost computation is done using `n_draws_cd` draws of the attribute distribution.

Once the costs are computed, we can compute welfare, which gives a `Dict` with welfare values. The dictionary reports the total welfare, the paid discovery costs, and the effective value of the chosen alternative. This is done as an average across all sessions, only those that had at least one search, and only those resulted in a purchase.

Going back to the example for the SD model, we can compute the costs and welfare as follows:

```@example 1
# Compute costs
n_draws_cd = 100_000
calculate_costs!(m_hat, d, n_draws_cd; seed)

# Compute welfare
n_draws_welfare = 1000
calculate_welfare(m_hat, d, n_draws_welfare; seed)
```

To compute revenues, we can use the `calculate_revenues` function. It requires specifying which product characteristic is the price attribute and the number of draws to use for the demand computation of each product. Note, this computation is done using the simulation procedure described in [Greminger (2025)](https://rgreminger.github.io/wp/heterogeneous_position_effects.html).

```@example 1
index_price_attribute = 1
n_draws_demand = 50
total_revenues, revenues_by_session = calculate_revenues(m_hat, d, index_price_attribute, n_draws_demand; seed)
```

## Computing Fit Measures

To evaluate how well an estimated model fits the data, we can use the `calculate_fit_measures` function. This function computes the fit measures for the estimated model and the data. It requires specifying the number of draws to use for the simulation. It returns a dictionary with the fit measures.

For the previously estimated SD model, we can compute the fit measures as follows:

```@example 1
n_draws = 1000
fit_measures = calculate_fit_measures(m_hat, d, n_draws; seed)
```

These fit measures then can be accessed in the usual way. For example, if we want to compare the position-specific click probabilities, we can do this as follows.

```@example 1
prob_sim = fit_measures[:click_stats_sim][:click_probability_per_position]
prob_data = fit_measures[:click_stats_data][:click_probability_per_position]
hcat(prob_sim, prob_data) # compare simulated and data click probabilities
```

Or if we want to compare the average number of clicks per session, we can do this as follows.

```@example 1
n_clicks_sim = fit_measures[:click_stats_sim][:no_clicks_per_session]
n_clicks_data = fit_measures[:click_stats_data][:no_clicks_per_session]
hcat(n_clicks_sim, n_clicks_data) # compare simulated and data clicks per session
```
## Estimating Shock Variances

To estimate the variances of the unobserved shocks, we can pass the `estimation_shock_variances` keyword argument to the `estimate` function. The following would estimate the variance of the unobserved shock `dE`. Note that trying to run this code will not produce valid results. This is because the variance is not identified in this example, which is not checked by the code.

See the [Estimation](estimation.md#options) section for more details on the options.

```julia
estimation_shock_variances = [:σ_dE]
m_hat, estimates, likelihood_at_estimates,
    result_solver, std_errors = estimate(m, e, d; seed, estimation_shock_variances)
```

## Specifying Solver Options

Solver options can be passed to the estimator. By default, the `SMLE` estimator uses the `LBFGS` optimizer, with gradients and Hessians computed using automatic differentiation. Specifically, calling `e=SMLE(100)` is equivalent to calling the following:
```julia
using Optimization # need to load to specify AutoDiff
using OptimizationOptimJL # provides Optim algorithms
using LineSearches
e = SMLE(100;
    options_optimization = (
        algorithm = LBFGS(; linesearch = LineSearches.BackTracking(order = 2)),
        differentiation = Optimization.AutoForwardDiff()
    )
)
```
It uses the `LBFGS` algorithm with the `BackTracking` line search algorithm, which I found to be a bit more stable for these models.

While this is the default, we can use any solver/algorithm accessible through the Optimization package, provided we load the necessary packages first. For example, we can use the `NelderMead` algorithm as follows:
```julia
using Optimization # need to load to specify AutoDiff
using OptimizationOptimJL # provides Optim algorithms (LBFGS, NelderMead etc.)
e = SMLE(100;
    options_optimization = (
        algorithm = NelderMead(),
        differentiation = Optimization.AutoForwardDiff()
    )
)

```
Note that the `NelderMead` algorithm does not use gradients, but the differentiation still needs to be specified as it has no default.

We can also pass additional options to the optimizer that are used by the `Optimization.jl` package as described in the [estimation](estimation.md) section. For example, we can print information on the solvers' iterations as follows:
```julia
e = SMLE(100;
        options_solver = (show_trace = true, show_every = 10)
)
```
This again uses the default `LBFGS` optimizer. These options may differ across optimizers. To see what's all possible, check the [estimation](estimation.md) section and the documentation of the [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) package.

## Parameter Rescaling

When model parameters differ greatly in how they affect the likelihood, the optimizer can take longer to converge. The `parameter_rescaling` option lets you provide a vector of scaling factors so that the optimizer works in a normalized space. The package offers a convenient way to construct a suitable scaling vector via `build_inverse_hessian_scaler`. This function approximates the curvature of the likelihood at a given point (which usually would be the starting values)and returns `1 ./ sqrt.(diag_H)`, which is a scaling vector ready to pass to the `parameter_rescaling` option. 

```julia
# Compute scaling vector from diagonal Hessian at starting values
x0 = vectorize_parameters(m)
e = SMLE(100) 
scale = build_inverse_hessian_scaler(m, e, d, x0)

# Update estimator with scaling vector
e.parameter_rescaling = scale

# Estimate with the updated estimator
m_hat, estimates, likelihood_at_estimates, result_solver, std_errors = estimate(m, e, d)
```

The scaling is applied transparently: the optimizer works with rescaled parameters `θ_rescaled = θ ./ scale`, while all outputs (estimates, standard errors) are returned on the original scale.

## Conditioning on Search

Sometimes, it is necessary to condition on search, for example, when estimating the model on data that only includes sessions with at least one click/search. This can be accommodated by setting the `conditional_on_search` option to `true`. The keyword applies to both the `generate_data` function and the `SMLE` estimator. Importantly, the keyword also needs to be set when calculating fit measures based on the estimated model. If the model is estimated conditional on search, while the fit measures are computed without it, the estimated model will likely not fit the data well.

Below is an example for an MC simulation for the WM model that conditions on search.

```@example
using StructuralSearchModels
using Distributions

# Define model
m = WM(
    β = [-0.1, 3.0], # preference parameters
    ρ = [-0.1],       # parameter governing decrease in discovery value across positions
    ξ = 1.5,          # search value
    dE = Normal(),    # Distribution of εᵢⱼ (shocks revealed from search)
    dV = Normal(),    # Distribution of νᵢⱼ (shocks revealed on list)
    dU0 = Uniform(),  # distribution of outside option shock η
    zsfun = "linear"     # functional form specification
)

n_consumers = 1000
n_sessions_per_consumer = 1
seed = 1
d = generate_data(m, n_consumers, n_sessions_per_consumer; seed, conditional_on_search = true)

e = SMLE(200; conditional_on_search = true) # Simulated MLE estimator with 200 simulation draws

m_hat, estimates, likelihood_at_estimates, result_solver, std_errors = estimate(m, e, d; seed) ;

n_sim = 1000
fit_measures = calculate_fit_measures(m_hat, d, n_sim; seed, conditional_on_search = true)
m_hat
```

## Generating Products

The `generate_data` function generates products using the `generate_products` function with default options. By default, it generates many different products, so that the products for each session are different. For each session, 30 products are sampled from the total number of products generated. For each product, product characteristics are drawn from a multivariate normal distribution, and the last product characteristic is assumed to be a dummy for the outside option.

The following generates products with a single product characteristic using the default options.
```@example 3
using Distributions, StructuralSearchModels
n_sessions = 2

product_ids, product_characteristics = generate_products(n_sessions, Normal())
product_characteristics[1] # show the product characteristics for the first session
```

We can update the defaults by specifying keyword arguments. The following generates 100 products, of which, 10 are sampled for each session. There are now five product characteristics, and the last one is a dummy for the outside option.

```@example 3
using LinearAlgebra # provides convenient identify matrix constructor I(n)
n_sessions = 2
product_ids, product_characteristics = generate_products(n_sessions, MvNormal(I(5));
    n_products = 100, n_products_per_session = 10)
product_characteristics[1] # show the product characteristics for the first session
```

We can also suppress the outside option indicator by adding `outside_option = false`.

```@example 3
product_ids, product_characteristics = generate_products(n_sessions, MvNormal(I(5));
    n_products = 100, n_products_per_session = 10, outside_option = false)
product_characteristics[1] # show the product characteristics for the first session
```
## Preparing Data

To estimate the model on real data, the data needs to be prepared. The following example shows how to do this when the data is in a `DataFrame`. The code loops over the sessions (which is fast in Julia) and creates the necessary structure for the `DataSD` type.

```@example 4
using DataFrames, StructuralSearchModels
# Generate example data
df = DataFrame(
    session_id = [1, 1, 1, 2, 2, 2],
    position = [0, 1, 2, 0, 1, 2],
    product_id = [1, 2, 3, 1, 2, 3],
    price = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    quality = [0.5, 0.6, 0.7, 0.5, 0.6, 0.7],
    discovered = [1, 1, 0, 1, 1, 1],
    clicked = [1, 0, 0, 0, 1, 0],
    purchased = [1, 0, 0, 0, 0, 0],
)


# Extract consumer ids. Have only one session per id.
n_sessions = length(unique(df.session_id))
consumer_ids = collect(1:n_sessions)

# Create empty arrays to store the data
product_ids = []
product_characteristics = []
positions = []
indices_list_characteristics = []
session_characteristics = []
consideration_sets = []
purchase_indices = []
stop_indices = []

# loop over sessions and add to arrays. For each session, adds outside option
for subdf in groupby(df, :session_id)

    # Product ids
    product_ids_i = vcat(0, subdf.product_id)
    push!(product_ids, product_ids_i)

    # Product characteristics
    pc_i = Matrix(subdf[:, [:price, :quality]]) # converts to matrix
    pc_i = hcat(pc_i, zeros(size(pc_i, 1))) # add characteristic for outside option
    pc_i = vcat(zeros(1, size(pc_i, 2)), pc_i) # add outside option as product
    pc_i[1, end] = 1 # set outside option dummy to 1 for outside option
    push!(product_characteristics, pc_i)

    # Positions
    positions_i = vcat(0, subdf.position)
    push!(positions, positions_i)

    # Consideration sets
    cs_i = vcat(0, subdf.clicked)
    push!(consideration_sets, cs_i)

    # stop indices
    si = findlast(x -> x == 1, subdf.discovered) # finds last 1 in discovered
    push!(stop_indices, isnothing(si) ? length(subdf.discovered) + 1 : si + 1) # + 1

    # purchase indices
    pi = findfirst(x -> x == 1, subdf.purchased) # returns nothing if not found
    push!(purchase_indices, isnothing(pi) ? 1 : pi + 1) # +1 for outside option
end

# Put together into data type
d = DataSD(; consumer_ids,
    product_ids,
    product_characteristics,
    positions,
    consideration_sets,
    purchase_indices,
    stop_indices)
```


## Changing the Number of Products in Each Position
The number of products in each position can be set by adjusting the positions in the `DataSD` type. The package provides a function to do this. The following updates the positions in the data object `d` to have 3 products in the first position (those revealed without cost), and then 2 products per position after that.
```julia
update_positions!(d, 3, 2)
```

To generate data with a different number of positions, we can also pass the keyword arguments `n_A0` and `n_d` to the `generate_data` function. The following generates data with 3 products in the first position (those in the initial awareness set), and then 2 products per position after that.
```julia
generate_data(model, n_consumers, n_sessions_per_consumer;
                        n_A0 = 3, n_d = 2)
```
