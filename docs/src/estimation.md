## Simulated Maximum Likelihood

For model estimation, this package implements estimators as subtypes of an abstract `Estimator` type. Currently, only simulated maximum likelihood is available. The baseline estimator can be created with `e = SMLE(n_draws)` where `n_draws` specifies the number of simulation draws used for numerical integration.


```@docs
SMLE
```
## Options

Various options for the optimization and numerical integration procedures used in estimation can be specified. Moreover, the `conditional_on_search` option allows specifying whether the sample is selected based on there being at least one click.

#### Optimization
Simulated maximum likelhood finds the parameter estimates by numerically maximizing the model likelihood function. This is implemented using the amazing [Optimization.jl](https://github.com/SciML/Optimization.jl) package, which allows to specify a range of optimizers (from other packages) and solvers. The `SMLE` struct is set up to to pass the respective options to the optimization package as keyword arguments.

Specifically, this is how the different fields are passed into the respective constructors and functions of the [Optimization.jl](https://github.com/SciML/Optimization.jl) package:
```julia
f = OptimizationFunction(loglikelihood, options_optimization.differentiation)
p = OptimizationProblem(f, startvals; options_problem...)
r = solve(p, options_optimization.algorithm; options_solver...)
```

#### Numerical Integration
Simulated maximum likelihood uses numerical integration to compute a likelihood function that has no closed-form. `SMLE` uses the `DefaultNI` by default, which is also the only option so far. In the future, I hope that alternative approaches to compute the likelihood function are implemented.

The `DefaultNI` for now uses the method proposed by [Greminger (2025)](https://rgreminger.github.io/wp/heterogeneous_position_effects.html) to integrate over the unobserved shocks for all available models.

```@docs
DefaultNI
```

#### Estimation of Shock Variances

Options whether to estimate the variances of the unobserved shocks are directly passed as keyword arguments to the `estimate` function, rather than being part of the `Estimator` type. This is because it is not specific to any estimator and rather a general option.

By default, the variances are not estimated. To estimate them, pass the keyword argument `estimation_shock_variances` to the `estimate` function. The following are valid options:

- `estimation_shock_variances = [:σ_dE]`: estimates the variance of the unobserved shock `dE`
- `estimation_shock_variances = [:σ_dV]`: estimates the variance of the unobserved shock `dV`.
- `estimation_shock_variances = [:dUequaldE, :σ_dE]`: estimates the variance of the unobserved shock `dE` and assumes that the distribution of the outside option shock `dU0` is equal to that of `dE`.
- `estimation_shock_variances = [:dUequaldV, :σ_dV]`: estimates the variance of the unobserved shock `dV` and assumes that the distribution of the outside option shock `dU0` is equal to that of `dV`.

Note that the code currently does not check for the validity of the options and will try to estimate them as specified, even if it is not possible. For example, setting `estimation_shock_variances = [:σ_dE, :σ_dV]` will run the estimation and try to estimate both variances, even though they are not separately identified.


#### Fixing Parameters

By passing a `fixed_parameters` argument to the `estimate` function, parameters can be fixed so that they are not estimated. This is useful to evaluate identification of different parameters. For example, estimating the model as follows will fix the parameter `β` to the values provided in the model `m`:

```julia
estimate(m, SMLE(100), d; fixed_parameters = [:β])
```

## Estimation Functions


For model estimation, the following abstract functions are available. These functions should be implemented for all available (and sensible) combinations of `Model`, `Estimator`, and `Data` types.

```@docs
calculate_likelihood
```
```@docs
estimate
```
```@docs
calculate_standard_errors
```
