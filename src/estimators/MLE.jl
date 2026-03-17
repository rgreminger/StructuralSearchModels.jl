"""
    SMLE(n_draws::Int; kwargs...)
    SMLE(; numerical_integration_method, kwargs...)

Simulated maximum likelihood estimator. The number of simulation draws is specified either via positional argument `SMLE(n_draws)`, which creates a `DefaultNI(n_draws)` integration method, or explicitly via `SMLE(; numerical_integration_method = DefaultNI(100))`.

# Arguments
- `n_draws::Int`: (Positional constructor only) Number of simulation draws; creates `DefaultNI(n_draws)`.

# Keyword Arguments
- `numerical_integration_method::NIMethod`: Numerical integration method for the main integration over shocks. Required if not using the positional constructor.
- `options_optimization::NamedTuple`: Options passed to `OptimizationFunction` in Optimization.jl. Uses LBFGS by default.
- `options_problem::NamedTuple`: Options passed to `OptimizationProblem` in Optimization.jl.
- `options_solver::NamedTuple`: Options passed to `solve` in Optimization.jl. Defaults to 100,000 iterations.
- `numerical_integration_method_heterogeneity::NIMethod`: Integration method for unobserved heterogeneity. Defaults to `QMC(n_draws = 30)`.
- `conditional_on_search::Bool`: Whether the likelihood is conditional on at least one search. Defaults to `false`.
- `parameter_rescaling::Union{Nothing, AbstractVector}`: Optional scaling vector; the optimizer works in the rescaled space `φ = θ ./ scale`. See `build_inverse_hessian_scaler` for a helper to construct a suitable vector. Defaults to `nothing`.

# Examples
```julia
e = SMLE(100)
e = SMLE(200; conditional_on_search = true)
e = SMLE(; numerical_integration_method = QMC(n_draws = 500))
```
"""
@with_kw mutable struct SMLE <: MLE
    options_optimization = (
        algorithm = LBFGS(; linesearch = LineSearches.BackTracking(order = 2)),
        differentiation = Optimization.AutoForwardDiff())
    options_problem = ()
    options_solver = (iterations = 100_000,) # increases default no. iterations
    numerical_integration_method::NIMethod = _smle_no_ni_error()
    numerical_integration_method_heterogeneity::NIMethod = QMC(n_draws = 30)
    conditional_on_search = false
    parameter_rescaling::Union{Nothing, AbstractVector} = nothing
end
_smle_no_ni_error() = throw(ArgumentError("Must specify either number of draws through convenience constructor SMLE(n_draws) or through numerical_integration_method keyword argument."))
SMLE(n_draws::Int; kwargs...) = SMLE(; numerical_integration_method = DefaultNI(n_draws), kwargs...)

function estimate(model::Model, estimator::SMLE, data::Data;
        startvals = nothing,
        compute_std_errors = true,
        print_solver_solution = false,
        kwargs...)

    run_compatibility_checks(model, data; kwargs...)

    # Check starting values have right length
    if !isnothing(startvals) && length(startvals) != length(vectorize_parameters(model; kwargs...))
        throw(ArgumentError("Starting values must have the same length as the model parameters. Have $(length(startvals)) but expected $(length(vectorize_parameters(model; kwargs...)))."))
    end

    # Prepare additional arguments for objective function
    args_likelihood_function = prepare_arguments_likelihood(
        model, estimator, data; kwargs...)

    # Set starting values based on defaults or user input
    startvals = isnothing(startvals) ? vectorize_parameters(model; kwargs...) : startvals

    ############################################################################
    # Optimization.jl optimization (wrapper around many solvers)

    scale = estimator.parameter_rescaling

    # Define objective function as negative likelihood function.
    # If parameter_rescaling is set, the optimizer works in rescaled space φ = θ ./ scale,
    # so the Hessian is ~I and LBFGS converges without numerical scaling issues.
    function objective_function(θ, p)
        θ_rescaled = isnothing(scale) ? θ : scale .* θ
        -loglikelihood(θ_rescaled, model, estimator, data, args_likelihood_function...; kwargs...)
    end

    startvals_opt = isnothing(scale) ? startvals : startvals ./ scale

    # Extract options
    options_optimization = estimator.options_optimization
    options_problem = estimator.options_problem
    options_solver = estimator.options_solver

    # Set up problem to solve for numerical optimizer
    obj = OptimizationFunction(objective_function, options_optimization.differentiation)
    prob = OptimizationProblem(obj, startvals_opt;
        options_problem...)

    # Run optimization
    result_solver = solve(prob, options_optimization.algorithm;
        options_solver...)

    # Print complete solver solution if requested
    if print_solver_solution
        println(result_solver.original)
    end

    estimates = isnothing(scale) ? result_solver.u : scale .* result_solver.u
    likelihood_at_estimates = -result_solver.objective
    model_hat = construct_model_from_pars(estimates, model; kwargs...)

    # Standard errors
    std_errors = compute_std_errors ?
                 calculate_standard_errors(model_hat, estimator, data; kwargs...) :
                 nothing
    GC.gc()
    return model_hat, estimates, likelihood_at_estimates, result_solver, std_errors
end
"""
    calculate_likelihood(model::Model, estimator::MLE, data::Data; kwargs...)

Calculate the log-likelihood of `model` given `data` using `estimator`.

# Returns
A `Float64` log-likelihood value.
"""
function calculate_likelihood(model::Model, estimator::MLE, data::Data; kwargs...)

    # Extract parameters
    θ = vectorize_parameters(model; kwargs...)

    # Prepare additional arguments for objective function
    args_likelihood_function = prepare_arguments_likelihood(
        model, estimator, data; kwargs...)

    # return likelihood
    ll = loglikelihood(θ, model, estimator, data, args_likelihood_function...; kwargs...)

    show_timing = get(kwargs, :show_timing, false)
    if show_timing
        @info "Likelihood calculation takes $(@elapsed loglikelihood(θ, model, estimator, data, args_likelihood_function...; kwargs...)) seconds and requires $(@allocations loglikelihood(θ, model, estimator, data, args_likelihood_function...; kwargs...)) allocations."
    end
    return ll
end

function calculate_standard_errors(model::Model, estimator::MLE, data::Data; kwargs...)

    # Extract parameters
    θ = vectorize_parameters(model; kwargs...)

    # Prepare additional arguments for objective function
    args_likelihood_function = prepare_arguments_likelihood(
        model, estimator, data; kwargs...)

    # Compute Hessian for negative likelihood function
    f(θ) = -loglikelihood(θ, model, estimator, data, args_likelihood_function...; kwargs...)

    options_optimization = estimator.options_optimization

    H = if options_optimization.differentiation == Optimization.AutoForwardDiff() # default, use autodiff
        @suppress ForwardDiff.hessian(f, θ)
    elseif options_optimization.differentiation == Optimization.AutoFiniteDiff() # use finite differences
        relstep = get(options_optimization, :relstep,
            FiniteDiff.default_relstep(Val(:hcentral), Float64))
        FiniteDiff.finite_difference_hessian(f, θ; relstep)
    else
        throw(ArgumentError("Only AutoForwardDiff and AutoFiniteDiff are supported for Hessian calculation."))
    end

    # Return standard errors if possible to invert Hessian
    try
        return sqrt.(diag(inv(H)))
    catch
        @warn "No standard errors could be computed. Hessian is not invertible."
        return nothing
    end
end
