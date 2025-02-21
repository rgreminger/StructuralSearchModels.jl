abstract type MLE <: Estimator end

@with_kw mutable struct SmoothMLE <: MLE
    options_optimization = (
        algorithm = LBFGS(; linesearch = Optim.BackTracking(order = 2)),
        differentiation = Optimization.AutoForwardDiff())
    options_problem = ()
    options_solver = ()
    options_numerical_integration = (n_draws = 100, n_draws_purchases = 100)
    conditional_on_search = false
end

function estimate_model(model::Model, estimator::SmoothMLE, data::Data;
        startvals = nothing,
        compute_std_errors = true,
        print_solver_solution = false,
        kwargs...)
    # Estimate the model using maximum likelihood estimation 

    # Prepare additional arguments for objective function 
    args_likelihood_function = prepare_arguments_likelihood(
        model, estimator, data; kwargs...)

    # Set starting values based on defaults or user input
    startvals = isnothing(startvals) ? vectorize_parameters(model; kwargs...) : startvals

    ############################################################################
    # Optimization.jl optimization (wrapper around many solvers)

    # Define objective function as negative likelihood function 
    function objective_function(θ, p)
        -loglikelihood(θ, model, estimator, data, args_likelihood_function...; kwargs...)
    end

    # Extract options 
    options_optimization = estimator.options_optimization
    options_problem = estimator.options_problem
    options_solver = estimator.options_solver

    # Set up problem to solve for numerical optimizer
    obj = OptimizationFunction(objective_function, options_optimization.differentiation)
    prob = OptimizationProblem(obj, startvals;
        options_problem...)

    # Run optimization 
    result_solver = solve(prob, options_optimization.algorithm;
        options_solver...)

    # Print complete solver solution if requested 
    if print_solver_solution
        println(result_solver.original)
    end

    estimates = result_solver.minimizer
    likelihood_at_estimates = -result_solver.minimum
    model_hat = construct_model_from_pars(estimates, model; kwargs...)

    # Standard errors
    seed = args_likelihood_function[end]
    std_errors = compute_std_errors ?
                 calculate_standard_errors(model_hat, estimator, data; kwargs..., seed) :
                 nothing
    GC.gc()
    return model_hat, estimates, likelihood_at_estimates, result_solver, std_errors
end

function calculate_likelihood(model::Model, estimator::MLE, data::Data; kwargs...)

    # Extract parameters 
    θ = vectorize_parameters(model; kwargs...)

    # Prepare additional arguments for objective function 
    args_likelihood_function = prepare_arguments_likelihood(
        model, estimator, data; kwargs...)

    # return likelihood
    return loglikelihood(θ, model, estimator, data, args_likelihood_function...; kwargs...)
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
        ForwardDiff.hessian(f, θ)
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
