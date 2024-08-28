abstract type MLE <: Estimator end 

@with_kw mutable struct SmoothMLE <: MLE
    options_optimization = (algorithm = LBFGS(), differentiation = Optimization.AutoForwardDiff())
    options_problem = () 
    options_solver = ()
    options_numerical_integration = (n_draws = 100, n_draws_purchases = 100 )
    conditional_on_search = false
end 

function estimate_model(model::Model, data::Data, estimator::MLE; 
                            startvals = vectorize_parameters(model),
                            print_solver_solution = false,
                            kwargs...) 
    # Estimate the model using maximum likelihood estimation 

    # Prepare additional arguments for objective function 
	args_likelihood_function = prepare_arguments_likelihood(model, estimator, data)

	############################################################################
	# Optimization.jl optimization (wrapper around many solvers)

	# Define objective function as negative likelihood function 
	objective_function(θ, p) = - loglikelihood(θ, model, estimator, data, args_likelihood_function...)

    # Extract options 
    options_optimization = estimator.options_optimization
    options_problem = estimator.options_problem
    options_solver = estimator.options_solver

    # Set up problem to solve for numerical optimizer
    obj = OptimizationFunction(objective_function, options_optimization.differentiation)
	prob = OptimizationProblem(obj, startvals; 
									options_problem...)

	# Run optimization 
	result_solver = 	solve(prob, options_optimization.algorithm; 
                            options_solver...);

    # Print complete solver solution if requested 
	if print_solver_solution
		println(res.original)
	end

	estimates = res.minimizer
	likelihood_at_estimates = res.minimum
				
    
    GC.gc()  
	return estimates, likelihood_at_estimates, result_solver
end

function calculate_likelihood(model::Model, estimator::MLE, data::Data; kwargs...)

    # Extract parameters 
    θ = vectorize_parameters(model; kwargs...)

    # Prepare additional arguments for objective function 
	args_likelihood_function = prepare_arguments_likelihood(model, estimator, data)

	# return likelihood
    return loglikelihood(θ, model, estimator, data, args_likelihood_function...; kwargs...)

end


function calculate_standard_errors(model::Model, estimator::MLE, data::Data; 
                                    bootstrap = false, kwargs...)

    # Extract parameters 
    θ = vectorize_parameters(model; kwargs...)

    # Prepare additional arguments for objective function 
	args_likelihood_function = prepare_arguments_likelihood(model, estimator, data)

	# Compute Hessian for negative likelihood function 
    f(θ) = - loglikelihood(θ, model, estimator, data, args_likelihood_function...)

    H = if options_optimization.differentiation == Optimization.AutoForwardDiff() # default, use autodiff 
		    ForwardDiff.hessian(f, θ)
        elseif options_optimization.differentiation == Optimization.AutoFiniteDiff() # use finite differences
            relstep = get(options_optimization, :relstep, FiniteDiff.default_relstep(Val(:hcentral),Float64))
            FiniteDiff.finite_difference_hessian(f, θ; relstep)
        else
            throw(ArgumentError("Only AutoForwardDiff and AutoFiniteDiff are supported for Hessian calculation.")) 
        end

    # Return standard errors
	return sqrt.(diag(inv(H)))
end

