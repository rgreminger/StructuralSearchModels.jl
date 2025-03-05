"""
    Abstract model type. All models are specified as subtypes of this abstract type.
"""
abstract type Model end

"""
    Abstract estimator type. All estimators are specified as subtypes of this abstract type.
"""
abstract type Estimator end

"""
    Abstract data type. All data objects are specified as subtypes of this abstract type.
"""
abstract type Data end

##################################################################################
# Data generation and sampling 

"""
    generate_data(model::Model, n_consumers, n_sessions_per_consumer; 
                        n_A0 = 1, n_d = 1, 
                        products = generate_products(n_consumers * n_sessions_per_consumer) ) 

Generate and return data generated for the model `model` for `n_consumers` and `n_sessions_per_consumer`. Default is to have one alternative in the initial awareness set, one alternative per position and to generate generic products. 
"""
function generate_data(model::Model, n_consumers, n_sessions_per_consumer;
        n_A0 = 1, n_d = 1,
        products = generate_products(n_consumers * n_sessions_per_consumer)) end

"""
    generate_data(model::Model, data::Data; kwargs...)
Generate and return data for the model `model` using the existing data object `data`. This allows simulating new search paths for the same consumers and products. 
"""
function generate_data(model::Model, data::Data; kwargs...) end

##################################################################################
# Estimation 

"""
    estimate_model(model::Model, estimator::Estimator, data::Data; kwargs...)
Estimate the `model` using `data` and `estimator`. Returns the estimated model, 
the parameter estimates, the objective function at the estimates, the solver results, and the standard errors as tuple. 
"""
function estimate_model(model::Model, estimator::Estimator, data::Data; kwargs...) end

##################################################################################
# Post estimation

""" 
    evaluate_fit(model::Model, data::Data, n_sim; kwargs...)
Evaluate the fit of the model `model` for `data`. Allows for kwargs to be passed to the evaluation function (see documentation). 
"""
function evaluate_fit(model::Model, data::Data, n_sim; kwargs...) end

""" 
    calculate_standard_errors(model::Model, estimator::Estimator, data::Data; kwargs...)
Calculate standard errors for the model `model` using the specified `data` and `estimator`. 
"""
function calculate_standard_errors(
        model::Model, estimator::Estimator, data::Data; kwargs...) end

##################################################################################
# Predictions 

# Welfare 
""" 
    calculate_welfare(model::Model, data::Data, n_sim; kwargs...)
Calculate consumer welfare for the model `model` using the data `data` and `n_sim` simulations. 
"""
function calculate_welfare(model::Model, data::Data, n_sim; kwargs...) end

# Revenues 
""" 
    calculate_revenues(model::Model, data::Data,, kprice, n_draws; kwargs...)
Calculate revenues for the model `model` using the data `data` and `n_draws` simulation draws. `kprice` indicates which column in the characteristics table is used as price. 
"""
function calculate_revenues(model::Model, data::Data, kprice, n_draws; kwargs...) end

# Demand 
""" 
    calculate_demand(model::Model, data::Data, i, j, n_draws; kwargs...)

Calculate demand for the model `model` using the data `data` and `n_draws` simulation draws. `i` is the consumer index and `j` is the product index for which demand is calculated. 
""" 
function calculate_demand(model::Model, data::Data, i, j, n_draws; kwargs...) end


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


