# Define abstract types of which the concrete types will be defined in the models
abstract type Model end 
abstract type Estimator end
abstract type Data end

# Define abstract functions to be implemented in the models


##################################################################################
# Data generation and sampling 

"""
    generate_data(model::Model, n_consumers, n_sessions_per_consumer; 
                        n_A0 = 1, n_d = 1, 
                        products = generate_products(n_consumers * n_sessions_per_consumer)  ) end 


Generate and return data generated for the model `model` for `n_consumers` and `n_sessions_per_consumer`. Default is to have one alternative in the initial awareness set, one alternative per position and to generate generic products. 

"""
function generate_data(model::Model, n_consumers, n_sessions_per_consumer; 
                        n_A0 = 1, n_d = 1, 
                        products = generate_products(n_consumers * n_sessions_per_consumer)  ) end 


"""
    generate_data(model::Model, data::Data; kwargs)

Generate and return data for the model `model` using the existing data object `data`. 
"""
function generate_data(model::Model, data::Data; kwargs...  ) end 

"""
    sample_data(data::Data, n)
Sample and return `n` consumers from the data object.

"""
function sample(data::Data, n::Int; kwargs...  ) end 

##################################################################################
# Estimation 

"""
    estimate_model(model::Model, data::Data, estimator::Estimator; kwargs...)

Estimate the `model` using `data` and `estimator`. Returns the estimated model.
"""
function estimate_model(model::Model, data::Data, estimator::Estimator; kwargs...) end


##################################################################################
# Post estimation

# Abstract function to evaluate fit 
""" 
    evaluate_fit(model::Model, data::Data; kwargs...)

Evaluate the fit of the model `model` using the data `data`.

"""
function evaluate_fit(model::Model, data::Data; kwargs...) end


# Abstract function to calculate standard errors 
""" 
    calculate_standard_errors(model::Model, data::Data, estimator::Estimator; bootstrap = false, kwargs...)

Calculate standard errors for the model `model` using the data `data` and estimator `estimator`. Set `bootstrap` to true to use bootstrap for standard error calculation. 
"""
function calculate_standard_errors(model::Model, data::Data, estimator::Estimator; bootstrap = false, kwargs...) end    
    