"""
    Abstract model type. All models are specified as subtypes of this abstract type.
"""
abstract type Model end

"""
    Abstract ranking model type. All ranking models are specified as subtypes of this abstract type.
"""
abstract type RankingModel <: Model end

"""
    Abstract search model type. All search models are specified as subtypes of this abstract type.
"""
abstract type SearchModel <: Model end

"""
    SearchRankingJointModel(search_model::SearchModel, ranking_model::RankingModel)
Search and ranking model that combines a search model and a ranking model.
"""
@with_kw mutable struct SearchRankingJointModel <: Model
    search_model::SearchModel
    ranking_model::RankingModel
end


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

Generate and return data generated for the model `model` for `n_consumers` and `n_sessions_per_consumer`. By default, this assumes that there is one alternative in the initial awareness set (`n_A0=1`), one alternative per position (`n_d=1`), and to generate generic products using `generate_products`. Undiscovered products by default are not dropped. `kwargs_path_generation` are passed to the function generating the search paths.
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
Generate and return data for the model `model` using the existing data object `data`. This allows simulating new search paths for the same consumers and products. If undiscovered products have been dropped, this function samples new products from `data` using `generate_products`. `kwargs_path_generation` are passed to the function generating the search paths.
"""
function generate_data(model::Model, data::Data; 
    drop_undiscovered_products = false,
    kwargs...) 
end

##################################################################################
# Estimation 

"""
    estimate(model::Model, estimator::Estimator, data::Data; kwargs...)
Estimate the `model` using `data` and `estimator`. Returns the estimated model, 
the parameter estimates, the objective function at the estimates, the solver results, and the standard errors as tuple. 
"""
function estimate(model::Model, estimator::Estimator, data::Data; kwargs...) end

##################################################################################
# Post estimation

""" 
    calculate_fit_measures(model::Model, data::Data, n_sim; kwargs...)
Compute fit measures of `model` for `data`. Allows for `kwargs` to be passed to the data generation. 
"""
function calculate_fit_measures(model::Model, data::Data, n_sim; kwargs...) end

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

##################################################################################
# Heterogeneity specification 

abstract type AbstractSpecification end 
abstract type AbstractHeterogeneitySpecification end