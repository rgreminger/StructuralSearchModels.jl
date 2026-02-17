module StructuralSearchModels

# Types to be exported
export Estimator, Model, Data, NUModel  # abstract types

# Main functions exported 
export generate_data, generate_products # data generation
export estimate # functions for estimation
export calculate_fit_measures, calculate_likelihood, calculate_standard_errors # post-estimation 
export calculate_welfare, calculate_costs! # welfare and costs 
export calculate_revenues # revenues 
export rank_alternatives! # ranking alternatives

# Additional helper functions 
export calculate_discovery_value
export vectorize_parameters, construct_model_from_pars
export fill_indices_min_discover!, update_positions!, expand_products!, merge_data
export add_product_fe!, find_products_appearing_min_n_times, add_product_fe_data!
export update_heterogeneity_specification!

# Export concrete model types
export SDCore, SD, WM, DataSD, SimpleWeightsRanking, SearchRankingJointModel
export InformationStructureSpecification
export HeterogeneitySpecification
export DefaultNI, QMC
export SDNU, WMNU, InformationStructureSpecificationNU, HeterogeneitySpecificationNU

# Export concrete estimator types
export SMLE

# Dependencies 
using Distributions
using FastGaussQuadrature
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using Optimization
using OptimizationOptimJL
using Parameters
using PreallocationTools
using QuadGK
using QuasiMonteCarlo
using Random
using Revise
using Roots
using SpecialFunctions: erf, erfinv
using StatsBase
using StatsFuns: sqrt2, invsqrt2
using Suppressor

# Add functions already exported by packages 
import Base: length, getindex, eachindex, == 

import Distributions: estimate 

# Load general code 
include("abstractions.jl")      # abstract types and functions
include("numerical_integration_options.jl")
include("constants.jl") 

# Load concrete estimator code
include("estimators/MLE.jl")                 # Maximum likelihood estimation

# Load abstract heterogenetiy code
include("models/heterogeneity.jl")     # Abstract heterogeneity model

# Load concrete model code 
include("models/SDCore.jl")     # Search and Discovery core model
include("models/WM.jl")
include("models/SD.jl")
include("models/SimpleWeightsRanking.jl")    # Simple weights ranking model


include("models/SDRModel.jl")
include("models/non_unicode.jl") # models that do not use Greek unicode letters for fields 

include("utils.jl")             # helper functions

end
