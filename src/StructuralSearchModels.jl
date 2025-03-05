module StructuralSearchModels

# Types to be exported
export Estimator, Model, Data  # abstract types

# Main functions exported 
export generate_data, generate_products # data generation
export estimate_model # functions for estimation
export evaluate_fit, calculate_likelihood, calculate_standard_errors # post-estimation 
export calculate_welfare, calculate_costs! # welfare and costs 
export calculate_revenues, calculate_demand # revenue and demand functions

# Additional helper functions 
export calculate_discovery_value, calculate_search_cost,
       calculate_discovery_cost, calculate_ξ
export vectorize_parameters, construct_model_from_pars, generate_draws_with_search
export add_indices_min_discover!, update_positions!

# Export concrete model types 
export SDCore, SD1, WM1, DataSD

# Export concrete estimator types
export SmoothMLE

# Dependencies 
using CairoMakie
using Colors
using Distributions
using ForwardDiff
using LinearAlgebra
using Optimization
using OptimizationOptimJL
using Parameters
using QuadGK
using Random
using Revise
using Roots
using SpecialFunctions: erf, erfinv
using StatsBase
using StatsFuns: sqrt2, invsqrt2
import Base: length, getindex, eachindex

# Load general code 
include("abstractions.jl")      # abstract types and functions
include("utils.jl")             # helper functions
include("constants.jl")

# Load concrete estimator code
include("estimators/MLE.jl")                 # Maximum likelihood estimation

# Load concrete model code 
include("models/SearchDiscoveryCore.jl")     # Search and Discovery core model
include("models/WM1.jl")
include("models/SD1.jl")

end
