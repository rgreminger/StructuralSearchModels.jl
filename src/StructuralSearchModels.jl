module StructuralSearchModels

# Types to be exported
export Estimator, Model, Data  # abstract types

# Functions to be exported
export generate_data, generate_products, sessions_with_clicks, sessions_with_purchase        # functions for data generation
export estimate_model                                               # functions for estimation
export evaluate_fit, calculate_likelihood, calculate_standard_errors, calculate_welfare   # functions for post-estimation
export calculate_costs!, calculate_discovery_value, calculate_search_cost, calculate_discovery_cost, calculate_ξ
export plot_across_positions, vectorize_parameters, construct_model_from_pars, generate_draws_with_search

# Export concrete model types 
export SDCore, SD1, WM1

# Export concrete estimator types
export SmoothMLE

# Load packages 
using Revise                                                        # for package development 
using Random, Distributions, StatsBase, QuadGK, Roots, ForwardDiff, LinearAlgebra  # math and other 
using Optimization, OptimizationOptimJL                             # optimization
using Parameters                                                    # utils 
using Colors, CairoMakie                                            # plotting

using StatsFuns: sqrt2, invsqrt2
using SpecialFunctions: erf, erfinv

# Import functions to add own definitions 
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


end # module StructuralSearchModels
