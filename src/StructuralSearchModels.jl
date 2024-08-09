module StructuralSearchModels

# Types to be exported
export Estimator, Model, Data  # abstract types

# Functions to be exported
export generate_data, generate_products, sessions_with_clicks, sessions_with_purchase        # functions for data generation
export estimate_model                                               # functions for estimation
export evaluate_fit, calculate_standard_errors, calculate_welfare   # functions for post-estimation
export calculate_costs!, calculate_discovery_value, calculate_search_cost, calculate_discovery_cost

# Export concrete model types 
export SD, SDCore 

# Load packages 
using Revise                                                        # for package development 
using Random, Distributions, StatsBase, QuadGK, Roots               # math and other 
using Parameters                                                    # utils 

# Import functions to add own definitions 
import Base: length, getindex, eachindex

# Load general code 
include("abstractions.jl")      # abstract types and functions
include("utils.jl")             # helper functions
include("constants.jl")

# Load concrete model code 
include("models/SearchDiscoveryCore.jl")     # Search and Discovery core model


end # module StructuralSearchModels
