module StructuralSearchModels

# Types to be exported
export Estimator, Model, Data  # abstract types

# Functions to be exported
export generate_data, sample, generate_products                    # functions for data generation
export estimate_model                           # functions for estimation
export evaluate_fit, calculate_standard_errors  # functions for post-estimation

# Export concrete model types 
export SDCore

# Load packages 
using Revise                    # for package development 
using Random, Distributions, StatsBase             # math and other 
using Parameters                # utils 

# Import functions to add own definitions 
import Base: length, getindex 

# Load general code 
include("abstractions.jl")      # abstract types and functions
include("utils.jl")             # helper functions
include("constants.jl")

# Load concrete model code 
include("models/SearchDiscoveryCore.jl")     # Search and Discovery core model




end # module StructuralSearchModels
