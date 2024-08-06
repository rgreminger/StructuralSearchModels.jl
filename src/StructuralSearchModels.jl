module StructuralSearchModels

# Types to be exported
export Estimator, Model, Data  # abstract types

# Functions to be exported
export generate_data, sample                    # functions for data generation
export estimate_model                           # functions for estimation
export evaluate_fit, calculate_standard_errors  # functions for post-estimation

# Load packages 
using Revise                    # for package development 

# Load general code 
include("abstractions.jl")      # abstract types and functions
include("utils.jl")             # helper functions

# Load concrete model code 




end # module StructuralSearchModels
