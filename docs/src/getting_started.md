# Getting Started 

## Requirements 

Julia 1.11 or newer is required to run this package. The package is designed to be compatible with the currently latest stable version of Julia. It may also work with older versions, but this is not guaranteed. To install julia, follow the [Installation Instructions](https://docs.julialang.org/en/v1/manual/installation/). 

If you are new to Julia but have experience with other programming languages (e.g., Python), I recommend reading the [Julia Introduction](https://docs.julialang.org/en/v1/). 

## Installation 

This package is registered in the Julia General registry. You can install it using the Julia package manager. Open a Julia REPL and run the following command:

```julia
using Pkg
Pkg.add("StructuralSearchModels")
```
Alternatively, you can install the package from the Julia REPL (using `]`). 

Note that this package continues to be developed and I aim to add new models, specifications, and estimation approaches over time. 


## Usage 

The following is a simple example of specifying a model, simulating data from it, and estimating the model on the simulated data. See [Tutorials](tutorials.md) for details and further examples and use cases.

```julia
using Distributions, StructuralSearchModels 
# Define model 
m = SD(
    β = [-0.05, 3.0], # preference parameters
    Ξ = 3.5,          # discovery value 
    ρ = [-0.1],       # parameter governing decrease in discovery value across positions 
    ξ = 2.5,          # search value 
    dE = Normal(),    # Distribution of εᵢⱼ (shocks revealed from search)
    dV = Normal(),    # Distribution of νᵢⱼ (shocks revealed on list)
    dU0 = Uniform(),  # distribution of outside option shock η
    zdfun = "log"     # functional form specification 
)

# Generate data 
n_consumers = 1000
n_sessions_per_consumer = 1 
seed = 1 
d = generate_data(m, n_consumers, n_sessions_per_consumer; seed)

# Estimate model
e = SMLE(200) # Simulated MLE estimator
m_hat, estimates, likelihood_at_estimates, result_solver, std_errors = estimate(m, e, d; seed);
```

The types and functions this package provides have docstrings, which provide details on the usage. These can be accessed in the usual way in Julia (e.g., `?SD` gives the docstring for the `SD` type).

## Multi-threading 
The package is fairly optimized and uses multi-threading for both estimation and data generation. For example, the above example shouldn't take more than a few seconds to run after once the code is compiled. To use multi-threading, Julia should be started with multiple threads. The [Julia manual](https://docs.julialang.org/en/v1/manual/multi-threading/) explains how you can do this.
