# StructuralSearchModels.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://rgreminger.github.io/StructuralSearchModels.jl/dev)

This is a Julia package that implements structural search models and related functions for estimation and consumer welfare analysis. In particular, it provides the following functionality for different structural search models:

- Estimation
- Data simulation
- Model fit evaluation
- Welfare analysis

## How the package works

The package defines types for _models_, _estimators_, and _data_. For example, the abstract type `Model` is defined for structural search models, and specific models (e.g., the search and discovery model) are implemented as subtypes of this abstract type.

The package then implements functions for these types while making use of Julia's multiple dispatch: functions execute different code depending on the input types. This setup leads to a convenient interface for users by exposing only high-level functions and types. For example, the `estimate(model, estimator, data)` function will dispatch to different function definitions depending on the types of its arguments. It also lends itself to future extension: new models and estimation approaches can be implemented as new types, and the same high-level functions can be used for estimation, data generation, etc. without changing the user interface.

## Installation

This package is registered in the Julia General registry. You can install it using the Julia package manager. Open a Julia REPL and run the following command:

```julia
using Pkg
Pkg.add("StructuralSearchModels")
```

## Minimal example

Below is a minimal example showing how to define a model, generate data from it, and estimate it on the simulated data. Further examples and tutorials are provided in the [documentation](https://rgreminger.github.io/StructuralSearchModels.jl/dev).

```julia
using Distributions, StructuralSearchModels
# Define model
m = SD(
    β = [-0.05, 3.0], # preference parameters
    Ξ = 4.0,          # discovery value
    ρ = [-0.1],       # parameter governing decrease in discovery value across positions
    ξ = 2.5,          # search value
    dE = Normal(),    # Distribution of εᵢⱼ (shocks revealed from search)
    dV = Normal(),    # Distribution of νᵢⱼ (shocks revealed on list)
    dU0 = Uniform(),  # distribution of outside option
    zdfun = "log"     # functional form specification
)

# Generate data
n_consumers = 100
seed = 1
d = generate_data(m, n_consumers, 1; seed)

# Estimate model
e = SMLE(100) # Simulated MLE estimator with 100 simulation draws
model_hat, estimates, likelihood_at_estimates, result_solver, std_errors = estimate(m, e, d; seed);
```

## Documentation

Detailed information on using the package can be found in the [documentation](https://rgreminger.github.io/StructuralSearchModels.jl/stable). It is organized as follows:

- [Getting Started](https://rgreminger.github.io/StructuralSearchModels.jl/dev/getting_started): Installation and usage instructions.
- [Tutorials and Examples](https://rgreminger.github.io/StructuralSearchModels.jl/dev/tutorials): Examples of how to use the package.
- [Models](https://rgreminger.github.io/StructuralSearchModels.jl/dev/models): Describes the available model types and specifications
- [Data](https://rgreminger.github.io/StructuralSearchModels.jl/dev/data_generation): Describes the data types and data generation functions.
- [Estimation](https://rgreminger.github.io/StructuralSearchModels.jl/dev/models): Describes the estimation functions.
- [Post Estimation](https://rgreminger.github.io/StructuralSearchModels.jl/dev/post_estimation): Describes the post estimation functions (e.g., welfare analysis, demand analysis, etc.)

## State and future development

This is the first public release of the package. The core functionality for estimating the search and discovery model and Weitzman model is implemented and tested, including accompanying data generation and post-estimation functions. I will also try to keep the public-facing types and functions stable, so that future updates do not break existing code.

For the next version, I plan to allow specifying and estimating parameters that govern observed and unobserved heterogeneity across consumers (e.g., heterogeneous preferences). There are already several placeholders as type definitions that hopefully are sufficient so that the public-facing types do not change when implementing this in the next version.

If you are interested in contributing to the development of the package, please get in touch! For example, there are many possible extensions to the package, such as implementing alternative estimation approaches (e.g., Bayesian estimation), or adding more structural search models.

## Citation

If you use this package in your research, please cite the following paper that introduced the estimation approach:

> Greminger, R. P. (2025). Trade-Offs Between Ranking Objectives: Descriptive Evidence and Structural Estimation. arXiv preprint available at https://arxiv.org/abs/2210.16408.

If you use the search and discovery model implemented in this package, please also cite the following paper that introduced the model:

> Greminger, R. P. (2022). Optimal Search and Discovery. *Management Science* 68(5), 3904–3924.

<details>
  <summary>BibTeX entries</summary>

```bibtex
@misc{greminger2025tradeoffs,
      title={Trade-Offs Between Ranking Objectives: Descriptive Evidence and Structural Estimation},
      author={Rafael P. Greminger},
      year={2025},
      eprint={2210.16408},
      archivePrefix={arXiv},
      primaryClass={econ.GN},
      url={https://arxiv.org/abs/2210.16408},
}

@article{Greminger2022,
  title = {Optimal Search and Discovery},
  author = {Greminger, Rafael P.},
  year = {2022},
  journal = {Management Science},
  volume = {68},
  number = {5},
  pages = {3904--3924}
}
```

</details>

## Acknowledgements

This package builds on the amazing Julia ecosystem. It would not have been possible without the great work done as part of the packages listed in the `Project.toml` file on GitHub.

The package heavily relies on the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) and [Optimization.jl](https://github.com/SciML/Optimization.jl) packages. Optimization.jl implements solvers that are published in different packages. By default, this package uses the solver implemented by [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

You can use the entries below to cite these packages.

<details>
  <summary>BibTeX entries</summary>

```bibtex
    @article{JSSv098i16,
       author = {Mathieu Besançon and Theodore Papamarkou and David Anthoff and Alex Arslan and Simon Byrne and Dahua Lin and John Pearson},
       title = {Distributions.jl: Definition and Modeling of Probability Distributions in the JuliaStats Ecosystem},
       journal = {Journal of Statistical Software},
       volume = {98},
       number = {16},
       year = {2021},
       keywords = {Julia; distributions; modeling; interface; mixture; KDE; sampling; probabilistic programming; inference},
       issn = {1548-7660},
       pages = {1--30},
       doi = {10.18637/jss.v098.i16},
       url = {https://www.jstatsoft.org/v098/i16}
    }

    @software{vaibhav_kumar_dixit_2023_7738525,
        author = {Vaibhav Kumar Dixit and Christopher Rackauckas},
        month = mar,
        publisher = {Zenodo},
        title = {Optimization.jl: A Unified Optimization Package},
        version = {v3.12.1},
        doi = {10.5281/zenodo.7738525},
        url = {https://doi.org/10.5281/zenodo.7738525},
        year = 2023}

    @article{mogensen2018optim,
      author  = {Mogensen, Patrick Kofod and Riseth, Asbj{\o}rn Nilsen},
      title   = {Optim: A mathematical optimization package for {Julia}},
      journal = {Journal of Open Source Software},
      year    = {2018},
      volume  = {3},
      number  = {24},
      pages   = {615},
      doi     = {10.21105/joss.00615}}
```

</details>
