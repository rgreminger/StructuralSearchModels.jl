# Models

At the core of the `StructuralSearchModels.jl` are different search models, implemented as subtypes of the abstract `SearchModel <: Model` type. Currently, the following two main search models are implemented:

- Search and Discovery Model
- Weitzman Model

Because the Weitzman model is a special case of the Search and Discovery model, specifications for both are implemented as subtypes of the general `SDModel` type.

## Search and Discovery Model

The empirical Search and Discovery model has been introduced by [Greminger (2025)](https://rgreminger.github.io/wp/heterogeneous_position_effects.html), building on the theory introduced by [Greminger (2022)](https://rgreminger.github.io/pub/search_discovery.html).

The model requires specifying `ξ` and `Ξ`, rather than the costs `cs` and `cd`. These costs can be computed and added to the model using `calculate_costs!(model, data)`.

By default, the model is specified so that all observable product characteristics are revealed on the list, and only a shock ``\varepsilon_{ij}`` is revealed on search. Moreover, note that the last element in ``\beta`` is for the outside option dummy. The specification for which product characteristics are revealed where can be set through the `information_structure`, as described below.

The functional form for how the discovery value ``z^d(h)`` depends on position ``h`` is specified through the string `zdfun`. This is done so that the model struct can be readily saved to disk (e.g., using the `JLD2` package) without having to save a function definition. Currently, the following functions are available:
- `"linear"`: ``z^d(h) = \Xi + \rho h ``
- `"log"`: ``z^d(h) = \Xi + \rho \log(1+h) ``
- `"exp"`:  ``z^d(h) = \Xi + \rho \exp(1+h) ``

Note that the `heterogeneity` specification for now is a placeholder that will be used in future versions to implement observed and unobserved heterogeneity across parameters.


```@docs
SD{T}
```

Which product characteristics are revealed where can be specified through specifying the `information_structure`. This specification has parameters ``\gamma`` and ``\kappa`` and allows specifying indices for which elements in `product_characteristics[i]` enter ``x_j\beta`` (on list), ``x_j\gamma`` (beliefs about search), and ``x_j\kappa`` (revealed upon search).

Where characteristics are revealed is specified through `indices_characteristics_β_individual` (and the others), which implies that all sessions reveal the same characteristics in the same place. Note that ``\beta`` needs to have the same length as all `product_characteristics[i]`, with zeros for product characteristics that are only revealed upon search. The last entry in `product_characteristics[i]` is for the outside option.

 If sessions differ in where they reveal characteristics, then this can be specified by specifying `indices_characteristics_β_individual` as a vector that needs to have the same length as the number of sessions in the data. The same holds for the indices for the other parameters, and `indices_characteristics_β_union` as the union of all characteristics revealed on the list for at least one session. The same holds for `indices_characteristics_γ_individual` and `indices_characteristics_κ_individual`.

```@docs
InformationStructureSpecification{T}
```

#### Estimation
The model can be estimated using simulated maximum likelihood. By default, the approach of [Greminger (2025)](https://rgreminger.github.io/wp/heterogeneous_position_effects.html) is used. This approach constructs a smooth likelihood function, is relatively fast to estimate, and does not use the search order. It requires the two shocks to be normally distributed (`dE <: Normal` and `dV <: Normal`).

The package also provides the following function to combine the model parameters into a vector.
```@docs
vectorize_parameters
```

#### Reservation value and cost calculations

The following function is available to compute costs and reservation values.
```@docs
calculate_costs!
```


## Weitzman Model

The empirical Weitzman model is a special case of the Search and Discovery model. It also requires specifying `ξ` rather than the search cost `cs`, where `calculate_costs!(model, data)` can be used to fill in the search costs.

Position specific search costs follow from the functional form on how ``\xi(h)`` changes with position `h`. This functional form is specified through the string `zsfun`. The following options are available:
- `"linear"`: ``\xi(h) = \xi + \rho h ``
- `"log"`: ``\xi(h) = \xi + \rho \log(1+h) ``
- `"exp"`:  ``\xi(h) = \xi + \rho \exp(1+h) ``

```@docs
WM{T}
```

As with the `SD` model, the `information_structure` determines which characteristics are revealed on the list, and which ones on the detail page.

#### Estimation
The model can be estimated using the simulated maximum likelihood. By default, the approach of [Greminger (2025)](https://rgreminger.github.io/wp/heterogeneous_position_effects.html) is used. This approach constructs a smooth likelihood function, is fast to estimate, and does not use the search order. It requires the two shocks to be normally distributed (`dE <: Normal` and `dV <: Normal`).

#### Reservation value and cost calculations

For the Weitzman model, the same `calculate_costs!` is available as for the SD model. The difference is that it computes the position-specific search costs as a vector `cs_h`, rather than a discovery cost.


## I don't like Greek unicode letters
To facilitate setting up the models, in particular from the Python wrapper, the package also implements the following non-unicode versions that do not use Greek unicode letters for the parameters. For this, it implements a `NUModel <: Model` type and the specifications below.

```@docs
SDNU
```
```@docs
WMNU
```
```@docs
InformationStructureSpecificationNU
```
