# Models 

At the core of the `StructuralSearchModels.jl` are different models, implemented as subtypes of the abstract `Model` type. Currently, the following two main search models are implemented:

- Weitzman Model
- Search and Discovery Model 

Because the Weitzman model is a special case of the Search and Discovery model, the Weitzman model is implemented as a subtype of the Search and Discovery model.

For each model, there are different specifications available. Specifications are implemented as subtypes of the respective abstract model type and differ in terms of shocks. 

Below is a list of the available models and specifications. 

## Weitzman Model 

Currently, the following specification is available for the Weitzman model:

```@docs
WM1{T}
```

## Search and Discovery Model 

Currently, the following specification is available for the Weitzman model:

```@docs
SD1{T}
```



