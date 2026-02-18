# Data

To simplify handling data for estimating different models, the package implements data types. The main one (for now) is the `DataSD` struct as a subtype of `Data`. This is a mutable struct type that gathers the different data for search sessions. For model estimation, data from other types (e.g., `DataFrame`) needs to be gathered into this type. The [Tutorials](tutorials.md) provide some examples of how this can be appropriately constructed.

Most fields are required, but some fill in `nothing` by default. For example, the search order or where consumers stopped scrolling may not be observed, in which case, these fields default to `nothing`.


```@docs
DataSD
```

The data tracks everything on a session level, with the `consumer_ids` field allowing to track consumers over multiple sessions (which is currently not yet used).

Data can be accessed with `d.fieldname`, where `d` is a `DataSD` object. Indexing is then based on the different fields. For example, the following accesses the product characteristics of the first session in the data: `d.product_characteristics[1]`.

For product characteristics, note that the last column always is a dummy indicator for the outside option.

## Data Generation

Data from a search model `model` can be generated with the `generate_data` function. There are two versions.

The first one generates data from scratch, which requires specifying how many consumers and sessions to simulate (each consumer can have multiple sessions, which is not yet used). By default, it generates generic products using `generate_products`.

The second one takes products, sessions etc. as given and only simulates new search paths from `model`.

```@docs
generate_data
```

```@docs
generate_products
```

#### Keyword options

The `generate_data` function passes `kwargs_path_generation` into the path generation. The following options currently are available:
- `seed`: use a specific seed to generate data.
- `conditional_on_search=false`: whether generate only search paths with at least one search. `false` by default.
- `conditional_on_search_iter=100`: to generate search paths with at least one search for a particular session (with products etc.), the code tries by default up to 100 times using new draws for the shocks.

## Convenience functions


The package exports some convenience functions for the `DataSD` type. These include:
- `d == d1` checks whether two `DataSD` objects are the same.
- `d[1:5]` selects the first five search sessions from the data
- `fill_indices_min_discover!(data::DataSD)` adds the `min_discover_indices` from the consideration sets.

The following are other functions that are available to manipulate the data and can be helpful for estimation.

```@docs
update_positions!
```
```@docs
add_product_fe!
```
```@docs
fill_indices_min_discover!
```
```@docs
merge_data
```
