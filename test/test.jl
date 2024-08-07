using StructuralSearchModels, Revise, Distributions, StatsBase, Random

m = SDCore(
    β = [1.0, 2.0, 3.0], 
    Ξ = 1.0, 
    ρ = [1.0, 2.0, 3.0], 
    ξ = 1.0, 
    ξρ = [1.0, 2.0, 3.0], 
    n_d = 3, 
    n_A0 = 1, 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(), 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 100
product_ids, product_characteristics, positions = generate_data(m, n_consumers, 1); 
positions

StructuralSearchModels.DataSDCore(;
    consumer_indices = fill(1:1, 100), 
    product_ids, 
    product_characteristics, 
    positions,
    consideration_sets = [fill(true, 31) for i in 1:n_consumers],
    purchase_indices = 1:n_consumers,
    stop_indices = 1:n_consumers
)