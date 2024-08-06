using StructuralSearchModels, Revise, Distributions, StatsBase, Random

m = SDCore(
    β = [1.0, 2.0, 3.0], 
    Ξ = 1.0, 
    ρ = [1.0, 2.0, 3.0], 
    ξ = 1.0, 
    ξρ = [1.0, 2.0, 3.0], 
    n_d = 3, 
    n_A0 = 3, 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(), 
    zdfun = "linear", 
    zsfun = "linear"
)


product_ids, product_characteristics = generate_data(m, 100, 10)