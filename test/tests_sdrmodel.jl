using Distributions 
using StableRNGs
using StructuralSearchModels
using Test 

# Define model 
search_model = SDCore(
    β = [-0.05, 3.0],
    Ξ = 4.5,
    ρ = [-0.1],
    ξ = 2.5,
    ξρ = [-0.1], 
    dE = Normal(),
    dV = Normal(),
    dW = Normal(),
    dU0 = Normal(),
    zdfun = "log",
    zsfun = "linear"
)

ranking_model = SimpleWeightsRanking(
    γ = [0.5]
)

m = SearchRankingJointModel(;
    search_model,
    ranking_model
)

# General parameter handling (with different options)
for estimation_shock_variances in [
    [:σ_dE, :dUequaldE],
    [:σ_dV, :dUequaldV],
    [:σ_dE],
    [:σ_dV]
]
    θ = vectorize_parameters(m; estimation_shock_variances)
    m1 = StructuralSearchModels.construct_model_from_pars(θ, m;
        estimation_shock_variances)
    @test θ == vectorize_parameters(m1; estimation_shock_variances)
end

# Data generation (combines models) 
n_consumers = 10
seed = 1
rng = StableRNG(seed)
d = generate_data(m, n_consumers, 1; seed, rng,
    conditional_on_search = false, conditional_on_search_iter = 100,
    products = generate_products(n_consumers, Normal(); seed, rng))

# Check search paths
d.search_paths[1][1:2] == [14, 5]
d.search_paths[2][1:2] == [9, 0] 

# Check likelihood computation using SD as search model 
search_model = SD(
    β = [-0.05, 3.0],
    Ξ = 4.5,
    ρ = [-0.1],
    ξ = 2.5,
    dE = Normal(),
    dV = Normal(),
    dU0 = Normal(),
    zdfun = "log"
)

m = SearchRankingJointModel(;
    search_model,
    ranking_model
)

@test calculate_likelihood(m, SMLE(100), d; seed) == -812.0955888362125
# log likelihoods sum up 
@test calculate_likelihood(m, SMLE(100), d; seed) == 
    calculate_likelihood(m.search_model, SMLE(100), d; seed) + 
    calculate_likelihood(m.ranking_model, SMLE(100), d; seed)


