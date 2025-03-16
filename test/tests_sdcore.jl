# Define model 
m = SDCore(
    β = [-0.05, 3.0],
    Ξ = 4.5,
    ρ = [-0.1],
    ξ = 2.5,
    ξρ = [-0.2],
    dE = Normal(),
    dV = Normal(),
    dU0 = Normal(),
    dW = Normal(),
    zdfun = "log",
    zsfun = "linear"
)

# Use stable RNG, guaranteeing draws stay the same across Julia versions
seed = 1
rng = StableRNG(seed)

# Generate data 
n_consumers = 10
d = generate_data(m, n_consumers, 1; rng, seed,
    conditional_on_search = false, conditional_on_search_iter = 100,
    products = generate_products(n_consumers; rng, seed))

@test d.consideration_sets[1][1:6] == Bool[0, 1, 0, 1, 1, 0]
@test d.purchase_indices[1:3] == [1, 2, 1]
@test d.min_discover_indices[1] == 5
@test d.search_paths[1][1:2] == [2, 4]
@test d.stop_indices[3] == 2

# Cost computation 

# Verify costs correct: need that Ξ again same as m.Ξ after getting cd and recomputing Ξ
# note: requires many draws for accuracy 
m = SDCore(
    β = [0.0, 5.0],
    Ξ = 1.0,
    ρ = [-0.01],
    ξ = 1.0,
    ξρ = [0.0],
    dE = Normal(),
    dV = Normal(),
    dU0 = Normal(),
    dW = Normal(0, 0),
    zdfun = "linear",
    zsfun = "linear"
)

n_consumers = 3000
data = generate_data(m, n_consumers, 2; seed, rng,
    conditional_on_search = false, conditional_on_search_iter = 100)

# Compute cd cost   
calculate_costs!(m, data, 1000000; seed, rng, position_at_which_correct_beliefs = 0)

# Compute discovery value from costs 
chars = vcat([data.product_characteristics[i][data.product_ids[i] .> 0, :]
              for i in eachindex(data)]...) # excludes outside option 
xβ = chars * m.β
μ = xβ
σ = std(xβ)

G = Normal(mean(xβ), sqrt(m.dV.σ^2 + var(xβ))) # products by default are drawn from normal distribution
Ξ = calculate_discovery_value(G, m)

# Compare computed discovery value with true one 
@test Ξ≈m.Ξ atol=1e-3

# Compute search value 
ξ = calculate_ξ(m)

# Compare computed search value with true one
@test ξ≈m.ξ atol=1e-6

# Compute search value for some other distributions 
for dist in [Exponential(), LogNormal(), Uniform(-3, 3)] 
    m1 = deepcopy(m) 
    m1.dE = dist
    calculate_costs!(m1, data, 1; seed, rng, position_at_which_correct_beliefs = 0)

    @assert m1.cs > 0  # if this is not the case, the test is not meaningful

    ξ1 = calculate_ξ(m1)
    @test ξ1 ≈ m1.ξ atol=1e-6
end 