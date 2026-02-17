# Define model 
m = SD(
    β = [-0.05, 3.0],
    Ξ = 4.5,
    ρ = [-0.1],
    ξ = 2.5,
    dE = Normal(),
    dV = Normal(),
    dU0 = Normal(),
    zdfun = "log"
)

# Use stable RNG, guaranteeing draws stay the same across Julia versions
seed = 1
rng = StableRNG(seed)

# Data generation 
n_consumers = 10
d = generate_data(m, n_consumers, 1; seed,
    conditional_on_search = false, conditional_on_search_iter = 100,
    products = generate_products(n_consumers, Normal();  seed, rng))

dr = deepcopy(d) 

# Define model
mr = SimpleWeightsRanking(
    γ = [1000000.], # huge coefficient makes sure it's sorted 
)

rank_alternatives!(dr, mr; seed) 

# Check sorting 
for i in 1:3
    @test issorted(dr.product_characteristics[i][2:end, 1]; rev=true)
end

# Redo ranking with more reasonable coefficient to check likelihood
mr.γ[1] = 1.0 
rank_alternatives!(dr, mr; seed) 

# Check likelihood 
e = SMLE(100) 
@test calculate_likelihood(mr, e, dr) == -656.9373550938587