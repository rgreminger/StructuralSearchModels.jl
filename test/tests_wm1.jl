
m = WM1(
    β = [-0.1, 1.5],
    ξ = 5.5,
    ρ = [-0.8],
    dE = Normal(0, 1.0),
    dV = Normal(0, 1.0),
    dU0 = Normal(0, 1.0),
    zsfun = "log"
)

# Use stable RNG, guaranteeing draws stay the same across Julia versions
seed = 123
rng = StableRNG(seed)

# Data generation 
n_consumers = 10
seed = 1
d, _ = generate_data(m, n_consumers, 1; rng, seed,
    conditional_on_search = false, conditional_on_search_iter = 100,
    products = generate_products(n_consumers; rng, seed))

d.search_paths
@test d.consideration_sets[1][1:5] == [0, 1, 0, 0, 1]
@test d.purchase_indices[1:3] == [25, 3, 18]
@test d.search_paths[1][1:2] == [25, 2]

## Core likelihood calculation 
e = SMLE()
@test calculate_likelihood(m, e, d; rng, seed) == -188.74515999418938

# Estimation 
e.options_solver = (f_calls_limit = 1,) # use only 1 iterations
startvals = vectorize_parameters(m)
model_hat, estimates, likelihood_at_estimates, result_solver,
std_errors = estimate_model(
    m, e, d; startvals, rng, seed, compute_std_errors = false)

@test estimates≈[-0.011658551995099603, 1.4945882966369255,
    5.46636864421011, -0.8677224270695694] atol=1e-12

# General parameter handling (with different options)
for distribution_options in [
    [false, false, false, false],
    [true, true, false, false],
    [true, false, true, false],
    [false, true, false, false]
]
    θ0 = vectorize_parameters(m; distribution_options)
    m1 = StructuralSearchModels.construct_model_from_pars(θ0, m;
        distribution_options)
    @test θ0 == vectorize_parameters(m1; distribution_options)
end

# Cost computation
calculate_costs!(m, d, 10000; rng, seed)
@test m.cs≈3.2550068765896386e-9 atol=1e-12

# Welfare 
w = calculate_welfare(m, d, 100; rng, seed, method = "effective_values")
calculate_welfare(m, d, 100; rng, seed, method = "effective_values")
@test w[:average][:eff_value_choice]≈2.9740008937555813 atol=1e-12
@test w[:average][:welfare]≈2.9740008937555813 atol=1e-12
@test w[:conditional_on_purchase][:eff_value_choice]≈2.998533779392342 atol=1e-12
@test w[:conditional_on_purchase][:welfare]≈2.998533779392342 atol=1e-12

# Demand computation 
dem = [calculate_demand(m, d, i, j, 5; rng, seed) for i in 1:2, j in 1:2]
@test dem≈[0.03690244166985573 1.875275451226644e-8;
           0.036393388945450425 6.001942664422739e-7] atol=1e-12

# Revenue computation 
rev = calculate_revenues(m, d, 1, 5; rng, seed)
@test rev[1]≈-1.4356899872294822 atol=1e-12
@test rev[2][1:2]≈[-0.361369698791823, -0.0023126112832036385] atol=1e-12