
# Define model 
m = SD1(
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
seed = 123
rng = StableRNG(seed)

# Data generation 
n_consumers = 10
seed = 1
d, _ = generate_data(m, n_consumers, 1; rng, seed,
    conditional_on_search = false, conditional_on_search_iter = 100,
    products = generate_products(n_consumers; rng, seed))

@test d.consideration_sets[1][1:6] == [0, 0, 0, 0, 1, 1]
@test d.purchase_indices[1:3] == [1, 1, 27]
@test d.min_discover_indices[1] == 31
@test d.search_paths[1][1:2] == [25, 12]
@test d.stop_indices[6] == 8

# Core likelihood calculation 
e = SMLE()
@test calculate_likelihood(m, e, d; rng, seed) == -193.60838494006262

# Estimation 
e.options_solver = (f_calls_limit = 1,) # use only 1 iterations
startvals = vectorize_parameters(m)
model_hat, estimates, likelihood_at_estimates, result_solver,
std_errors = estimate_model(
    m, e, d; startvals, rng, seed, compute_std_errors = false)

@test estimates ≈ [-0.33380157449721964, 2.952996485252172,
    4.492705726089025, -0.1449647262037665, 2.517376324194926] atol = 1e-12

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
@test m.cd ≈ 0.0003447560453978281 atol = 1e-12
@test m.cs ≈ 0.002004137179128191 atol = 1e-12

# Welfare 
w = calculate_welfare(m, d, 100; rng, seed, method = "effective_values")
calculate_welfare(m, d, 100; rng, seed, method = "effective_values")
@test w[:average][:discovery_costs] ≈ 0.008513750541099367 atol = 1e-12
@test w[:average][:eff_value_choice] ≈ 3.418247414691211 atol = 1e-12
@test w[:conditional_on_purchase][:discovery_costs] ≈ 0.009273746090065253 atol = 1e-12
@test w[:conditional_on_search][:discovery_costs] ≈ 0.00942015249067753 atol = 1e-12
@test w[:conditional_on_purchase][:eff_value_choice] ≈ 3.22686635230007 atol = 1e-12
@test w[:conditional_on_search][:eff_value_choice] ≈ 3.2510120783847696 atol = 1e-12

# Demand computation 
dem = [calculate_demand(m, d, i, j, 5; rng, seed) for i in 1:2, j in 1:2]
@test dem ≈ [0.6451646244745709 0.0051664017164279;
    0.6444150527878119 0.015511365724323661] atol = 1e-12
# Revenue computation 
rev = calculate_revenues(m, d, 1, 5; rng, seed) 
@test rev[1] ≈ -0.9775653397690145 atol = 1e-12
@test rev[2][1:2] ≈ [-0.0645893438951959, -0.03469696420344893] atol = 1e-12
