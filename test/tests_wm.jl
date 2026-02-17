m = WM(
    β = [-0.1, 1.5],
    ξ = 5.5,
    ρ = [-0.8],
    dE = Normal(0, 1.0),
    dV = Normal(0, 1.0),
    dU0 = Normal(0, 1.0),
    zsfun = "log"
)

# Use stable RNG, guaranteeing draws stay the same across Julia versions
seed = 1
rng = StableRNG(seed)

# Data generation 
n_consumers = 10
d = generate_data(m, n_consumers, 1; rng, seed,
    conditional_on_search = false, conditional_on_search_iter = 100,
    products = generate_products(n_consumers, Normal(); rng, seed))

d.search_paths
@test d.consideration_sets[1][1:5] == [0, 1, 0, 0, 1]
@test d.purchase_indices[1:3] == [25, 3, 18]
@test d.search_paths[1][1:2] == [25, 2]

## Core likelihood calculation 
e = SMLE(100)
@test calculate_likelihood(m, e, d; rng, seed) == -188.74515999418938
e_conditional = SMLE(100; conditional_on_search = true) # test conditional on search
@test calculate_likelihood(m, e_conditional, d; rng, seed) == -181.33845598253282

# Estimation 
e.options_solver = (f_calls_limit = 1,) # use only 1 iterations
startvals = vectorize_parameters(m)
model_hat, estimates, likelihood_at_estimates, result_solver,
std_errors = estimate(
    m, e, d; startvals, rng, seed, compute_std_errors = false)

@test estimates≈[-0.011658551995099603, 1.4945882966369255,
    5.46636864421011, -0.8677224270695694] atol=1e-12

# General parameter handling (with different options)
for estimation_shock_variances in [
    [:σ_dE, :dUequaldE],
    [:σ_dV, :dUequaldV],
    [:σ_dE],
    [:σ_dV]
]
    θ0 = vectorize_parameters(m; estimation_shock_variances)
    m1 = StructuralSearchModels.construct_model_from_pars(θ0, m;
        estimation_shock_variances)
    @test θ0 == vectorize_parameters(m1; estimation_shock_variances)
end

# Cost computation
calculate_costs!(m, d, 10000; rng, seed)
@test m.cs[1][2]≈3.2550068765896386e-9 atol=1e-12

# Demand computation 
# dem = [calculate_demand(m, d, i, j, 5; rng, seed) for i in 1:2, j in 1:2]
# @test dem≈[0.03690244166985573 1.875275451226644e-8;
#            0.036393388945450425 6.001942664422739e-7] atol=1e-12

# Revenue computation 
rev = calculate_revenues(m, d, 1, 5; rng, seed) 
@test rev[:revenues] ≈ -1.6032108037892683 atol = 1e-12
@test rev[:demand] ≈ 8.6 atol = 1e-12


# Test when ρ is scalar 
m.ρ = m.ρ[1] 
d = generate_data(m, n_consumers, 1; rng, seed,
    conditional_on_search = false, conditional_on_search_iter = 100,
    products = generate_products(n_consumers, Normal(); rng, seed))
@test calculate_likelihood(m, e, d; rng, seed) == -188.74515999418938

############################################################################################
# Tests when only one characteristic on landing page

m = WM(
    β = [-0.05, 0.0, 0.8],
    ξ = 5.5,
    ρ = [-0.8],
    dE = Normal(0, 1.0),
    dV = Normal(0, 1.0),
    dU0 = Normal(0, 1.0),
    zsfun = "log",
    information_structure = InformationStructureSpecification(
        γ = [0.0, 0.0, 0.0], 
        κ = [0.0, 0.1, 0.0],
        indices_characteristics_β_union = 1:1, 
        indices_characteristics_γ_union= 1:0,
        indices_characteristics_κ_union = 2:2,
    )
)

d = generate_data(m, n_consumers, 1; seed, rng,
    conditional_on_search = false, conditional_on_search_iter = 100,
    products = generate_products(n_consumers, MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]); seed, rng))

# Likelihood 
@test calculate_likelihood(m, e, d; rng, seed) == -210.80306282999703
@test calculate_likelihood(m, e_conditional, d; rng, seed) == -203.54861051449006

# Demand 
# @test calculate_demand(m, d, 1, 1, 20; rng, seed) == 0.007936350892620288
# @test calculate_demand(m, d, 1, 2, 20; rng, seed) == 0.052936204593320255

#########################################################################################
## Test with inference about detail page 
m.information_structure.γ[1] = 0.1 
m.ξ = 3.5
m.information_structure.indices_characteristics_γ_union = 1:1 
m.information_structure.indices_characteristics_γ_individual = 1:1
d = generate_data(m, n_consumers, 1; seed, rng,
    products = generate_products(n_consumers, MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]); seed, rng))

@test d.consideration_sets[1][1:6] == Bool[0, 1, 0, 0, 1, 0]
@test d.purchase_indices[1:3] == [19, 3, 2]
@test d.search_paths[1][1:2] == [2, 25]


calculate_costs!(m, d, 1000; seed, rng, position_at_which_correct_beliefs = 0)
@test m.cs[1][2] ≈ 0.0005976045705265474 atol=1e-12

# Check welfare 
W = calculate_welfare(m, d, 100; rng, seed)

@test W[:average][:welfare] ≈ 2.694750336682387 atol=1e-12
@test W[:conditional_on_purchase][:welfare] ≈ 2.7119046026756166 atol=1e-12
@test W[:conditional_on_search][:welfare] ≈ 2.6933060879298827 atol=1e-12



#########################################################################################
## Test heterogeneity specification 

# pars_to_test = [  
#     [[:β], [:ξ], [:ρ]],
#     [(:β, :ξ), (:ξ, :ρ), (:β, :ξ, :ρ)]
# ]
# m = WM(
#     β = [-0.1, 1.5],
#     ξ = 5.5,
#     ρ = [-0.8],
#     dE = Normal(0, 1.0),
#     dV = Normal(0, 1.0),
#     dU0 = Normal(0, 1.0),
#     zsfun = "log"
# )
    
# test_observed_heterogeneity_spec(m, pars_to_test, -168.44562611498037) 

# ll_het = [-192.14247757635047
# -166.27913626830156
# -187.77744509987946
# -190.4229746243133
# -178.75892372626254
# -183.46868411595455]

# ll_het_qmc = [-192.95475898522332
# -165.41155241936355
# -187.09135939412693
# -190.06664100062636
# -178.18509322027276
# -183.38988512434014]

# test_unobserved_heterogeneity_spec(m, pars_to_test, ll_het, ll_het_qmc)

#########################################################################################
## Test constructors 
function construct_wm_model(β) 
    m = WM(
        β = β,
        ρ = [-0.1],
        ξ = 2.5,
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zsfun = "log"
    )
    return m
end
m = construct_wm_model([0.0, 1.0])

for β in [Int.(m.β), convert(Vector{Any}, m.β)] 
    @test isequal(construct_wm_model(β), m)
end
