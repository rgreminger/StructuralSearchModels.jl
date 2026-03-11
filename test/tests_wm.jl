@testset "data generation" begin
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

    @test d.consideration_sets[1][1:5] == [0, 1, 0, 0, 1]
    @test d.purchase_indices[1:3] == [25, 3, 18]
    @test d.search_paths[1][1:2] == [25, 2]
end

@testset "likelihood and estimation" begin
    m = WM(
        β = [-0.1, 1.5],
        ξ = 5.5,
        ρ = [-0.8],
        dE = Normal(0, 1.0),
        dV = Normal(0, 1.0),
        dU0 = Normal(0, 1.0),
        zsfun = "log"
    )

    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10
    d = generate_data(m, n_consumers, 1; rng, seed,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))

    # Core likelihood calculation
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
end

@testset "parameter handling" begin
    m = WM(
        β = [-0.1, 1.5],
        ξ = 5.5,
        ρ = [-0.8],
        dE = Normal(0, 1.0),
        dV = Normal(0, 1.0),
        dU0 = Normal(0, 1.0),
        zsfun = "log"
    )

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
end

@testset "cost computation" begin
    m = WM(
        β = [-0.1, 1.5],
        ξ = 5.5,
        ρ = [-0.8],
        dE = Normal(0, 1.0),
        dV = Normal(0, 1.0),
        dU0 = Normal(0, 1.0),
        zsfun = "log"
    )

    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10
    d = generate_data(m, n_consumers, 1; rng, seed,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))

    calculate_costs!(m, d, 10000; rng, seed)
    @test m.cs[1][2]≈3.2550068765896386e-9 atol=1e-12
end

@testset "revenue computation" begin
    m = WM(
        β = [-0.1, 1.5],
        ξ = 5.5,
        ρ = [-0.8],
        dE = Normal(0, 1.0),
        dV = Normal(0, 1.0),
        dU0 = Normal(0, 1.0),
        zsfun = "log"
    )

    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10
    d = generate_data(m, n_consumers, 1; rng, seed,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))

    rev = calculate_revenues(m, d, 1, 5; rng, seed)
    @test rev[:revenues] ≈ -1.6032108037892683 atol = 1e-12
    @test rev[:demand] ≈ 8.6 atol = 1e-12
end

@testset "scalar rho" begin
    m = WM(
        β = [-0.1, 1.5],
        ξ = 5.5,
        ρ = [-0.8],
        dE = Normal(0, 1.0),
        dV = Normal(0, 1.0),
        dU0 = Normal(0, 1.0),
        zsfun = "log"
    )

    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10

    e = SMLE(100)
    e.options_solver = (f_calls_limit = 1,)

    # Test when ρ is scalar
    m.ρ = m.ρ[1]
    d = generate_data(m, n_consumers, 1; rng, seed,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))
    @test calculate_likelihood(m, e, d; rng, seed) == -188.74515999418938
end

@testset "one characteristic on landing page" begin
    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10

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

    e = SMLE(100)
    e.options_solver = (f_calls_limit = 1,)
    e_conditional = SMLE(100; conditional_on_search = true)

    d = generate_data(m, n_consumers, 1; seed, rng,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, MvNormal(LinearAlgebra.Diagonal([1.0, 1.0])); seed, rng))

    # Likelihood
    @test calculate_likelihood(m, e, d; rng, seed) == -210.80306282999703
    @test calculate_likelihood(m, e_conditional, d; rng, seed) == -203.54861051449006
end

@testset "inference about detail page" begin
    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10

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

    m.information_structure.γ[1] = 0.1
    m.ξ = 3.5
    m.information_structure.indices_characteristics_γ_union = 1:1
    m.information_structure.indices_characteristics_γ_individual = 1:1
    d = generate_data(m, n_consumers, 1; seed, rng,
        products = generate_products(n_consumers, MvNormal(LinearAlgebra.Diagonal([1.0, 1.0])); seed, rng))

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
end

@testset "constructors" begin
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
end
