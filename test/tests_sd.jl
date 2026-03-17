@testset "data generation" begin
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
    d = generate_data(m, n_consumers, 1; rng, seed,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))

    @test d.consideration_sets[1][1:6] == [0, 0, 0, 0, 1, 1]
    @test d.purchase_indices[1:3] == [1, 1, 27]
    @test d.min_discover_indices[1] == 31
    @test d.search_paths[1][1:2] == [25, 12]
    @test d.stop_indices[6] == 8
end

@testset "likelihood and estimation" begin
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

    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10
    d = generate_data(m, n_consumers, 1; rng, seed,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))

    # Likelihood calculation with stop indices
    e = SMLE(100)
    @test calculate_likelihood(m, e, d; rng, seed) == -196.01499750022353
    e_conditional = SMLE(100; conditional_on_search = true) # test conditional on search
    @test calculate_likelihood(m, e_conditional, d; rng, seed) == -185.14129722567333

    # Estimation with stop indices
    e.options_solver = (f_calls_limit = 1,) # use only 1 iterations
    startvals = vectorize_parameters(m)

    model_hat, estimates, likelihood_at_estimates, result_solver,
    std_errors = estimate(
        m, e, d; startvals, rng, seed, compute_std_errors = false)

    @test estimates ≈ [-0.2034767906790771, 2.9557276433518527,
        4.489520406052184, -0.19405332073951995, 2.5034182412550066] atol = 1e-12

    # Likelihood calculation without stop indices
    d.stop_indices = nothing
    @test calculate_likelihood(m, e, d; rng, seed) == -193.60838494006262
    @test calculate_likelihood(m, e_conditional, d; rng, seed) == -191.51347119337805

    # Estimation without stop indices
    model_hat, estimates, likelihood_at_estimates, result_solver,
    std_errors = estimate(
        m, e, d; startvals, rng, seed, compute_std_errors = false)

    @test estimates ≈ [-0.33380157449721964, 2.952996485252172,
        4.492705726089025, -0.1449647262037665, 2.517376324194926] atol = 1e-12

    # Parameter rescaling: scale = ones should give identical results
    e_scaled = SMLE(100; parameter_rescaling = ones(length(startvals)), options_solver = (f_calls_limit = 1,))
    _, estimates_scaled, likelihood_scaled, _, _ = estimate(
        m, e_scaled, d; startvals, rng, seed, compute_std_errors = false)
    @test likelihood_scaled ≈ likelihood_at_estimates atol = 1e-12

    # build_inverse_hessian_scaler: returns a vector of positive scaling factors
    scale = build_inverse_hessian_scaler(m, SMLE(100), d, startvals; seed, rng)

    @test scale isa Vector
    @test length(scale) == length(startvals)
    @test all(scale .> 0)
end

@testset "parameter handling" begin
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

    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10
    d = generate_data(m, n_consumers, 1; rng, seed,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))
    d.stop_indices = nothing

    calculate_costs!(m, d, 10000; rng, seed)
    @test m.cd[1] ≈ 0.000274314103444567 atol = 1e-12
    @test m.cs[1][2] ≈ 0.002004137179128191 atol = 1e-12
end

@testset "revenue computation" begin
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

    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10
    d = generate_data(m, n_consumers, 1; rng, seed,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))
    d.stop_indices = nothing

    rev = calculate_revenues(m, d, 1, 5; rng, seed)
    @test rev[:revenues] ≈ -3.3629844129520774 atol = 1e-12
    @test rev[:demand] ≈ 4.4 atol = 1e-12
end

@testset "scalar rho" begin
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
    @test calculate_likelihood(m, e, d; rng, seed) == -196.01499750022353
end

@testset "one characteristic on landing page" begin
    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10

    m = SD(
        β = [-0.05, 0.0, 0.8],
        Ξ = 1.0,
        ρ = [-0.1],
        ξ = 1.0,
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zdfun = "linear",
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

    @test calculate_likelihood(m, e, d; rng, seed) == -27.309162449059276
    @test calculate_likelihood(m, e_conditional, d; rng, seed) == -0.11566765613705599
end

@testset "inference about detail page" begin
    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10

    m = SD(
        β = [-0.05, 0.0, 0.8],
        Ξ = 1.0,
        ρ = [-0.1],
        ξ = 1.0,
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zdfun = "linear",
        information_structure = InformationStructureSpecification(
            γ = [0.0, 0.0, 0.0],
            κ = [0.0, 0.1, 0.0],
            indices_characteristics_β_union = 1:1,
            indices_characteristics_γ_union= 1:0,
            indices_characteristics_κ_union = 2:2,
        )
    )

    m.information_structure.γ[1] = 0.1
    m.information_structure.indices_characteristics_γ_union = 1:1
    m.information_structure.indices_characteristics_γ_individual = 1:1
    d = generate_data(m, n_consumers, 1; seed, rng,
        products = generate_products(n_consumers, MvNormal(LinearAlgebra.Diagonal([1.0, 1.0])); seed, rng))

    @test d.consideration_sets[1][1:6] == Bool[0, 1, 0, 1, 0, 0]
    @test d.purchase_indices[1:3] == [4, 1, 4]
    @test d.min_discover_indices[1] == 4
    @test d.search_paths[1][1:2] == [2, 4]
    @test d.stop_indices[2] == 2

    calculate_costs!(m, d, 1000; seed, rng, position_at_which_correct_beliefs = 0)
    @test m.cd[1] ≈ 0.6916216022294652 atol=1e-12
    @test m.cs[1][2] ≈ 0.08626262028418104 atol=1e-12

    # Check welfare
    W = calculate_welfare(m, d, 100; rng, seed)

    @test W[:average][:welfare] ≈ 0.771399304172124 atol=1e-12
    @test W[:conditional_on_purchase][:welfare] ≈ 0.32547137914156704 atol=1e-12
    @test W[:conditional_on_search][:welfare] ≈ 0.35539302417365465 atol=1e-12
end

@testset "drop undiscovered products" begin
    seed = 1
    rng = StableRNG(seed)
    n_consumers = 10

    m = SD(
        β = [-0.05, 0.0, 0.8],
        Ξ = 1.0,
        ρ = [-0.1],
        ξ = 1.0,
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zdfun = "linear",
        information_structure = InformationStructureSpecification(
            γ = [0.0, 0.0, 0.0],
            κ = [0.0, 0.1, 0.0],
            indices_characteristics_β_union = 1:1,
            indices_characteristics_γ_union= 1:0,
            indices_characteristics_κ_union = 2:2,
        )
    )
    m.information_structure.γ[1] = 0.1
    m.information_structure.indices_characteristics_γ_union = 1:1
    m.information_structure.indices_characteristics_γ_individual = 1:1

    products = generate_products(n_consumers, MvNormal(LinearAlgebra.Diagonal([1.0, 1.0])); seed, rng)
    d_full = generate_data(m, n_consumers, 1; seed, rng, products,
        drop_undiscovered_products = false)
    d_dropped = generate_data(m, n_consumers, 1; seed, rng,
        products,
        drop_undiscovered_products = true)

    e = SMLE(100)

    for i in eachindex(d_full)
        @test length(d_dropped.product_ids[i]) == d_full.stop_indices[i]
    end

    @test calculate_likelihood(m, e, d_dropped; rng, seed) == calculate_likelihood(m, e, d_full; rng, seed)
end

@testset "constructors" begin
    function construct_sd_model(β)
        m = SD(
            β = β,
            Ξ = 4.5,
            ρ = [-0.1],
            ξ = 2.5,
            dE = Normal(),
            dV = Normal(),
            dU0 = Normal(),
            zdfun = "log"
        )
        return m
    end
    m = construct_sd_model([0.0, 1.0])

    for β in [Int.(m.β), convert(Vector{Any}, m.β)]
        m1 = construct_sd_model(β)
        @test isequal(m1, m)
    end
end
