@testset "data generation" begin
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
        products = generate_products(n_consumers, Normal(); rng, seed))

    @test d.consideration_sets[1][1:6] == Bool[0, 1, 0, 1, 1, 0]
    @test d.purchase_indices[1:3] == [1, 2, 1]
    @test d.min_discover_indices[1] == 5
    @test d.search_paths[1][1:2] == [2, 4]
    @test d.stop_indices[3] == 2
end

@testset "cost computation" begin
    seed = 1
    rng = StableRNG(seed)

    m = SDCore(
        β = [0.5, 1.0],
        Ξ = 5.0,
        ρ = -0.0, # set to zero so that beliefs are correct with characteristics random across pos.
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
    d = generate_data(m, n_consumers, 1; seed, rng,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))

    # Compute cd cost
    calculate_costs!(m, d, 100000; seed, rng)

    # Compute discovery value from costs
    chars = vcat([d.product_characteristics[i][d.product_ids[i] .> 0, :]
                  for i in eachindex(d)]...) # excludes outside option
    xβ = chars * m.β

    G = Normal(mean(xβ), sqrt(m.dV.σ^2 + var(xβ))) # products by default are drawn from normal distribution
    Ξ = StructuralSearchModels.calculate_discovery_value(G, m, m.ξ, m.cs[1][2], m.cd[1])

    # Compare computed discovery value with true one
    # loose tolerance because of random draws
    @test abs(Ξ - m.Ξ) ≤ 0.05

    # Compute search value
    ξ = StructuralSearchModels.calculate_ξ(m.cs[1][2], m.dE)

    # Compare computed search value with true one
    @test ξ≈m.ξ atol=1e-6

    # Compute search value for some other distributions
    for dist in [Exponential(), LogNormal(), Uniform(-3, 3)]
        m1 = deepcopy(m)
        m1.dE = dist
        calculate_costs!(m1, d, 10000; seed, rng, position_at_which_correct_beliefs = 0)

        @assert m1.cs[1][2] > 0  # if this is not the case, the test is not meaningful

        ξ1 = StructuralSearchModels.calculate_ξ(m1.cs[1][2], m1.dE)
        @test ξ1 ≈ m1.ξ atol=1e-6
    end

    # Test position-specific search costs
    m.ξρ = [-0.2]
    calculate_costs!(m, d, 1000000; seed, rng)
    @test m.cs[1][2] == 0.08331547058768628
    @test m.cs[1][3] == 0.12020723389476541
end

@testset "welfare calculations" begin
    seed = 1
    rng = StableRNG(seed)

    m_base = SDCore(
        β = [0.5, 1.0],
        Ξ = 5.0,
        ρ = -0.0,
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
    d = generate_data(m_base, n_consumers, 1; seed, rng,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))

    m = SDCore(
        β = [0.5, 1.0],
        Ξ = 1.0,
        ρ = -0.01,
        ξ = 1.0,
        ξρ = [0.0],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        dW = Normal(0, 0),
        zdfun = "linear",
        zsfun = "linear"
    )
    # Compute cd cost
    calculate_costs!(m, d, 10000000; seed, rng)

    W = calculate_welfare(m, d, 1000; rng, seed)

    @test W[:average][:welfare] ≈ 1.4573242820447725 atol=0.02
    @test W[:average][:utility] ≈ 1.8843141581120142 atol=0.02
    @test W[:average][:discovery_costs] ≈ 0.3276147375436621 atol=0.02
    @test W[:average][:search_costs] ≈ 0.09937513852358099 atol=0.02

    @test W[:conditional_on_purchase][:welfare] ≈ 1.1706549615235518 atol=0.03
    @test W[:conditional_on_purchase][:utility] ≈ 1.9374599650712043 atol=0.03
    @test W[:conditional_on_purchase][:discovery_costs] ≈ 0.5982386403361757 atol=0.03
    @test W[:conditional_on_purchase][:search_costs] ≈ 0.1685663632114759 atol=0.03

    @test W[:conditional_on_search][:welfare] ≈ 1.1798425658448595 atol=0.03
    @test W[:conditional_on_search][:utility] ≈ 1.8695443523707278 atol=0.03
    @test W[:conditional_on_search][:discovery_costs] ≈ 0.531337975733615 atol=0.03
end

@testset "revenue calculations" begin
    seed = 1
    rng = StableRNG(seed)

    m_base = SDCore(
        β = [0.5, 1.0],
        Ξ = 5.0,
        ρ = -0.0,
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
    d = generate_data(m_base, n_consumers, 1; seed, rng,
        conditional_on_search = false, conditional_on_search_iter = 100,
        products = generate_products(n_consumers, Normal(); rng, seed))

    m = SDCore(
        β = [0.5, 1.0],
        Ξ = 1.0,
        ρ = -0.01,
        ξ = 1.0,
        ξρ = [0.0],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        dW = Normal(0, 0),
        zdfun = "linear",
        zsfun = "linear"
    )
    calculate_costs!(m, d, 10000000; seed, rng)

    R = calculate_revenues(m, d, 1, 100; seed, rng)

    @test R[:revenues] ≈ 823.245049712528 atol=1e-6
    @test R[:demand] ≈ 1629.76 atol=1e-6
end

@testset "one characteristic on landing page" begin
    seed = 1
    rng = StableRNG(seed)

    n_consumers = 3000
    m = SDCore(
        β = [-0.05, 0.0, 0.5],
        Ξ = 1.0,
        ρ = [-0.01],
        ξ = 1.0,
        ξρ = [0.0],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        dW = Normal(0, 0),
        zdfun = "linear",
        zsfun = "linear",
        information_structure = InformationStructureSpecification(
            γ = [0.0, 0.0, 0.0],
            κ = [0.0, 0.1, 0.0],
            indices_characteristics_β_union = 1:1,
            indices_characteristics_γ_union= 1:0,
            indices_characteristics_κ_union = 2:2,
        )
    )

    d = generate_data(m, n_consumers, 1; seed, rng,
        products = generate_products(n_consumers, MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]); seed, rng))

    @test d.consideration_sets[1][1:6] == Bool[0, 1, 0, 1, 0, 0]
    @test d.purchase_indices[1:3] == [4, 8, 4]
    @test d.min_discover_indices[1] == 4
    @test d.search_paths[1][1:2] == [2, 4]
    @test d.stop_indices[2] == 8

    # Verify costs (which are now different with some characteristics hidden on list page)
    calculate_costs!(m, d, 10000; seed, rng, position_at_which_correct_beliefs = 0)
    @test m.cd[1] ≈ 0.18833486964459709 atol=0.02
    @test m.cs[1][2] ≈ 0.08331547058768628 atol=1e-12

    # With position-specific search costs
    m.ξρ = [-0.2]
    calculate_costs!(m, d, 10000; seed, rng, position_at_which_correct_beliefs = 0)
    @test m.cs[1][3] ≈ 0.12020723389476541 atol=1e-12
    @test m.cs[1][4] ≈ 0.16867273224175544 atol=1e-12

    # Check welfare
    W = calculate_welfare(m, d, 100; rng, seed)

    @test W[:average][:welfare] ≈ 0.6045585458932817 atol=0.05
    @test W[:conditional_on_purchase][:welfare] ≈ 0.3624239459695144 atol=0.06
    @test W[:conditional_on_search][:welfare] ≈ 0.29837085631879956 atol=0.06
end

@testset "inference about detail page" begin
    seed = 1
    rng = StableRNG(seed)

    n_consumers = 3000
    m = SDCore(
        β = [-0.05, 0.0, 0.5],
        Ξ = 1.0,
        ρ = [-0.01],
        ξ = 1.0,
        ξρ = [-0.2],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        dW = Normal(0, 0),
        zdfun = "linear",
        zsfun = "linear",
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
        products = generate_products(n_consumers, MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]); seed, rng))

    @test d.consideration_sets[1][1:6] == Bool[0, 1, 0, 1, 0, 0]
    @test d.purchase_indices[1:3] == [4, 5, 6]
    @test d.min_discover_indices[1] == 4
    @test d.search_paths[1][1:2] == [2, 4]
    @test d.stop_indices[2] == 5

    calculate_costs!(m, d, 1000; seed, rng, position_at_which_correct_beliefs = 0)
    @test m.cd[1] ≈ 0.18972651307956953 atol=0.02
    @test m.cs[1][2] ≈ 0.10263615330290497 atol=1e-12

    # Check welfare
    W = calculate_welfare(m, d, 100; rng, seed)

    @test W[:average][:welfare] ≈ 0.6119811728332721 atol=0.05
    @test W[:conditional_on_purchase][:welfare] ≈ 0.37228130555093314 atol=0.06
    @test W[:conditional_on_search][:welfare] ≈ 0.30916964482558706 atol=0.06
end

@testset "no product attributes on list page" begin
    seed = 1
    rng = StableRNG(seed)

    n_consumers = 3000
    m = SDCore(
        β = [0.0, 0.0, 0.5],
        Ξ = 1.0,
        ρ = [-0.01],
        ξ = 1.0,
        ξρ = [0.0],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        dW = Normal(0, 0),
        zdfun = "linear",
        zsfun = "linear",
        information_structure = InformationStructureSpecification(
            γ = [0.0, 0.0, 0.0],
            κ = [-0.05, 0.1, 0.0],
            indices_characteristics_β_union = 1:0,
            indices_characteristics_γ_union = 1:0,
            indices_characteristics_κ_union = 1:2,
        )
    )

    d = generate_data(m, n_consumers, 1; seed, rng,
        products = generate_products(n_consumers, MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]); seed, rng))

    @test d.consideration_sets[1][1:6] == Bool[0, 1, 0, 1, 0, 0]
    @test d.purchase_indices[1:3] == [4, 8, 4]
    @test d.min_discover_indices[1] == 4
    @test d.search_paths[1][1:2] == [2, 4]
    @test d.stop_indices[2] == 8

    # Verify costs when no characteristics are revealed on list page
    calculate_costs!(m, d, 10000; seed, rng, position_at_which_correct_beliefs = 0)
    @test m.cd[1] ≈ 0.17857355508943365 atol=0.02
    @test m.cs[1][2] ≈ 0.08331547058768628 atol=1e-12

    # Check welfare
    W = calculate_welfare(m, d, 100; rng, seed)

    @test W[:average][:welfare] ≈ 1.2395397927861695 atol=0.05
    @test W[:conditional_on_purchase][:welfare] ≈ 1.0948697715654319 atol=0.06
    @test W[:conditional_on_search][:welfare] ≈ 1.1003996057162748 atol=0.06
end

@testset "varying number of attributes across sessions" begin
    seed = 1
    rng = StableRNG(seed)

    m = SDCore(
        β = [0.5, 1.0, 0.8],
        Ξ = 1.0,
        ρ = [-0.01],
        ξ = 1.0,
        ξρ = [0.0],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        dW = Normal(0, 0),
        zdfun = "linear",
        zsfun = "linear",
        information_structure = InformationStructureSpecification(
            γ = [0.0, 0.0, 0.0],
            κ = [0.1, 0.1, 0.1],
            indices_characteristics_β_union = 1:3,
            indices_characteristics_γ_union = 1:0,
            indices_characteristics_κ_union = 1:3,
            indices_characteristics_β_individual = [1:0, 1:1, 1:2],
            indices_characteristics_κ_individual = [1:3, 2:3, 3:3]
        )
    )

    # Create small dataset with 3 consumers for this test
    n_consumers_vary = 3
    d_vary = generate_data(m, n_consumers_vary, 1; seed, rng,
        products = generate_products(n_consumers_vary, MvNormal([0.0, 0.0, 0.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]); seed, rng))

    # Test that data generation succeeds with varying information structures
    @test length(d_vary.consideration_sets) == n_consumers_vary
    @test length(d_vary.purchase_indices) == n_consumers_vary

    # Verify the information structures work correctly
    # Session 1 has 0 attributes on list (β_individual = 1:0)
    # Session 2 has 1 attribute on list (β_individual = 1:1)
    # Session 3 has 2 attributes on list (β_individual = 1:2)
    @test d_vary.consideration_sets[1][1:6] == Bool[0, 1, 0, 1, 0, 0]
    @test d_vary.purchase_indices[1:3] == [4, 1, 7]
    @test d_vary.min_discover_indices == [4, 2, 7]

    # Calculate costs with varying information structures
    calculate_costs!(m, d_vary, 1000; seed, rng, position_at_which_correct_beliefs = 0)
    @test m.cd[1] ≈ 0.1759812253066064 atol=0.02
    @test m.cd[2] ≈ 0.18416607630197235 atol=0.02
    @test m.cd[3] ≈ 0.3010074860206985 atol=0.02

    # Check welfare with varying information structures
    W_vary = calculate_welfare(m, d_vary, 100; rng, seed)
    @test haskey(W_vary, :average)
    @test haskey(W_vary, :conditional_on_purchase)
    @test haskey(W_vary, :conditional_on_search)
end
