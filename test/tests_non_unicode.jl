using StructuralSearchModels
using StructuralSearchModels: convert_to_greek
using Test
using StableRNGs
using Distributions

############################################################################################
# Test HeterogeneitySpecificationNU conversion
############################################################################################

@testset "HeterogeneitySpecificationNU conversion" begin
    # Test basic conversion with empty specification
    hs_ng = HeterogeneitySpecificationNU()
    hs_greek = convert_to_greek(hs_ng)

    @test hs_greek isa HeterogeneitySpecification
    @test hs_greek.parameters_with_observed_heterogeneity == hs_ng.parameters_with_observed_heterogeneity
    @test hs_greek.ψ == hs_ng.psi
    @test hs_greek.parameters_with_unobserved_heterogeneity == hs_ng.parameters_with_unobserved_heterogeneity
    @test hs_greek.distribution == hs_ng.distribution

    # Test conversion with observed heterogeneity
    hs_ng_obs = HeterogeneitySpecificationNU(
        parameters_with_observed_heterogeneity = Dict(:β => [1]),
        psi = [[0.5]]
    )
    hs_greek_obs = convert_to_greek(hs_ng_obs)

    @test hs_greek_obs.parameters_with_observed_heterogeneity == Dict(:β => [1])
    @test hs_greek_obs.ψ == [[0.5]]

    # Test conversion with unobserved heterogeneity
    hs_ng_unobs = HeterogeneitySpecificationNU(
        parameters_with_unobserved_heterogeneity = Dict(:β => [1]),
        distribution = MvNormal(Diagonal([1.0]))
    )
    hs_greek_unobs = convert_to_greek(hs_ng_unobs)

    @test hs_greek_unobs.parameters_with_unobserved_heterogeneity == Dict(:β => [1])
    @test hs_greek_unobs.distribution == MvNormal(Diagonal([1.0]))
end

############################################################################################
# Test InformationStructureSpecificationNU conversion
############################################################################################

@testset "InformationStructureSpecificationNU conversion" begin
    # Test basic conversion
    sv_ng = InformationStructureSpecificationNU(
        gamma = [0.1, 0.2],
        kappa = [0.3, 0.4],
        indices_characteristics_beta_union = 1:1,
        indices_characteristics_gamma_union = 1:1,
        indices_characteristics_kappa_union = 2:2
    )

    sv_greek = convert_to_greek(sv_ng)

    @test sv_greek isa InformationStructureSpecification
    @test sv_greek.γ == sv_ng.gamma
    @test sv_greek.κ == sv_ng.kappa
    @test sv_greek.indices_characteristics_β_union == sv_ng.indices_characteristics_beta_union
    @test sv_greek.indices_characteristics_γ_union == sv_ng.indices_characteristics_gamma_union
    @test sv_greek.indices_characteristics_κ_union == sv_ng.indices_characteristics_kappa_union
    @test sv_greek.indices_characteristics_β_individual == sv_ng.indices_characteristics_beta_individual
    @test sv_greek.indices_characteristics_γ_individual == sv_ng.indices_characteristics_gamma_individual
    @test sv_greek.indices_characteristics_κ_individual == sv_ng.indices_characteristics_kappa_individual

    # Test conversion with default constructor
    sv_ng_default = InformationStructureSpecificationNU(3)
    sv_greek_default = convert_to_greek(sv_ng_default)

    @test sv_greek_default.γ == zeros(3)
    @test sv_greek_default.κ == zeros(3)
    @test sv_greek_default.indices_characteristics_β_union == 1:2
    @test sv_greek_default.indices_characteristics_γ_union == 1:0
    @test sv_greek_default.indices_characteristics_κ_union == 1:0

    # Test equality after conversion
    sv_ng_test = InformationStructureSpecificationNU(
        gamma = [0.0, 0.0],
        kappa = [0.0, 0.0],
        indices_characteristics_beta_union = 1:1,
        indices_characteristics_gamma_union = 1:0,
        indices_characteristics_kappa_union = 1:0
    )

    sv_greek_test = convert_to_greek(sv_ng_test)
    sv_greek_expected = InformationStructureSpecification(
        γ = [0.0, 0.0],
        κ = [0.0, 0.0],
        indices_characteristics_β_union = 1:1,
        indices_characteristics_γ_union = 1:0,
        indices_characteristics_κ_union = 1:0
    )

    @test sv_greek_test == sv_greek_expected

    # Test all constructors match InformationStructureSpecification equivalents
    γ = [0.1, 0.0]
    κ = [0.0, 0.2]
    idx_β = 1:1
    idx_γ = 1:1
    idx_κ = 2:2

    # Keyword constructor
    @test convert_to_greek(InformationStructureSpecificationNU(
        gamma = γ, kappa = κ,
        indices_characteristics_beta_union = idx_β,
        indices_characteristics_gamma_union = idx_γ,
        indices_characteristics_kappa_union = idx_κ
    )) == InformationStructureSpecification(
        γ = γ, κ = κ,
        indices_characteristics_β_union = idx_β,
        indices_characteristics_γ_union = idx_γ,
        indices_characteristics_κ_union = idx_κ
    )

    # Int constructor
    @test convert_to_greek(InformationStructureSpecificationNU(3)) ==
        InformationStructureSpecification(3)

    # Positional 5-argument constructor
    @test convert_to_greek(InformationStructureSpecificationNU(γ, κ, idx_β, idx_γ, idx_κ)) ==
        InformationStructureSpecification(γ, κ, idx_β, idx_γ, idx_κ)
end

############################################################################################
# Test SDNU model conversion
############################################################################################

@testset "SDNU model conversion" begin
    # Create SDNU model
    m_ng = SDNU(
        beta = [-0.05, 3.0],
        Xi = 4.5,
        rho = [-0.1],
        xi = 2.5,
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zdfun = "log"
    )

    # Convert to Greek
    m_greek = convert_to_greek(m_ng)

    # Test type
    @test m_greek isa SD

    # Test parameter conversion
    @test m_greek.β == m_ng.beta
    @test m_greek.Ξ == m_ng.Xi
    @test m_greek.ρ == m_ng.rho
    @test m_greek.ξ == m_ng.xi
    @test m_greek.dE == m_ng.dE
    @test m_greek.dV == m_ng.dV
    @test m_greek.dU0 == m_ng.dU0
    @test m_greek.zdfun == m_ng.zdfun
    @test m_greek.cs == m_ng.cs
    @test m_greek.cd == m_ng.cd

    # Test information structure conversion
    @test m_greek.information_structure.γ == m_ng.information_structure.gamma
    @test m_greek.information_structure.κ == m_ng.information_structure.kappa

    # Test heterogeneity conversion
    @test m_greek.heterogeneity.ψ == m_ng.heterogeneity.psi
    @test m_greek.heterogeneity.distribution == m_ng.heterogeneity.distribution

    # Test with custom information structure
    m_ng_custom = SDNU(
        beta = [-0.05, 0.0, 0.8],
        Xi = 1.0,
        rho = [-0.1],
        xi = 1.0,
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zdfun = "linear",
        information_structure = InformationStructureSpecificationNU(
            gamma = [0.1, 0.0, 0.0],
            kappa = [0.0, 0.1, 0.0],
            indices_characteristics_beta_union = 1:1,
            indices_characteristics_gamma_union = 1:1,
            indices_characteristics_kappa_union = 2:2
        )
    )

    m_greek_custom = convert_to_greek(m_ng_custom)

    @test m_greek_custom.information_structure.γ == [0.1, 0.0, 0.0]
    @test m_greek_custom.information_structure.κ == [0.0, 0.1, 0.0]
    @test m_greek_custom.information_structure.indices_characteristics_β_union == 1:1
    @test m_greek_custom.information_structure.indices_characteristics_γ_union == 1:1
    @test m_greek_custom.information_structure.indices_characteristics_κ_union == 2:2
end

############################################################################################
# Test WMNU model conversion
############################################################################################

@testset "WMNU model conversion" begin
    # Create WMNU model
    m_ng = WMNU(
        beta = [-0.05, 3.0],
        xi = 2.5,
        rho = [-0.1],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zsfun = "log"
    )

    # Convert to Greek
    m_greek = convert_to_greek(m_ng)

    # Test type
    @test m_greek isa WM

    # Test parameter conversion
    @test m_greek.β == m_ng.beta
    @test m_greek.ξ == m_ng.xi
    @test m_greek.ρ == m_ng.rho
    @test m_greek.dE == m_ng.dE
    @test m_greek.dV == m_ng.dV
    @test m_greek.dU0 == m_ng.dU0
    @test m_greek.zsfun == m_ng.zsfun
    @test m_greek.cs == m_ng.cs

    # Test information structure conversion
    @test m_greek.information_structure.γ == m_ng.information_structure.gamma
    @test m_greek.information_structure.κ == m_ng.information_structure.kappa

    # Test heterogeneity conversion
    @test m_greek.heterogeneity.ψ == m_ng.heterogeneity.psi
    @test m_greek.heterogeneity.distribution == m_ng.heterogeneity.distribution

    # Test with custom information structure
    m_ng_custom = WMNU(
        beta = [-0.05, 0.0, 0.8],
        xi = 1.0,
        rho = [-0.1],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zsfun = "linear",
        information_structure = InformationStructureSpecificationNU(
            gamma = [0.1, 0.0, 0.0],
            kappa = [0.0, 0.1, 0.0],
            indices_characteristics_beta_union = 1:1,
            indices_characteristics_gamma_union = 1:1,
            indices_characteristics_kappa_union = 2:2
        )
    )

    m_greek_custom = convert_to_greek(m_ng_custom)

    @test m_greek_custom.information_structure.γ == [0.1, 0.0, 0.0]
    @test m_greek_custom.information_structure.κ == [0.0, 0.1, 0.0]
    @test m_greek_custom.information_structure.indices_characteristics_β_union == 1:1
    @test m_greek_custom.information_structure.indices_characteristics_γ_union == 1:1
    @test m_greek_custom.information_structure.indices_characteristics_κ_union == 2:2
end

############################################################################################
# Test SDNU constructors match SD constructors
############################################################################################

@testset "SDNU constructors match SD" begin
    β = [-0.05, 3.0]
    Ξ = 4.5
    ρ = [-0.1]
    ξ = 2.5
    dE = Normal()
    dV = Normal()
    dU0 = Uniform()
    zdfun = "log"
    is_nu = InformationStructureSpecificationNU(
        gamma = [0.1, 0.0],
        kappa = [0.0, 0.2],
        indices_characteristics_beta_union = 1:1,
        indices_characteristics_gamma_union = 1:1,
        indices_characteristics_kappa_union = 2:2
    )
    is_greek = convert_to_greek(is_nu)
    het_nu = HeterogeneitySpecificationNU()
    het_greek = convert_to_greek(het_nu)

    m_sd_kw = SD(β = β, Ξ = Ξ, ρ = ρ, ξ = ξ, dE = dE, dV = dV, dU0 = dU0,
        zdfun = zdfun, information_structure = is_greek, heterogeneity = het_greek)

    # Keyword constructor
    m_nu_kw = SDNU(β, Ξ, ρ, ξ, dE, dV, dU0, zdfun;
        information_structure = is_nu, heterogeneity = het_nu)
    @test convert_to_greek(m_nu_kw) == m_sd_kw

    # Keyword constructor with defaults
    m_sd_defaults = SD(β = β, Ξ = Ξ, ρ = ρ, ξ = ξ, dE = dE, dV = dV, dU0 = dU0,
        zdfun = zdfun)
    m_nu_defaults = SDNU(β, Ξ, ρ, ξ, dE, dV, dU0, zdfun)
    @test convert_to_greek(m_nu_defaults) == m_sd_defaults

    # Positional constructor
    m_nu_pos = SDNU(β, Ξ, ρ, ξ, dE, dV, dU0, zdfun, is_nu, het_nu, nothing, nothing)
    @test convert_to_greek(m_nu_pos) == m_sd_kw

    # Positional constructor with Vector{Any} β
    m_nu_any = SDNU(Any[-0.05, 3.0], Ξ, ρ, ξ, dE, dV, dU0, zdfun, is_nu, het_nu, nothing, nothing)
    @test convert_to_greek(m_nu_any) == m_sd_kw
end

############################################################################################
# Test WMNU constructors match WM constructors
############################################################################################

@testset "WMNU constructors match WM" begin
    β = [-0.05, 3.0]
    ξ = 2.5
    ρ = [-0.1]
    dE = Normal()
    dV = Normal()
    dU0 = Uniform()
    zsfun = "log"
    is_nu = InformationStructureSpecificationNU(
        gamma = [0.1, 0.0],
        kappa = [0.0, 0.2],
        indices_characteristics_beta_union = 1:1,
        indices_characteristics_gamma_union = 1:1,
        indices_characteristics_kappa_union = 2:2
    )
    is_greek = convert_to_greek(is_nu)
    het_nu = HeterogeneitySpecificationNU()
    het_greek = convert_to_greek(het_nu)

    m_wm_kw = WM(β = β, ξ = ξ, ρ = ρ, dE = dE, dV = dV, dU0 = dU0,
        zsfun = zsfun, information_structure = is_greek, heterogeneity = het_greek)

    # Keyword constructor
    m_nu_kw = WMNU(β, ξ, ρ, dE, dV, dU0, zsfun;
        information_structure = is_nu, heterogeneity = het_nu)
    @test convert_to_greek(m_nu_kw) == m_wm_kw

    # Keyword constructor with defaults
    m_wm_defaults = WM(β = β, ξ = ξ, ρ = ρ, dE = dE, dV = dV, dU0 = dU0, zsfun = zsfun)
    m_nu_defaults = WMNU(β, ξ, ρ, dE, dV, dU0, zsfun)
    @test convert_to_greek(m_nu_defaults) == m_wm_defaults

    # Positional constructor
    m_nu_pos = WMNU(β, ξ, ρ, dE, dV, dU0, zsfun, is_nu, het_nu, nothing, nothing)
    @test convert_to_greek(m_nu_pos) == m_wm_kw

    # Positional constructor with Vector{Any} β
    m_nu_any = WMNU(Any[-0.05, 3.0], ξ, ρ, dE, dV, dU0, zsfun;
        information_structure = is_nu, heterogeneity = het_nu)
    @test convert_to_greek(m_nu_any) == m_wm_kw
end

############################################################################################
# Test that non-unicode models work with existing functions via conversion
############################################################################################

@testset "SDNU integration with existing functions" begin
    seed = 1
    rng = StableRNG(seed)

    # Create SDNU model
    m_ng = SDNU(
        beta = [-0.05, 3.0],
        Xi = 4.5,
        rho = [-0.1],
        xi = 2.5,
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zdfun = "log"
    )

    # Create equivalent Greek model
    m_greek = SD(
        β = [-0.05, 3.0],
        Ξ = 4.5,
        ρ = [-0.1],
        ξ = 2.5,
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zdfun = "log"
    )

    # Test data generation
    n_consumers = 5
    products = generate_products(n_consumers, Normal(); rng, seed)

    rng = StableRNG(seed)
    d_ng = generate_data(m_ng, n_consumers, 1; rng, seed, products = products)

    rng = StableRNG(seed)
    d_greek = generate_data(m_greek, n_consumers, 1; rng, seed, products = products)

    @test d_ng.purchase_indices == d_greek.purchase_indices
    @test d_ng.consideration_sets == d_greek.consideration_sets

    # Test likelihood calculation
    e = SMLE(100)

    rng = StableRNG(seed)
    ll_ng = calculate_likelihood(m_ng, e, d_ng; rng, seed)

    rng = StableRNG(seed)
    ll_greek = calculate_likelihood(m_greek, e, d_greek; rng, seed)

    @test ll_ng ≈ ll_greek atol=1e-10

    # Test vectorize_parameters
    θ_ng = vectorize_parameters(m_ng)
    θ_greek = vectorize_parameters(m_greek)

    @test θ_ng == θ_greek
end

@testset "WMNU integration with existing functions" begin
    seed = 2
    rng = StableRNG(seed)

    # Create WMNU model
    m_ng = WMNU(
        beta = [-0.05, 3.0],
        xi = 2.5,
        rho = [-0.1],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zsfun = "log"
    )

    # Create equivalent Greek model
    m_greek = WM(
        β = [-0.05, 3.0],
        ξ = 2.5,
        ρ = [-0.1],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        zsfun = "log"
    )

    # Test data generation
    n_consumers = 5
    products = generate_products(n_consumers, Normal(); rng, seed)

    rng = StableRNG(seed)
    d_ng = generate_data(m_ng, n_consumers, 1; rng, seed, products = products)

    rng = StableRNG(seed)
    d_greek = generate_data(m_greek, n_consumers, 1; rng, seed, products = products)

    @test d_ng.purchase_indices == d_greek.purchase_indices
    @test d_ng.consideration_sets == d_greek.consideration_sets

    # Test likelihood calculation
    e = SMLE(100)

    rng = StableRNG(seed)
    ll_ng = calculate_likelihood(m_ng, e, d_ng; rng, seed)

    rng = StableRNG(seed)
    ll_greek = calculate_likelihood(m_greek, e, d_greek; rng, seed)

    @test ll_ng ≈ ll_greek atol=1e-10

    # Test vectorize_parameters
    θ_ng = vectorize_parameters(m_ng)
    θ_greek = vectorize_parameters(m_greek)

    @test θ_ng == θ_greek
end
