############################################################################################

@with_kw mutable struct HeterogeneitySpecificationNU{T} <: AbstractHeterogeneitySpecification  where {T <: Real}
    # Parameters for observed heterogeneity (multiplied with session_characteristics)
    parameters_with_observed_heterogeneity::Dict{Symbol, Any} = Dict{Symbol, Any}()
    psi::Vector{Vector{T}} = Vector{Vector{Float64}}()

    _observed_hs::HeterogeneityCases = parse_heterogeneity_specification(parameters_with_observed_heterogeneity)

    # Parameters and specification for unobserved heterogeneity
    parameters_with_unobserved_heterogeneity::Dict{Symbol, Any} = Dict{Symbol, Any}()
    distribution::D where D <: MvNormal = MvNormal(Diagonal([0.0]))
    _unobserved_hs::HeterogeneityCases = parse_heterogeneity_specification(parameters_with_unobserved_heterogeneity)

    @assert length(psi) == length(parameters_with_observed_heterogeneity) "Length of psi must be equal to the length of parameters_with_observed_heterogeneity. Have length(psi) = $(length(psi)) and length(parameters_with_observed_heterogeneity) = $(length(parameters_with_observed_heterogeneity))."
    @assert length(distribution) == length(parameters_with_unobserved_heterogeneity) ||
        (isempty(parameters_with_unobserved_heterogeneity) && distribution.Σ[1] == 0) "Distribution must match number of parameters with heterogeneity. Have length(distribution) = $(length(distribution))  and length(parameters_with_unobserved_heterogeneity) = $(length(parameters_with_unobserved_heterogeneity)). The covariance matrix is $(distribution.Σ), and should be set to match parameters_with_unobserved_heterogeneity."
end

function convert_to_greek(hs::HeterogeneitySpecificationNU)
    return HeterogeneitySpecification(
        parameters_with_observed_heterogeneity = hs.parameters_with_observed_heterogeneity,
        ψ = hs.psi,
        parameters_with_unobserved_heterogeneity = hs.parameters_with_unobserved_heterogeneity,
        distribution = hs.distribution
    )
end

############################################################################################
"""
    InformationStructureSpecificationNU{T} <: AbstractSpecification
Non-unicode version of the specification for the information structure in the Search and Discovery model. This specification includes the parameter gamma for the search value, as well as selectors for the characteristics that enter both the search value and utility through xbeta, and the characteristics that enter only the search value through xgamma. By default, is initialized with everything as nothing, which means `gamma` is not used, and all characteristics are used for both xbeta.

## Fields:
- `gamma::Vector{T}`: Vector of parameters for the search value. If `empty` (default), no search value is used.
- `kappa::Vector{T}`: Vector of parameters for the utility only characteristics. If `empty`, no utility only characteristics.
- `indices_characteristics_beta_union::Union{UnitRange{Int}, Vector{Int}}`: All characteristics that enter the search value and utility through `xbeta` in at least one session.
- `indices_characteristics_gamma_union::Union{UnitRange{Int}, Vector{Int}}`: All characteristics that enter the search value and utility through `xgamma` in at least one session.
- `indices_characteristics_kappa_union::Union{UnitRange{Int}, Vector{Int}}`: All characteristics that enter the search value and utility through `xkappa` in at least one session.
- `indices_characteristics_beta_individual::Union{UnitRange{Int}, Vector{Int}, Vector{UnitRange{Int}}, Vector{Vector{Int}}}`. By default is the same as `indices_characteristics_beta_union`, but can be set to individual indices for each session.
- `indices_characteristics_gamma_individual::Union{UnitRange{Int}, Vector{Int}, Vector{UnitRange{Int}}, Vector{Vector{Int}}}`. By default is the same as `indices_characteristics_gamma_union`, but can be set to individual indices for each session.
- `indices_characteristics_kappa_individual::Union{UnitRange{Int}, Vector{Int}, Vector{UnitRange{Int}}, Vector{Vector{Int}}}`. By default is the same as `indices_characteristics_kappa_union`, but can be set to individual indices for each session.
"""
@with_kw mutable struct InformationStructureSpecificationNU{T} <: AbstractSpecification where {T <: Real}
    gamma::Vector{T}
    kappa::Vector{T}

    indices_characteristics_beta_union::Union{UnitRange{Int}, Vector{Int}}
    indices_characteristics_gamma_union::Union{UnitRange{Int}, Vector{Int}}
    indices_characteristics_kappa_union::Union{UnitRange{Int}, Vector{Int}}

    indices_characteristics_beta_individual::Union{UnitRange{Int}, Vector{Int},
        Vector{UnitRange{Int}}, Vector{Vector{Int}}} = indices_characteristics_beta_union
    indices_characteristics_gamma_individual::Union{UnitRange{Int}, Vector{Int},
        Vector{UnitRange{Int}}, Vector{Vector{Int}}} = indices_characteristics_gamma_union
    indices_characteristics_kappa_individual::Union{UnitRange{Int}, Vector{Int},
        Vector{UnitRange{Int}}, Vector{Vector{Int}}} = indices_characteristics_kappa_union
end

function convert_to_greek(sv::InformationStructureSpecificationNU)
    return InformationStructureSpecification(
        γ = sv.gamma,
        κ = sv.kappa,
        indices_characteristics_β_union = sv.indices_characteristics_beta_union,
        indices_characteristics_γ_union = sv.indices_characteristics_gamma_union,
        indices_characteristics_κ_union = sv.indices_characteristics_kappa_union,
        indices_characteristics_β_individual = sv.indices_characteristics_beta_individual,
        indices_characteristics_γ_individual = sv.indices_characteristics_gamma_individual,
        indices_characteristics_κ_individual = sv.indices_characteristics_kappa_individual
    )
end

function InformationStructureSpecificationNU(n::Int)
    gamma = zeros(n)
    kappa = zeros(n)
    indices_characteristics_beta = 1:n-1
    indices_characteristics_gamma = 1:0
    indices_characteristics_kappa = 1:0

    return InformationStructureSpecificationNU(gamma, kappa, indices_characteristics_beta,
        indices_characteristics_gamma, indices_characteristics_kappa)
end

function InformationStructureSpecificationNU(gamma, kappa, indices_characteristics_beta,
        indices_characteristics_gamma, indices_characteristics_kappa)

    return InformationStructureSpecificationNU(gamma, kappa, indices_characteristics_beta,
        indices_characteristics_gamma, indices_characteristics_kappa,
        indices_characteristics_beta, indices_characteristics_gamma, indices_characteristics_kappa)
end

function ==(s1::InformationStructureSpecificationNU, s2::InformationStructureSpecificationNU)
    return s1.gamma == s2.gamma && s1.kappa == s2.kappa &&
        s1.indices_characteristics_beta_union == s2.indices_characteristics_beta_union &&
        s1.indices_characteristics_gamma_union == s2.indices_characteristics_gamma_union &&
        s1.indices_characteristics_kappa_union == s2.indices_characteristics_kappa_union &&
        s1.indices_characteristics_beta_individual == s2.indices_characteristics_beta_individual &&
        s1.indices_characteristics_gamma_individual == s2.indices_characteristics_gamma_individual &&
        s1.indices_characteristics_kappa_individual == s2.indices_characteristics_kappa_individual
end

####################################################################
abstract type NUModel end

"""
*Search and Discovery* model `SDNU{T} <: NUModel`. This is the same as `SD` without Greek
unicode letters for the fields. The parameterization of the `SD` model is as follows:
- uᵢⱼ = xⱼβ + xⱼκ + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ = xⱼβ + xⱼγ + ξ + νᵢⱼ
- uᵢ₀ = x₀'β + ηᵢ , ηᵢ ~ dU0
- zd(h) = zdfun(Ξ, ρ, pos) with ρ ≤ 0

## Fields:
- `beta::Vector{T}`: Vector of preference weights.
- `Xi::T`: Baseline Ξ for position 1 (not demeaned).
- `rho::Union{T, Vector{T}}`: Parameter(s) governing decrease of Ξ across positions.
- `xi::T`: Baseline ξ.
- `dE::Distribution`: Distribution of εᵢⱼ.
- `dV::Distribution`: Distribution of νᵢⱼ.
- `dU0::Distribution`: Distribution of ηᵢ.
- `zdfun::String`: Select functional form f(Ξ, ρ, h) that determines the discovery value in position h.
- `information_structure::InformationStructureSpecificationNU{T}`: Specification of information structure, including `gamma`, `kappa` and characteristics for `beta`, `gamma`, and `kappa`. See `InformationStructureSpecificationNU` for details.
- `cs::Union{T, Nothing}=nothing`: Search costs. Initialized as `nothing` and only used for welfare calculations. Can be added through `calculate_costs!(m, data; kwargs...)`.
- `cd::Union{T, Nothing}=nothing`: Discovery costs. Initialized in the same way as `cs`, and is also added in `calculate_costs!(m, data; kwargs...)`.
- `heterogeneity::HeterogeneitySpecificationNU`: Specification of heterogeneity (unobserved and observed) in the model. By default assumes homogeneous model.
"""
@with_kw mutable struct SDNU{T} <: NUModel where T <: Real
    beta::Vector{T}
    Xi::T
    rho::Union{T, Vector{T}}
    xi::T
    dE::Distribution
    dV::Distribution
    dU0::Distribution
    zdfun::String

    # Parameters with defaults
    information_structure::InformationStructureSpecificationNU{T} = InformationStructureSpecificationNU(length(beta))
    heterogeneity::HeterogeneitySpecificationNU{T} = HeterogeneitySpecificationNU()

    cs::Union{T, Nothing} = nothing
    cd::Union{T, Nothing} = nothing

    @assert rho[1]<=0 "ρ[1] must be less or equal to zero for weakly decreasing discovery value across positions."
end

function SDNU(β, Ξ, ρ, ξ, dE, dV, dU0, zdfun, cs, cd, heterogeneity)
    Ξ, ξ = promote(Ξ, ξ)
    T = eltype(Ξ)
    ρ = convert_ρ(ρ, T)
    return SDNU(convert(Vector{T}, β), Ξ, ρ, ξ, dE, dV, dU0, zdfun, cs, cd, heterogeneity)
end

# this one for some reason is necessary as Parameters.jl does not seem to enforce T for β
function SDNU(β::Vector{Any}, Ξ, ρ, ξ, dE, dV, dU0, zdfun, cs, cd, heterogeneity)
    Ξ, ξ = promote(Ξ, ξ)
    T = eltype(Ξ)
    ρ = convert_ρ(ρ, T)
    return SDNU(convert(Vector{T}, β), Ξ, ρ, ξ, dE, dV, dU0, zdfun, cs, cd, heterogeneity)
end

function SDNU(β, Ξ, ρ, ξ, dE, dV, dU0, zdfun; cs = nothing, cd = nothing,
    heterogeneity = HeterogeneitySpecificationNU())
    Ξ, ξ = promote(Ξ, ξ)
    T = eltype(Ξ)
    ρ = convert_ρ(ρ, T)
    return SDNU(convert(Vector{T}, β), Ξ, ρ, ξ, dE, dV, dU0, zdfun, cs, cd, heterogeneity)
end

function convert_to_greek(m::SDNU)
    return SD(
        β = m.beta,
        Ξ = m.Xi,
        ρ = m.rho,
        ξ = m.xi,
        dE = m.dE,
        dV = m.dV,
        dU0 = m.dU0,
        zdfun = m.zdfun,
        cs = m.cs,
        cd = m.cd,
        information_structure = convert_to_greek(m.information_structure),
        heterogeneity = convert_to_greek(m.heterogeneity)
    )
end

"""
*Weitzman model* `WMNU{T} <: NUModel`. This is the same as `WM` without Greek
unicode letters for the fields. The parameterization of the `WM` model is as follows:
- uᵢⱼ = xⱼβ + xⱼκ + νᵢⱼ + εᵢⱼ,  εᵢⱼ ~ dE, νᵢⱼ ~ dV
- zsᵢⱼ(h) = xⱼβ + xⱼγ + ξ(h) + νᵢⱼ
- uᵢ₀ = x₀'β + ηᵢ , ηᵢ ~ dU0
- ξ(h) = zsfun(ξ, ρ, pos)

## Fields:
- `beta::Vector{T}`: Vector of preference weights.
- `xi::T`: Baseline xi.
- `rho::Union{T, Vector{T}} `: Parameters governing decrease of ξ across positions.
- `dE::Distribution`: Distribution of εᵢⱼ.
- `dV::Distribution`: Distribution of νᵢⱼ.
- `dU0::Distribution`: Distribution of ηᵢ.
- `zsfun::String`: Select functional form f(ξ, ρ, h) that determines the search value in position h.
- `information_structure::InformationStructureSpecificationNU{T}`: Specification of information structure, including `gamma`, `kappa` and characteristics for `beta`, `gamma`, and `kappa`. See `InformationStructureSpecificationNU` for details.
- `cs::Union{T, Nothing}`: Search costs. Initialized as `nothing` and only used for welfare calculations. Can be added through `calculate_costs!(m, data; kwargs...)`.
- `cs_h::Union{Vector{T}, Nothing}`: Initialized in the same way as `cs`, and is also added in `calculate_costs!(m, data; kwargs...)`.
- `heterogeneity::HeterogeneitySpecificationNU`: Specification of heterogeneity (unobserved and observed) in the model. By default assumes homogeneous model.
"""
@with_kw mutable struct WMNU{T} <: NUModel where T <: Real
    beta::Vector{T}
    rho::Union{T, Vector{T}}
    xi::T
    dE::Distribution
    dV::Distribution
    dU0::Distribution
    zsfun::String

    # Parameters with defaults
    information_structure::InformationStructureSpecificationNU{T} = InformationStructureSpecificationNU(length(beta))
    heterogeneity::HeterogeneitySpecificationNU{T} = HeterogeneitySpecificationNU()
    cs::Union{T, Nothing} = nothing
    cs_h::Union{Vector{T}, Nothing} = nothing
end

function WMNU(β, ξ, ρ, dE, dV, dU0, zsfun, cs, cs_h, heterogeneity)
    T = eltype(ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    cs_h = convert_cs(cs_h, T)
    if cs_h isa Real
        throw(ArgumentError("cs_h must be a vector or nothing."))
    end
    return WMNU(convert(Vector{T}, β), ξ, ρ, dE, dV, dU0, zsfun, cs, cs_h, heterogeneity)
end

function WMNU(β::Vector{Any}, ξ, ρ, dE, dV, dU0, zsfun; cs = nothing, cs_h = nothing,
    heterogeneity = HeterogeneitySpecificationNU())
    T = eltype(ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    cs_h = convert_cs(cs_h, T)
    return WMNU(convert(Vector{T}, β), ξ, ρ, dE, dV, dU0, zsfun, cs, cs_h, heterogeneity)
end

function WMNU(β, ξ, ρ, dE, dV, dU0, zsfun; cs = nothing, cs_h = nothing,
    heterogeneity = HeterogeneitySpecificationNU())
    T = eltype(ξ)
    ρ = convert_ρ(ρ, T)
    cs = convert_cs(cs, T)
    cs_h = convert_cs(cs_h, T)
    return WMNU(convert(Vector{T}, β), ξ, ρ, dE, dV, dU0, zsfun, cs, cs_h, heterogeneity)
end

function convert_to_greek(m::WMNU)
    return WM(
        β = m.beta,
        ρ = m.rho,
        ξ = m.xi,
        dE = m.dE,
        dV = m.dV,
        dU0 = m.dU0,
        zsfun = m.zsfun,
        cs = m.cs,
        information_structure = convert_to_greek(m.information_structure),
        heterogeneity = convert_to_greek(m.heterogeneity)
    )
end



function calculate_welfare(m::M, d::DataSD, n_sim; kwargs...) where M <: NUModel
    calculate_welfare(convert_to_greek(m), d, n_sim; kwargs...)
end
function generate_data(m::M, n_consumers, n_products; kwargs...) where M <: NUModel
    generate_data(convert_to_greek(m), n_consumers, n_products; kwargs...)
end
generate_data(m::M, d::DataSD; kwargs...) where M <: NUModel =
    generate_data(convert_to_greek(m), d; kwargs...)

function calculate_fit_measures(m::M, d::DataSD, n_sim; kwargs...) where M <: NUModel
    calculate_fit_measures(convert_to_greek(m), d, n_sim; kwargs...)
end

function calculate_costs!(m::M, d, n_draws_cd; kwargs...) where M <: NUModel
    m1 = convert_to_greek(m)
    calculate_costs!(m1, d, n_draws_cd; kwargs...)
    for f in  [:cs, :cd, :cs_h]
        if hasfield(M, f)
            setfield!(m, f, getfield(m1, f))
        end
    end
    return nothing
end

function vectorize_parameters(m::M; kwargs...) where M <: NUModel
    vectorize_parameters(convert_to_greek(m); kwargs...)
end

function estimate(model::Model, estimator::SMLE, d::Data; kwargs...) where Model <: NUModel
    estimate(convert_to_greek(model), estimator, d; kwargs...)
end

function calculate_likelihood(m::M, e::SMLE, d::DataSD; kwargs...) where M <: NUModel
    calculate_likelihood(convert_to_greek(m), e, d; kwargs...)
end

function calculate_revenues(m::M, d::DataSD, kprice, n_draws; kwargs...) where M <: NUModel
    calculate_revenues(convert_to_greek(m), d, kprice, n_draws; kwargs...)
end

function calculate_revenues_i(m::M, d::DataSD, i, kprice, n_draws; kwargs...) where M <: NUModel
    calculate_revenues_i(convert_to_greek(m), d, i, kprice, n_draws; kwargs...)
end

function calculate_demand(m::M, d::DataSD, i, j, n_draws; kwargs...) where M <: NUModel
    calculate_demand(convert_to_greek(m), d, i, j, n_draws; kwargs...)
end
