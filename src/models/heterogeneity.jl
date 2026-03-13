@with_kw mutable struct HeterogeneityCases
    β::Vector{Int} = Vector{Int}()
    γ::Vector{Int} = Vector{Int}()
    Ξ::Bool = false
    ρ::Vector{Int} = Vector{Int}()
    ξ::Bool = false
    ξρ::Vector{Int} = Vector{Int}()
end


"""
    HeterogeneitySpecification{T}

Specification of observed and unobserved consumer heterogeneity for a search model.

# Fields
- `parameters_with_observed_heterogeneity`: Dict mapping parameter symbols (e.g., `:β`, `:γ`) to indices of heterogeneity dimensions for observed heterogeneity (interacted with session characteristics via `ψ`).
- `ψ`: Coefficients for observed heterogeneity, one vector per heterogeneous parameter.
- `parameters_with_unobserved_heterogeneity`: Dict mapping parameter symbols to indices of heterogeneity dimensions for unobserved heterogeneity (integrated out via `distribution`).
- `distribution`: Multivariate normal distribution for unobserved heterogeneity; its dimension must match `parameters_with_unobserved_heterogeneity`.
"""
@with_kw mutable struct HeterogeneitySpecification{T} <: AbstractHeterogeneitySpecification  where {T <: Real}
    # Parameters for observed heterogeneity (multiplied with session_characteristics)
    parameters_with_observed_heterogeneity::Dict{Symbol, Any} = Dict{Symbol, Any}()
    ψ::Vector{Vector{T}} = Vector{Vector{Float64}}()

    _observed_hs::HeterogeneityCases = parse_heterogeneity_specification(parameters_with_observed_heterogeneity)

    # Parameters and specification for unobserved heterogeneity
    parameters_with_unobserved_heterogeneity::Dict{Symbol, Any} = Dict{Symbol, Any}()
    distribution::D where D <: MvNormal = MvNormal(Diagonal([0.0]))
    _unobserved_hs::HeterogeneityCases = parse_heterogeneity_specification(parameters_with_unobserved_heterogeneity)

    @assert length(ψ) == length(parameters_with_observed_heterogeneity) "Length of ψ must be equal to the length of parameters_with_observed_heterogeneity. Have length(ψ) = $(length(ψ)) and length(parameters_with_observed_heterogeneity) = $(length(parameters_with_observed_heterogeneity))."
    @assert length(distribution) == length(parameters_with_unobserved_heterogeneity) ||
        (isempty(parameters_with_unobserved_heterogeneity) && distribution.Σ[1] == 0) "Distribution must match number of parameters with heterogeneity. Have length(distribution) = $(length(distribution))  and length(parameters_with_unobserved_heterogeneity) = $(length(parameters_with_unobserved_heterogeneity)). The covariance matrix is $(distribution.Σ), and should be set to match parameters_with_unobserved_heterogeneity."
end

@inline has_observed_heterogeneity(hp::HeterogeneitySpecification) =
    length(hp.parameters_with_observed_heterogeneity) > 0
@inline has_unobserved_heterogeneity(hp::HeterogeneitySpecification) =
    length(hp.parameters_with_unobserved_heterogeneity) > 0

@inline has_heterogeneity(hp::HeterogeneitySpecification) =
    has_observed_heterogeneity(hp) || has_unobserved_heterogeneity(hp)


@inline has_observed_heterogeneity(m::Model) =
    has_observed_heterogeneity(m.heterogeneity)
@inline has_unobserved_heterogeneity(m::Model) =
    has_unobserved_heterogeneity(m.heterogeneity)
@inline has_heterogeneity(m::Model) =
    has_heterogeneity(m.heterogeneity)


function ==(hp1::HeterogeneitySpecification, hp2::HeterogeneitySpecification)
    return hp1.parameters_with_observed_heterogeneity == hp2.parameters_with_observed_heterogeneity &&
           hp1.parameters_with_unobserved_heterogeneity == hp2.parameters_with_unobserved_heterogeneity &&
           hp1.distribution == hp2.distribution &&
           hp1.ψ == hp2.ψ
end

function extract_heterogeneity_parameters(m::M, θ::Vector{T}, c; kwargs...) where {M <: SearchModel, T <: Real}

    ψ = if has_observed_heterogeneity(m.heterogeneity) &&
            (!haskey(kwargs, :fixed_parameters) || :ψ ∉ get(kwargs, :fixed_parameters, nothing))

        ψ = Vector{Vector{T}}()
        for i in eachindex(m.heterogeneity.ψ)
            n_ψ = length(m.heterogeneity.ψ[i])
            ψ = push!(ψ, θ[c:c + n_ψ - 1])
            c += n_ψ
        end
        ψ
    else
        [T.(ψ) for ψ in m.heterogeneity.ψ]
    end

    U = if has_unobserved_heterogeneity(m.heterogeneity) &&
            (!haskey(kwargs, :fixed_parameters) || :Σ ∉ get(kwargs, :fixed_parameters, nothing))

        U = zeros(T, size(m.heterogeneity.distribution.Σ))
        for i in axes(U, 1), j in axes(U, 2)[i:end]
            U[i, j] = θ[c]
            c += 1
        end
        U
    else
        Matrix{T}(cholesky(m.heterogeneity.distribution.Σ).U)
    end

    return ψ, U, c
end

"""
    update_heterogeneity_specification!(m)

Reparse the heterogeneity dictionaries in `m.heterogeneity` and update the internal
`_observed_hs` and `_unobserved_hs` case objects in-place. Call this after modifying
`parameters_with_observed_heterogeneity` or `parameters_with_unobserved_heterogeneity`
on an existing model.
"""
function update_heterogeneity_specification!(m)
    m.heterogeneity._observed_hs =
        parse_heterogeneity_specification(m.heterogeneity.parameters_with_observed_heterogeneity)
    m.heterogeneity._unobserved_hs =
        parse_heterogeneity_specification(m.heterogeneity.parameters_with_unobserved_heterogeneity)
end

function add_unobserved_heterogeneity_parameters(m::M, θ::Vector{T}) where {M <: SDModel, T <: Real}


    U = cholesky(m.heterogeneity.distribution.Σ).U

    for i in axes(U, 1), j in axes(U, 2)[i:end]
        θ = vcat(θ, U[i, j])
    end

    return θ

end

function construct_heterogeneity_distribution(dist::Distribution, U::Matrix{T}) where {T <: Real}
    if length(U) == 1 && U[1] == 0
        return MvNormal(Diagonal([0.0]))
    else
        return MvNormal(U' * U)
    end
end

function parse_heterogeneity_specification(parameters_with_heterogeneity)
    # Parse the heterogeneity specification and return a HeterogeneityCases object
    cases = HeterogeneityCases()
    for (key, value) in parameters_with_heterogeneity
        if key == :β
            cases.β = value
        elseif key == :γ
            cases.γ = value
        elseif key == :Ξ
            cases.Ξ = true
        elseif key == :ρ
            cases.ρ = value
        elseif key == :ξ
            cases.ξ = true
        elseif key == :ξρ
            cases.ξρ = value
        end
    end
    return cases
end


function construct_new_heterogeneity_specification(m::M, ψ, U)  where M <: SDModel
    heterogeneity = deepcopy(m.heterogeneity)
    heterogeneity.ψ = ψ
    heterogeneity.distribution = construct_heterogeneity_distribution(m.heterogeneity.distribution, U)

    if !isequal(cholesky(heterogeneity.distribution.Σ).U, U)
        @warn "Cholesky decomposition of the covariance matrix does not match the estimated U matrix. This renders standard errors and confidence intervals invalid."
    end

    return heterogeneity
end
