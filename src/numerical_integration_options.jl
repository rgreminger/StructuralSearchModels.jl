"""
    Default numerical integration that is used to integrate over the idiosyncratic shocks.
## Options
- `n_draws::Int`: Number of draws for main integration over shocks.
"""
@with_kw struct DefaultNI <: NIMethod
    n_draws::Int
end

"""
    QMC(n_draws; sampler))
    QME(; n_draws, sampler)
Quasi Monte Carlo integration method for numerical integration over unobserved heterogeneity. Builds on `QuasiMonteCarlo.jl` package to provide functionality of taking different series.

## Arguments
- `n_draws::Int`: Number of draws for main integration over shocks.

## Optional Keyword Arguments
- `sampler::Union{Distribution, SamplingAlgorithm}`: Sampling algorithm for the QMC method. Defaults to `Uniform()`, i.e., takes draws just from the uniform distribution.
"""
@with_kw struct QMC <: NIMethod
    n_draws::Int
    n_draws_discard::Int = 0
    sampler::Union{Distribution, SamplingAlgorithm} = Uniform()

    @assert typeof(sampler) <: SamplingAlgorithm  || sampler == Uniform() "Sampler must be either Uniform() or a SamplingAlgorithm from the QuasiMonteCarlo package. Have $(typeof(sampler))."
end
QMC(n_draws::Int) = QMC(;n_draws)


# Quasi-Monte-Carlo based on different sampling schemes (from QuasiMonteCarlo package)
function generate_ni_points(n_dims, n_cons, method::M, rng) where M <: QMC
    n_draws_total = method.n_draws * n_cons
    n_discard = method.n_draws_discard
    sampler = method.sampler

    points = if typeof(method.sampler) <: Distribution
        rand(rng, Uniform(), n_dims, n_draws_total + n_discard) # by using rand rather than QuasiMonteCarlo.sample, we can use the rng
    else
        QuasiMonteCarlo.sample(n_draws_total + n_discard, fill(0, n_dims), fill(1, n_dims), sampler)
    end

    # Discard first n_discard draws
    points = points[:, n_discard+1:end]

    # Transform to standard normal
    Threads.@threads for i in eachindex(points)
        points[i] = Distributions.quantile(Normal(), points[i])
    end

    # Reshape into array of arrays per consumer
    chunks = Iterators.partition(1:n_draws_total, method.n_draws)
    points = [points[:, chunk] for chunk in chunks]

    # Equal weights with QMC
    weights = ones(method.n_draws) / method.n_draws

    return points, weights
end
