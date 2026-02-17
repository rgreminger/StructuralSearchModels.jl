@with_kw mutable struct SimpleWeightsRanking{T} <: RankingModel where {T <: Real}
    γ::Vector{T} # weighting for attributes 
    indices_ranking_characteristics::Union{Vector{Int}, UnitRange{Int}, Nothing} = nothing # indices of attributes used for ranking
    dE::Distribution = Gumbel() 
end

function rank_alternatives!(data::DataSD, model::RankingModel; kwargs...)

    rng = StructuralSearchModels.get_rng(kwargs)

    _, data_chunks = StructuralSearchModels.get_chunks(length(data))

    dE = model.dE # SimpleWeights Ranking model uses Gumbel distribution for the error term

    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            xγ = zeros(size(data.consideration_sets[1], 1)) 
            ranking = zeros(Int, size(data.consideration_sets[1], 1))
            for i in chunk
                rank_alternatives!(data, model, ranking, xγ, dE, i, rng) 
            end
        end
    end

    fetch.(tasks)

    return nothing     

end


@inline function rank_alternatives!(d::DataSD, m::SimpleWeightsRanking, ranking, xγ, dE::Distribution, i, rng) 
    with_outside_option = d.product_ids[i][1] == 0
    product_chars = if isnothing(m.indices_ranking_characteristics) 
            @views d.product_characteristics[i][:, 1:end-with_outside_option]
        else
            @views d.product_characteristics[i][:, m.indices_ranking_characteristics]
        end
    n_products = length(d.consideration_sets[i]) 
    xγ .= @views product_chars * m.γ + rand(rng, dE, n_products) 
    if with_outside_option
        xγ[1] = Inf # outside option always on top 
    end
    ranking .= sortperm(xγ, rev=true)
    d.product_characteristics[i] = d.product_characteristics[i][ranking, :]
    d.product_ids[i] = d.product_ids[i][ranking]

    return nothing
end

function loglikelihood(θ::Vector{T}, model::M, estimator::SMLE, data::DataSD,
    args...; kwargs...) where {M <: RankingModel, T <: Real}

    # Extract arguments 
    γ = extract_parameters(model, θ; kwargs...)
    if !(typeof(model.dE) <: Gumbel)
        throw(ArgumentError("Estimation for SimpleWeightsRanking model only works with Gumbel distribution. Specify dE = Gumbel() in model definition."))
    end
    σ = model.dE.θ

    # Define chunks for parallelization. Each chunk is a range of consumers for which a single task 
    # calculates and sums up the likelihood. 
    _, data_chunks = get_chunks(length(data))

    # Create and define tasks for each chunk
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            # Pre-allocate arrays per task -> avoid memory allocation by not having to re-create these arrays for every consumer 
            local L = zero(T)

            for i in chunk  # Iterate over consumers in chunk 
                L += ll_rank(model, γ, σ, data, i) 
            end

            return L # Return likelihood for chunk
        end
    end

    LL = sum(fetch.(tasks))

    # prevent Inf values, helps AD
    if isinf(LL) || isnan(LL) || LL >= 0
        return -T(1e100)
    else
        return LL
    end
end

function ll_rank(m::SimpleWeightsRanking, γ::Vector{T}, σ, data::DataSD, i) where T <: Real
    with_outside_option = data.product_ids[i][1] == 0
    product_characteristics = if isnothing(m.indices_ranking_characteristics)
        @views data.product_characteristics[i][:, 1:end-with_outside_option]
    else
        @views data.product_characteristics[i][:, m.indices_ranking_characteristics]
    end
    L = zero(eltype(γ))
    denominator = zero(eltype(γ))
    for r in axes(product_characteristics, 1)[1+with_outside_option:end] # skip first product (outside option)

        # Calculate the utility of the product in rank r 
        numerator = exp(@views product_characteristics[r, :]' * γ / σ )

        denominator = numerator 
        for r2 in axes(product_characteristics, 1)[r+1:end]
            denominator += exp(@views product_characteristics[r2, :]' * γ / σ)
        end

        # Calculate the likelihood of the ranking 
        L += log(numerator / denominator)
    end
    return L
end

function vectorize_parameters(model::M; kwargs...) where {M <: SimpleWeightsRanking}
    
    @unpack γ = model

    # Vectorize parameters
    θ = γ

    return θ

end

function extract_parameters(m::SimpleWeightsRanking, θ::Vector{T}; kwargs...) where T <: Real

    γ = θ 

    return γ
end

function prepare_arguments_likelihood(
    model::SimpleWeightsRanking, estimator::SMLE, data::DataSD; kwargs...)
    # Prepare arguments for the likelihood function 

    return [nothing]
end


function construct_model_from_pars(θ::Vector{T}, model::M; kwargs...) where {T <: Real, M <: SimpleWeightsRanking}

    # Extract parameters 
    γ = extract_parameters(model, θ; kwargs...)

    # Construct model 
    model_hat = typeof(model)(γ, model.indices_ranking_characteristics, model.dE)  

    return model_hat
end

function calculate_fit_measures(m::SimpleWeightsRanking, data::DataSD, n_sim; kwargs...)
    # Calculate fit measures for the ranking model 
    # Not implemented yet 
    return "Not yet implemented."
end