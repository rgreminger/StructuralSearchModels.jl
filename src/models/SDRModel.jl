###########################################################################################
# Parameter handling
function vectorize_parameters(m::M; kwargs...) where {M <: SearchRankingJointModel}

    θ1 = vectorize_parameters(m.search_model; kwargs...)

    θ2 = vectorize_parameters(m.ranking_model; kwargs...)

    return vcat(θ1, θ2)
end

function construct_model_from_pars(θ::Vector{T}, m::M; kwargs...) where {M <: SearchRankingJointModel, T <: Real}

    search_model_new, ind_last_par = construct_model_from_pars(θ, m.search_model; kwargs..., return_ind_last_par = true)

    ranking_model_new = construct_model_from_pars(θ[ind_last_par:end], m.ranking_model; kwargs...)
    # Construct new model
    m_new = M(search_model_new, ranking_model_new)

    return m_new
end

############################################################################################
# Data generation

function generate_data(m::SearchRankingJointModel, n_consumers, n_sessions_per_consumer; kwargs...)
    # Produce first data from search model
    d = generate_data(m.search_model, n_consumers, n_sessions_per_consumer; kwargs...)
    # Rank alternatives using ranking model
    rank_alternatives!(d, m.ranking_model; kwargs...)
    # Generate search paths again using the newly ranked alternatives
    d = generate_data(m.search_model, d; kwargs...)
    return d
end



############################################################################################
# Likelihood calculation

function prepare_arguments_likelihood(
    model::SearchRankingJointModel, estimator::SMLE, data::DataSD; kwargs...)

    # Prepare arguments for the likelihood function
    args = prepare_arguments_likelihood(model.search_model, estimator, data; kwargs...)

    return args
end

function loglikelihood(θ::Vector{T}, model::M, estimator::SMLE, data::DataSD,
        args...; kwargs...) where {M <: SearchRankingJointModel, T <: Real}

    ll_search_model, ind_last_par = loglikelihood(θ, model.search_model, estimator, data, args...; kwargs..., return_ind_last_par = true)
    ll_ranking_model = loglikelihood(θ[ind_last_par:end], model.ranking_model, estimator, data; kwargs...)

    return ll_search_model + ll_ranking_model

end


#############################################################################################
# Fit measures
function calculate_fit_measures(m::SearchRankingJointModel, data::DataSD, n_sim; kwargs...)
    # Calculate fit measures for the search model
    fit_measures_search = calculate_fit_measures(m.search_model, data, n_sim; kwargs...)

    # Calculate fit measures for the ranking model
    fit_measures_ranking = calculate_fit_measures(m.ranking_model, data, n_sim; kwargs...)

    # Combine fit measures
    fit_measures = Dict(:search_model => fit_measures_search,
                        :ranking_model => fit_measures_ranking)

    return fit_measures
end
