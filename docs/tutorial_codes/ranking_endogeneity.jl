using Distributions
using StructuralSearchModels

seed = 1
n_consumers = 1000

# Define model
msd = SD(
    β = [-0.1, 0.2, 5],
    Ξ = 4.5,
    ρ = [-0.1],
    ξ = 2.5,
    dE = Normal(),
    dV = Normal(),
    dU0 = Normal(),
    zdfun = "log"
)

# Set up products
Σ = [1.0 0.5; 0.5 1.0]
dc = MvNormal([0.0, 0.0], Σ)
products = generate_products(n_consumers, Normal(); seed, distribution = dc)

# Define where characteristics shown
indices_list_characteristics = 1:2 # only first one on list

# Define ranking model
mr = SimpleWeightsRanking(
    γ = [-2, 2.0], # huge coefficient makes sure it's sorted
    dE = Gumbel(0, 1)
)


# Combine models
m = SearchRankingJointModel(msd, mr)

du = generate_data(m, n_consumers, 1; seed,
        indices_list_characteristics,
        products)

dr = generate_data(m, n_consumers, 1; seed,
    indices_list_characteristics,
    products)

fitm = calculate_fit_measures(m, du, 100; seed)
println(fitm[:search_model][:click_stats_data][:click_probability_per_position][1:10])
hcat(dr.product_characteristics[1], du.product_characteristics[1])

##
function run_estimation(msearch, mjoint, du, dr, seed)
    _, estimates_u, _ =
        estimate(msearch, SMLE(100), du; seed)
    _, estimates_r, _ =
        estimate(msearch, SMLE(100), dr; seed)
    _, estimates_joint, _ =
        estimate(mjoint, SMLE(100), dr; seed)

    estimates_search_joint = estimates_joint[1:length(estimates_r)]
    return hcat(estimates_u, estimates_r, estimates_search_joint,
    estimates_r - estimates_search_joint,  vectorize_parameters(msearch))
end

estimates = run_estimation(msd, m, du, dr, seed)

############################################################################################
## Now drop second characteristic -> omitted variable

# Define new model
msdo = deepcopy(msd)
msdo.β = msd.β[[1, 3]]
mro = deepcopy(mr)
mro.γ = mro.γ[[1]]

du_o = deepcopy(du)
du_o.product_characteristics = [du.product_characteristics[i][:, [1, 3]] for i in eachindex(du)]
dr_o = deepcopy(dr)
dr_o.product_characteristics = [dr.product_characteristics[i][:, [1, 3]] for i in eachindex(du)]
mo = SearchRankingJointModel(msdo, mro)

estimates = run_estimation(msdo, mo, du_o, dr_o, seed)
