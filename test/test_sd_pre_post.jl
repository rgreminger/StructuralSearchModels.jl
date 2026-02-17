
using Distributions
using StructuralSearchModels
using Revise

# Define model
m = SD(
    β = [-0.05, 0.5, 0.5],
    ξ = 0.5,
    Ξ = 2.0,
    ρ = [-0.01],
    dE = Normal(0, 1.0),
    dV = Normal(0, 1.0),
    dU0 = Normal(0, 1.0),
    zdfun = "log"
)

# m = WM(
#     β = [-0.05, 0.5, 0.5],
#     ξ = 0.5,
#     ρ = [-0.1],
#     dE = Normal(0, 1.0),
#     dV = Normal(0, 1.0),
#     dU0 = Normal(0, 1.0),
#     zsfun = "log",
#     indices_list_characteristics = 1:1
# )

# Use stable RNG, guaranteeing draws stay the same across Julia versions
seed = 1

# Generate data
n_consumers = 5000
@time d = generate_data(m, n_consumers, 1; seed,
    conditional_on_search = false, conditional_on_search_iter = 100,
    drop_undiscovered_products = false,
    products = generate_products(n_consumers, MvNormal([0.0, 0.0], [2.0 0.0; 0.0 2]); seed, rng)) ;

@time d1 = generate_data(m, n_consumers, 1; seed,
    conditional_on_search = false, conditional_on_search_iter = 100,
    indices_list_characteristics = 1:1,
    products = generate_products(n_consumers, MvNormal([0.0, 0.0], [2.0 0.0; 0.0 2]); seed, rng)) ;

e = SMLE()

for data in [d, d1]
    @time calculate_likelihood(m, e, data; seed)
end

;

##

@time generate_data(m, d; seed)

@time calculate_demand(m, d, 1, 2, 20; seed)


;

##
function drop_undiscovered_products(d)
    d1 = deepcopy(d)
    for i in eachindex(d)
        stop_index = d.stop_indices[i]
        d1.search_paths[i] = d.search_paths[i][1:stop_index - 1]
        d1.product_characteristics[i] = d.product_characteristics[i][1:stop_index, :]
        d1.product_ids[i] = d.product_ids[i][1:stop_index]
    end
    return d1
end
d1 = drop_undiscovered_products(d)

println("size d: ", size(d.product_characteristics[1]))
println("size d1: ", size(d1.product_characteristics[1]))

##
@time ll0 = calculate_likelihood(m, e, d; seed)
@time ll1 = calculate_likelihood(m, e, d1; seed)
println("ll d: ", ll0)
println("ll d1: ", ll1)
println("Δll: ", ll0 - ll1)


##
m.indices_list_characteristics = 1:1
m.β[2] = 0.0
calculate_costs!(m, d, 5000000; seed)


hcat(cs_h, m.cs_h)
##



hcat(m.cs_h, cs_h0)

##

a, b = StructuralSearchModels.get_xβ(m, d)

##
true_vals = vectorize_parameters(m)
startvals = copy(true_vals)

model_hat, estimates, likelihood_at_estimates, result_solver,
std_errors = estimate_model(
    m, e, d; startvals, seed, compute_std_errors = true)

hcat(true_vals, estimates, std_errors, true_vals - estimates)

##

@time calculate_likelihood(m, e, d; seed)
##
m1 = deepcopy(m)
m1.indices_list_characteristics = 1:1
calculate_demand(m1, d, 1, 2, 20; seed)

##

m.indices_list_characteristics = 1:1
@time dem = [calculate_demand(m, d, i, j, 20) for i in 1:100, j in 2:20]

fit_measures = calculate_fit_measures(m, d[1:100], 1000; seed)

hcat(mean(dem; dims = 1)', fit_measures[:purchase_stats_sim][:purchase_probability_per_position][2:20])



##
