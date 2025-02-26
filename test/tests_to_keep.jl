using StructuralSearchModels, Revise, Distributions, StatsBase, Random


## 


function test_discovery_cost_correct()
    # Verify costs correct: need that Ξ again same as m.Ξ after getting cd and recomputing Ξ

    m = SDCore(
        β = [0.0, 5.0],
        Ξ = 1.0,
        ρ = [-0.01],
        ξ = 1.0,
        ξρ = [0.0],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        dW = Normal(0, 0),
        zdfun = "linear",
        zsfun = "linear"
    )

    n_consumers = 20000
    data, utility_purchases = generate_data(m, n_consumers, 2; seed = 123,
        conditional_on_search = false, conditional_on_search_iter = 100)

    # Get cd cost   
    calculate_costs!(m, data, 10090000; seed = 127, position_at_which_correct_beliefs = 0)

    # Get discovery value 
    chars = vcat([data.product_characteristics[i][data.product_ids[i] .> 0, :]
                  for i in eachindex(data)]...) # excludes outside option 
    xβ = chars * m.β
    μ = xβ
    σ = std(xβ)

    G = Normal(mean(xβ), sqrt(m.dV.σ^2 + var(xβ))) # products by default are drawn from normal distribution
    Ξ = calculate_discovery_value(G, m)

    println("###################")
    println("cd = $(m.cd)")
    println("Ξ = $(Ξ)")
    println("m.Ξ = $(m.Ξ)")
end

test_discovery_cost_correct()

function test_search_cost_correct()
    # Verify costs correct: need that Ξ again same as m.Ξ after getting cd and recomputing Ξ
    m = SDCore(
        β = [0.0, 5.0],
        Ξ = 1.0,
        ρ = [-0.01],
        ξ = 0.5,
        ξρ = [0.0],
        dE = Normal(),
        dV = Normal(),
        dU0 = Normal(),
        dW = Normal(0, 0),
        zdfun = "linear",
        zsfun = "linear"
    )

    n_consumers = 20000
    data, utility_purchases = generate_data(m, n_consumers, 2; seed = 123,
        conditional_on_search = false, conditional_on_search_iter = 100)

    calculate_costs!(m, data, 1) # cd irrelevant here, so few draws only 

    ξ = calculate_ξ(m)
    println("###################")
    println("m.cs = $(m.cs)")
    println("m.ξ = $(m.ξ)")
    println("ξ = $(ξ)")
end
test_search_cost_correct()

##
function test_welfare_calculations_same()
    m = SDCore(
        β = [0.3, -1e-16],
        Ξ = 5.0,
        ρ = [-0.6],
        ξ = -1e-17,
        ξρ = [0.0],
        dE = Normal(0.0, 1.0),
        dV = Normal(0.0, 1.0),
        dU0 = Normal(0, 1.0),
        dW = Normal(0, 0.0),
        zdfun = "linear",
        zsfun = "linear"
    )

    n_consumers = 100
    seed = 92458 # seed only matters for products, rest shocks are zero 
    @time data, utility_purchases = generate_data(m, n_consumers, 1; seed,
        conditional_on_search = false, conditional_on_search_iter = 100)

    calculate_costs!(m, data, 100000)

    @time we = calculate_welfare(m, data, 1000; method = "effective_values", seed)

    @time wc = calculate_welfare(m, data, 1000; method = "simulate_paths", seed)

    println("###################")
    println("Avg welfare effective values = $(we[1][1])")
    println("Avg welfare simulate paths = $(wc[1][1])")
    println("Welfare conditional on click = $(we[2][1])")
    println("Welfare conditional on click = $(wc[2][1])")
    println("Welfare conditional on purchase = $(we[3][1])")
    println("Welfare conditional on purchase = $(wc[3][1])")

    println("###################")
    println("Discovery costs paid avg effective values = $(we[1][3])")
    println("Discovery costs paid avg simulate paths = $(wc[1][4])")

    println("###################")
    println("cs = $(m.cs)")
    println("cd = $(m.cd)")
end

test_welfare_calculations_same()

##

m = SDCore(
    β = [0.3, -1e-16],
    Ξ = 5.0,
    ρ = [-0.6],
    ξ = -1e-17,
    ξρ = [0.0],
    dE = Normal(0.0, 1.0),
    dV = Normal(0.0, 1.0),
    dU0 = Normal(0, 1.0),
    dW = Normal(0, 0.0),
    zdfun = "linear",
    zsfun = "linear"
)

n_consumers = 100
seed = 928 # seed only matters for products, rest shocks are zero 
data, utility_purchases = generate_data(m, n_consumers, 1;
    seed = 1,
    conditional_on_search = false, conditional_on_search_iter = 100);

calculate_costs!(m, data, 100000)

n_sim = 10
n_products = maximum(length.(d.product_ids))
# draws_u0 = [[rand(m.dU0) for i in 1:length(d)] for s in 1:n_sim] 
@time wc = calculate_welfare(m, data, n_sim; method = "simulate_paths", draws_u0);

println("Avg. welfare = $(wc[1][1])")

## 

draws_u0 = [[rand(m.dU0) for i in 1:length(d)] for s in 1:n_sim]

## 
@time wc = calculate_welfare(m, data, n_sim; method = "effective_values");