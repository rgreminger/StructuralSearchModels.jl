using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools, CairoMakie


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
        dW = Normal(0, 0) , 
        zdfun = "linear", 
        zsfun = "linear"
    )


    n_consumers = 20000
    data, utility_purchases = 
                    generate_data(m, n_consumers, 2; seed = 123, 
                    conditional_on_click = false, conditional_on_click_iter = 100); 

    # Get cd cost   
    calculate_costs!(m, data, 10090000; seed = 127, position_at_which_correct_beliefs = 0)

    # Get discovery value 
    chars = vcat([d.product_characteristics[i][d.product_ids[i] .> 0, :] for i in eachindex(d)]...) # excludes outside option 
    xβ = chars * m.β
    μ = xβ
    σ = std(xβ)

    G = Normal(mean(xβ), sqrt(m.dV.σ^2 + var(xβ)) ) # products by default are drawn from normal distribution
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
        ξ = 3.0, 
        ξρ = [0.0], 
        dE = Normal(), 
        dV = Normal(), 
        dU0 = Normal(), 
        dW = Normal(0, 0) , 
        zdfun = "linear", 
        zsfun = "linear"
    )

    calculate_costs!(m, data, 1) # cd irrelevant here, so few draws only 
    
    ξ = calculate_ξ(m) 
    println("###################")
    println("m.cs = $(m.cs)")
    println("m.ξ = $(m.ξ)")
    println("ξ = $(ξ)")
end
test_search_cost_correct()
    