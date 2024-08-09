using StructuralSearchModels, Revise, Distributions, StatsBase, Random, BenchmarkTools

m = SDCore(
    β = [1.0, 1.0], 
    Ξ = 5.0, 
    ρ = [-1.0], 
    ξ = 0.5, 
    ξρ = [0.0], 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(), 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 100000
@time data, utility_purchases = 
                generate_data(m, n_consumers, 2, 1; 
                conditional_on_click = false, conditional_on_click_iter = 100); 


data.consumer_indices
## 
using BenchmarkTools
n = 10000
a = rand(n,n) 
b = [rand(n) for i in 1:n]


function filla(a)
    Threads.@threads for i in eachindex(a) 
        a[i] = rand()
    end
end
function fillb(b)
    Threads.@threads for i in eachindex(b)
        for j in eachindex(b[i]) 
            b[i][j] = rand()
        end
    end
end

filla(a)
fillb(b)

a = rand(n,n) 
b = [rand(n) for i in 1:n]

@btime filla(a)
@btime fillb(b)