using StructuralSearchModels, Revise, Distributions, StatsBase, Random

m = SDCore(
    β = [1.0, -100.0], 
    Ξ = 10.0, 
    ρ = [-1.0], 
    ξ = 0.3, 
    ξρ = [0.0], 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(), 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 100
product_ids, product_characteristics, positions, paths, consideration_sets, indices_purchase, indices_stop = generate_data(m, n_consumers, 1, 1); 
paths[1] 

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