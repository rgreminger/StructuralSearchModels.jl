using StructuralSearchModels, Revise, Distributions, StatsBase, Random

m = SDCore(
    β = [1.0, 2.0, 3.0], 
    Ξ = 1.0, 
    ρ = [1.0, 2.0, 3.0], 
    ξ = 1.0, 
    ξρ = [1.0, 2.0, 3.0], 
    dE = Normal(), 
    dV = Normal(), 
    dU0 = Normal(), 
    dW = Normal(), 
    zdfun = "linear", 
    zsfun = "linear"
)


n_consumers = 100
product_ids, product_characteristics, positions = generate_data(m, n_consumers, 1, 1); 
positions


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