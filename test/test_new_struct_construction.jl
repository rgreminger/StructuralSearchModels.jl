using StructuralSearchModels 

function construct_model_from_pars(θ::Vector{T}, m::SD1) where T <: Real
    m1 = SD1{T}( 
        β = [θ[2], .5], 
        Ξ = θ[1], 
        ρ = [-0.7], 
        ξ = θ[3],
        dE = Normal(0, 3.0), 
        dV = Normal(0, 3.0), 
        dU0 = Normal(0, 1), 
        zdfun = "log", 
    )

    return m1 
end


m1 = construct_model_from_pars([5.0, 1.0, 2.0], m)

function f(θ) 

    m1 = construct_model_from_pars(θ, m)

    return m1.Ξ * 10 + m1.ξ - 3* m1.β[1]

end

using ForwardDiff 

@btime ForwardDiff.gradient(f, [5.0, 1.0, 2.0])

