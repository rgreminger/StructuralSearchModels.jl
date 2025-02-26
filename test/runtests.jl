using Distributions
using StructuralSearchModels
using Test 


@testset "SearchDiscoveryCore" verbose = true begin
    include("test/search_discovery_core.j.l")
end

@testset "SD1" begin
    include("test/SD1.jl")
end

@testset "SD2" begin
    include("test/SD2.jl")
end