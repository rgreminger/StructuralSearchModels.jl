using Distributions 
using StableRNGs
using StructuralSearchModels
using Test 

if Threads.nthreads() != 1 
    error("Tests are only reproducible if run on single thread.")
end

@testset "SearchDiscoveryCore" verbose = true begin
    include("tests_sdcore.jl")
end

@testset "SD1" begin
    include("tests_sd1.jl")
end

@testset "WM1" begin
    include("tests_wm1.jl")
end