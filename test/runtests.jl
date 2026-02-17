using Distributions
using QuasiMonteCarlo
using Random
using StableRNGs
using StructuralSearchModels
using Test

if Threads.nthreads() != 1
    error("Tests are only reproducible if run on single thread.")
end

@testset "SearchDiscoveryCore" verbose = true begin
    include("tests_sdcore.jl")
end

@testset "SD" begin
    include("tests_sd.jl")
end

@testset "WM" begin
    include("tests_wm.jl")
end

