using Distributions
using LinearAlgebra
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

@testset "SD" verbose = true begin
    include("tests_sd.jl")
end

@testset "WM" verbose = true begin
    include("tests_wm.jl")
end

@testset "NonUnicode" verbose = true begin
    include("tests_non_unicode.jl")
end

