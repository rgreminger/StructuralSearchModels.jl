# This script serves the docs using LiveServer.jl for testing locally.

using Pkg; Pkg.activate("docs/."); Pkg.instantiate() ; Pkg.update()
using LiveServer
servedocs()
