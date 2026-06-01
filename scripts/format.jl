import Pkg
Pkg.activate(; temp=true)
Pkg.add("JuliaFormatter")

using JuliaFormatter
format(".")
