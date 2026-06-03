import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "notebooks"))
Pkg.precompile()

import Pluto
Pluto.run(; host="0.0.0.0", port=1234, auto_reload_from_file=true)
