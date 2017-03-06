VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module Generator
  
  using Distributions, JLD
  
  export gold2matrix,
         synthetic, real
  
  include("task.jl")

end
