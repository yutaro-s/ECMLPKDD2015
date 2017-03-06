VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module ZSL
  
  export 
    train_x2y, predict_x2y, 
    train_y2x, predict_y2x,
    predict

  include("regression.jl")

end
