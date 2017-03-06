using Base.Test


function gold2matrix(gold::Array)
  return sparse(gold[:,1], gold[:,2], 1)
end

##########################################################

function synthetic(d, n)
  Z = rand(Normal(0,1), d[1], n)
  R_x = rand(Uniform(-1,1), d[2], d[1])
  R_y = rand(Uniform(-1,1), d[3], d[1])
  X = R_x * Z
  Y = R_y * Z
  # add noise
   X += randn(size(X,1), size(X,2))
   Y += randn(size(Y,1), size(Y,2))
  gold = [collect(1:n) collect(1:n)]
  return X, Y, gold
end

#########################################################

function real(file_path)
  data = jldopen(file_path, "r") do file
    read(file, "X", "Y", "train", "test")
  end
  return data
end
