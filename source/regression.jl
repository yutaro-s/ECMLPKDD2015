
function train_x2y(X::Matrix, Y::Matrix, λ)
  return Y * X' * inv(X * X' + λ * eye(size(X,1), size(X,1)))
end

function train_y2x(X::Matrix, Y::Matrix, λ)
  return X * Y' * inv(Y * Y' + λ * eye(size(Y,1), size(Y,1))) 
end

function train_x2y_GD(X::Matrix, Y::Matrix, λ)
  M = eye(size(Y,1), size(X,1))
  α = 0.1
  iter = 0
  loss = Inf
  loss_old = 0
  loss_best = Inf
  M_best = M
  while (abs(loss - loss_old) > 0.001 || iter < 30) && iter < 1000
    M -= α * (M * (X * X' + λ * eye(size(X,1), size(X,1))) - Y * X')
    iter += 1
    loss_old = loss
    loss = sum((M * X - Y).^2) + λ * vecnorm(M)^2
    #loss = trace((M * X - Y)' * (M * X - Y) + λ * trace(M' * M))
    println("iter=", iter, ", loss=", loss)
    if loss < loss_best && iter > 20
      loss_best = loss
      M_best = M
    end
    α = loss > loss_old ? α * 0.5 : α * 1.001
  end
  @show loss_best
  return M_best
end

#########################################################################

"Calculating distance between source objects and target objects"
function distance(X::Matrix, Y::Matrix)
  #return sqrt(broadcast(+, diag(X' * X), broadcast(-, diag(Y' * Y), 2 * (X'*Y)')'))'
  #return sqrt(broadcast(+, diag(X' * X), broadcast(-, diag(Y' * Y)', 2 * X' * Y)))'
  return broadcast(-, diag(Y' * Y), 2 * (X' * Y)')
end


"findmin for matirx"
function find_nearest_neighbor(D::Matrix)
  R = []
  for i = 1:size(D,2)
    push!(R, findmin(D[:,i])[2])
  end
  return R
end

#########################################################################

"Projecting source objects into target space"
function predict_x2y(X::Matrix, Y::Matrix, M::Matrix)
  return find_nearest_neighbor(distance(M * X, Y))
end

"Projecting target objects into source space"
function predict_y2x(X::Matrix, Y::Matrix, W::Matrix)
  return find_nearest_neighbor(distance(X, W * Y))
end

function predict(X::Matrix, Y::Matrix)
  return find_nearest_neighbor(distance(X, Y))
end

