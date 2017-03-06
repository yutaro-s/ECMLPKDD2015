

push!(LOAD_PATH,"../source/")
tests = ["synthetic", "train", "predict"]

println("Running tests:")
for i in tests
    test = "$i.jl"
    println("  $test")
    include(test)
end
println("Done!")
