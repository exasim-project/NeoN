function use_pointer(p::Ptr{Cvoid})
    arr = unsafe_wrap(Array, Ptr{Float64}(p), 10)
    arr[1] = 42.0
    return nothing
end

function test()
    println("hi")
    # arr = unsafe_wrap(Array, Ptr{Float64}(p), n)
    # return nothing
end

# function use_gpu_ptr(p::Ptr{Cvoid}, n::Int, iter::Int)
#     start = time()
#     ptr = reinterpret(CuPtr{Int32}, p)
#     A = unsafe_wrap(CuArray, ptr, n; own=false)
#     println("coping $n elements from a to b took $(time() - start)s")
#     s = time()
#     test(CUDABackend(), 32)(A;ndrange=n)
#     if iter == 1
#         println("First kernel took $(time() - s)s")
#     else
#         println("Subsequent kernel  took $(time() - s)s")

#     end

#     synchronize()

#     return nothing
# end