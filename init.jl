using Pkg;
Pkg.activate(@__DIR__)
# using CUDA
function use_pointer(ptr::Ptr{Cvoid})
	#vec = unsafe_wrap(CuArray, ptr,10)
    # vec[1:3] = [1.0,2.0,3.0]
end
