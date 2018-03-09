import numpy as np
import pyopencl as cl
import time

matrix1 = np.random.random((500,500)).astype(np.float32)
matrix2 = np.random.random((500,500)).astype(np.float32)

cl_context = cl.create_some_context()
queue = cl.CommandQueue(cl_context)
mf = cl.mem_flags
matrix1_in_gpu = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix1)
matrix2_in_gpu = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix2)
result_in_gpu = cl.Buffer(cl_context, mf.WRITE_ONLY, matrix1.nbytes)

cl_program = cl.Program(cl_context, """
__kernel void product(
    int size, __global const float *matrix1, __global const float *matrix2, __global float *result)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    result[i + size * j] = 0;
    for (int k = 0; k < size; k++)
    {
        result[i + size * j] += matrix1[k + size * i] * matrix2[j + size * k];
    }
}
""").build()


t0 = time.time()
cl_program.product(queue, matrix1.shape, None, np.int32(len(matrix1)), matrix1_in_gpu, matrix2_in_gpu, result_in_gpu)
t1 = time.time()
gpu_time = t1 - t0
print "GPU Time", gpu_time

result_in_numpy = np.empty_like(matrix1)
cl.enqueue_copy(queue, result_in_numpy, result_in_gpu)

t0 = time.time()
cpu_result = np.dot(matrix1, matrix2)
t1 = time.time()
cpu_time = t1 - t0
print "CPU Time", cpu_time

print "Norm of Difference", np.linalg.norm(result_in_numpy - cpu_result.T)

# GPU Time 0.00202608108521
# CPU Time 0.00995397567749
# Norm of Difference 0.0
