import numpy as np
import pyopencl as cl
import time

vector1 = np.random.random(5000000).astype(np.float32)
vector2 = np.random.random(5000000).astype(np.float32)

cl_context = cl.create_some_context()
queue = cl.CommandQueue(cl_context)
mf = cl.mem_flags
vector1_in_gpu = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector1)
vector2_in_gpu = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector2)
result_in_gpu = cl.Buffer(cl_context, mf.WRITE_ONLY, vector1.nbytes)

cl_program = cl.Program(cl_context, """
__kernel void sum(
    __global const float *vector1, __global const float *vector2, __global float *result)
{
  int i = get_global_id(0);
  result[i] = vector1[i] + vector2[i];
}
""").build()

t0 = time.time()
cl_program.sum(queue, vector1.shape, None, vector1_in_gpu, vector2_in_gpu, result_in_gpu)
t1 = time.time()
gpu_time = t1 - t0
print "GPU Time", gpu_time

result_in_numpy = np.empty_like(vector1)
cl.enqueue_copy(queue, result_in_numpy, result_in_gpu)

t0 = time.time()
cpu_result = vector1 + vector2
t1 = time.time()
cpu_time = t1 - t0
print "CPU Time", cpu_time

print "Norm of Difference", np.linalg.norm(result_in_numpy - cpu_result)

# GPU Time 0.00202608108521
# CPU Time 0.00995397567749
# Norm of Difference 0.0
