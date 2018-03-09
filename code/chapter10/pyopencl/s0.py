import pyopencl as cl

print "OpenCL Platforms and Devices"
for platform in cl.get_platforms():
    print "Platform Name: ", platform.name
    print "Platform Vendor", platform.vendor
    print "Platform Version:", platform.version
    print "Platform Profile:", platform.profile
    for device in platform.get_devices():
        print "\n"
        print "\tDevice Name ", device.name
        print "\tDevice Type ",  cl.device_type.to_string(device.type)
        print "\tDevice Max Clock Speed ", "{0} Mhz".format(device.max_clock_frequency)
        print "\tDevice Compute Units ",  "{0}".format(device.max_compute_units)
        print "\tDevice Local Memory ",  "{0:.0f} KB".format(device.local_mem_size/1024.0)
        print "\tDevice Constant Memory ", "{0:.0f} KB".format(device.max_constant_buffer_size/1024.0)
        print "\tDevice Global Memory " "{0:.0f} GB".format(device.global_mem_size/(1024*1024*1024.0))

# OpenCL Platforms and Devices
# Platform Name:  Apple
# Platform Vendor Apple
# Platform Version: OpenCL 1.2 (Nov 18 2015 20:45:47)
# Platform Profile: FULL_PROFILE
#
#
# 	Device Name  Intel(R) Core(TM) i7-4770HQ CPU @ 2.20GHz
# 	Device Type  CPU
# 	Device Max Clock Speed  2200 Mhz
# 	Device Compute Units  8
# 	Device Local Memory  32 KB
# 	Device Constant Memory  64 KB
# 	Device Global Memory 16 GB
#
#
# 	Device Name  Iris Pro
# 	Device Type  GPU
# 	Device Max Clock Speed  1200 Mhz
# 	Device Compute Units  40
# 	Device Local Memory  64 KB
# 	Device Constant Memory  64 KB
# 	Device Global Memory 2 GB