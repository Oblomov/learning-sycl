#ifndef SAMPLE_COMMON_H
#define SAMPLE_COMMON_H

#include <iostream>
#include <string>

#include <CL/sycl.hpp>
#if SYCL_LANGUAGE_VERSION >= 202001
#include <sycl/backend/opencl.hpp>
#include <sycl/backend/cuda.hpp>
#endif

/* Device selector that uses OCL_PLATFORM and OCL_DEVICE environment variable
 * to select an OpenCL platform and device. If none has been requested,
 */
class env_device_selector : public cl::sycl::device_selector
{
private:
	cl::sycl::default_selector def;
	bool over = false;
	cl_platform_id p = NULL;
	cl_device_id d = NULL;
public:
	env_device_selector()
	{
		int p_num = 0, d_num = 0;
		const char *p_env = std::getenv("OCL_PLATFORM");
		if (p_env) {
			over = true;
			p_num = std::atoi(p_env);
		}

		const char *d_env = std::getenv("OCL_DEVICE");
		if (d_env) {
			over = true;
			d_num = std::atoi(d_env);
		}

		if (!over) return;

		// host can be selected by OCL_PLATFORM=-1
		if (p_num < 0) return;

		cl_uint n;
		cl_int err = clGetPlatformIDs(0, NULL, &n);
		if (err != CL_SUCCESS || p_num >= n)
			throw std::runtime_error("cannnot select OpenCL platform #" + std::to_string(p_num) +
				" (err: " + std::to_string(err) + ")");

		cl_platform_id *plats = new cl_platform_id[n];
		err = clGetPlatformIDs(n, plats, NULL);
		if (err != CL_SUCCESS) {
			delete[] plats;
			throw std::runtime_error("failed to get OpenCL platform #" + std::to_string(p_num) +
				" (err: " + std::to_string(err) + ")");
		}
		p = plats[p_num];
		delete[] plats;

		err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &n);
		if (err != CL_SUCCESS || d_num >= n)
			throw std::runtime_error("cannnot select OpenCL device #" + std::to_string(d_num) +
				" (err: " + std::to_string(err) + ")");
		cl_device_id *devs = new cl_device_id[n];
		err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, n, devs, NULL);
		if (err != CL_SUCCESS) {
			delete[] devs;
			throw std::runtime_error("failed to get OpenCL device #" + std::to_string(d_num) +
				" (err: " + std::to_string(err) + ")");
		}
		d = devs[d_num];
		delete[] devs;
	}
	env_device_selector(const env_device_selector &rhs) :
		def(rhs.def),
		over(rhs.over),
		p(rhs.p),
		d(rhs.d)
	{}

	env_device_selector &operator=(const env_device_selector &rhs)
	{
		over = rhs.over;
		p = rhs.p;
		d = rhs.d;
		return *this;
	}

	virtual ~env_device_selector() {}

	// device score
	int operator()(const cl::sycl::device& device) const override
	{
		// if no override was requested, let the default selector do the ranking
		if (!over) return def(device);
		// skip host, unless p == NULL
		if (device.is_host()) {
			if (p) return -1;
			else return 1000;
#if SYCL_LANGUAGE_VERSION < 202001
		} else if (device.get() == d) return 100;
#else
		} else if (sycl::get_native<sycl::backend::opencl>(device) == d) return 100;
#endif
		return -1;
	}
};

std::string event_status_name(cl::sycl::info::event_command_status status)
{
	switch (status) {
	case cl::sycl::info::event_command_status::submitted: return "submitted";
	case cl::sycl::info::event_command_status::running: return "running";
	case cl::sycl::info::event_command_status::complete: return "complete";
	default: return "unknown (" + std::to_string(int(status)) + ")";
	}
}

std::string event_status_name(cl::sycl::event evt)
{
	return event_status_name(evt.get_info<cl::sycl::info::event::command_execution_status>());
}

cl_ulong event_runtime_ns(cl::sycl::event evt)
{
	cl_ulong start_time = evt.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
	cl_ulong end_time = evt.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
	return end_time - start_time;
}

double event_runtime_ms(cl::sycl::event evt)
{
	return event_runtime_ns(evt)*1.0e-6;
}


#endif
