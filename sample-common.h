#ifndef SAMPLE_COMMON_H
#define SAMPLE_COMMON_H

#include <iostream>
#include <string>
#include <cstdint>

#include <sycl/sycl.hpp>
#if SYCL_LANGUAGE_VERSION < 202001
#error "Source is designed for SYCL 2020 or later"
#endif

/* Device selector that uses SYCL_PLATFORM and SYCL_DEVICE environment variable
 * to select a SYCL platform and device, falling back to the default in case of no override
 */
int env_device_selector(const sycl::device& dev)
{
	bool over = false;

	int p_num = 0, d_num = 0;
	const char *p_env = std::getenv("SYCL_PLATFORM");
	if (p_env) {
		over = true;
		p_num = std::atoi(p_env);
	}

	const char *d_env = std::getenv("SYCL_DEVICE");
	if (d_env) {
		over = true;
		d_num = std::atoi(d_env);
	}

	if (!over) return sycl::default_selector_v(dev);

	auto platforms = sycl::platform::get_platforms();
	const size_t np = platforms.size();

	if (p_num >= np)
		throw std::runtime_error("cannnot select SYCL platform #" + std::to_string(p_num) +
			" (only " + std::to_string(np) + " platforms present)");

	const auto& p = platforms[p_num];

	auto devices = p.get_devices();
	const size_t nd = devices.size();

	if (d_num >= nd)
		throw std::runtime_error("cannnot select SYCL device #" + std::to_string(d_num)
			+ " (only " + std::to_string(nd) + " devices available on platform #"
			+ std::to_string(p_num) + " "
			+ p.get_info<sycl::info::platform::name>()
			+ ")");

	const auto& d = devices[d_num];

	if (dev == d) return 100;
	return -1;
}

std::string event_status_name(sycl::info::event_command_status status)
{
	switch (status) {
	case sycl::info::event_command_status::submitted: return "submitted";
	case sycl::info::event_command_status::running: return "running";
	case sycl::info::event_command_status::complete: return "complete";
	default: return "unknown (" + std::to_string(int(status)) + ")";
	}
}

std::string event_status_name(sycl::event const& evt)
{
	return event_status_name(evt.get_info<sycl::info::event::command_execution_status>());
}

uint64_t event_runtime_ns(sycl::event const& evt)
{
	uint64_t start_time = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
	uint64_t end_time = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
	return end_time - start_time;
}

double event_runtime_ms(sycl::event const& evt)
{
	return event_runtime_ns(evt)*1.0e-6;
}


#endif
