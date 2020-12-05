#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <CL/sycl.hpp>

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
	return event_runtime_ns(evt)/1.0e6;
}

struct vecinit
{
	using accessor = cl::sycl::accessor<int, 1, cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>;

	int nels;
	// LESSON LEARNED: this _has_ to be an accessor, you can't just store the address of the first element you get
	// by dereferencing the accessor!
	accessor vec;

	vecinit(int nels_, accessor& vec_) :
		nels(nels_),
		vec(vec_)
	{}

	void operator()(cl::sycl::item<1> item) const {
		int i = item.get_id(0);
		if (i < nels) {
			vec[i] = nels - i;
		}
	}
};

template<typename Accessor>
void verify_init(int nels, Accessor vec) {
	for (int i = 0; i < nels; ++i) {
		int expected = nels - i;
		int computed = vec[i];
		if (computed != expected) {
			throw std::runtime_error("smooth failed at " + std::to_string(i) +
				": " + std::to_string(computed) + " != " + std::to_string(expected));
		}
	}
}


int main(int argc, char *argv[]) {

	if (argc != 2)
		throw std::invalid_argument("syntax: ./sample nels");

	const int nels = std::atoi(argv[1]);
	if (nels < 0)
		throw std::invalid_argument("nels < 0");

	/* allocate memory */
	auto d_vec = cl::sycl::buffer<int>(nels);

	/* init queue */

	cl::sycl::queue q({cl::sycl::property::queue::enable_profiling()});

	std::cout << "Host? " << (q.is_host() ? "true" : "false") << std::endl;
	std::cout << "Platform name: " << q.get_device().get_platform().get_info<cl::sycl::info::platform::name>() << std::endl;
	std::cout << "Device name: " << q.get_device().get_info<cl::sycl::info::device::name>() << std::endl;

	/* enqueue vecinit */
	const int gws = nels;

	std::cout << "Submit ..." << std::endl;
	auto init_evt = q.submit([&](cl::sycl::handler& hand) {
		auto vec = d_vec.get_access<cl::sycl::access::mode::discard_write>(hand);

		auto kernel = vecinit(nels, vec);
		hand.parallel_for(cl::sycl::range<1>(gws), kernel);
	});

	/* sync check */
	std::cout << "Event status: " << event_status_name(init_evt) << std::endl;

	std::cout << "Wait ..." << std::endl;

	init_evt.wait();

	std::cout << "Event status: " << event_status_name(init_evt) << std::endl;

	std::cout << "Queue wait ..." << std::endl;

	q.wait();

	std::cout << "Event status: " << event_status_name(init_evt) << std::endl;

	std::cout << "Runtime: " << event_runtime_ms(init_evt) << "ms" << std::endl;

	/* verify */

	verify_init(nels, d_vec.get_access<cl::sycl::access::mode::read>());

	std::cout << "OK." << std::endl;

	return 0;
}
