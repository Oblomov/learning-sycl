#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <sycl/sycl.hpp>

std::string event_status_name(sycl::info::event_command_status status)
{
	switch (status) {
	case sycl::info::event_command_status::submitted: return "submitted";
	case sycl::info::event_command_status::running: return "running";
	case sycl::info::event_command_status::complete: return "complete";
	default: return "unknown (" + std::to_string(int(status)) + ")";
	}
}

std::string event_status_name(sycl::event evt)
{
	return event_status_name(evt.get_info<sycl::info::event::command_execution_status>());
}

uint64_t event_runtime_ns(sycl::event evt)
{
	uint64_t start_time = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
	uint64_t end_time = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
	return end_time - start_time;
}

double event_runtime_ms(sycl::event evt)
{
	return event_runtime_ns(evt)/1.0e6;
}

struct vecinit
{
	using accessor = sycl::accessor<int, 1, sycl::access::mode::discard_write, sycl::access::target::device>;

	int nels;
	// LESSON LEARNED: this _has_ to be an accessor, you can't just store the address of the first element you get
	// by dereferencing the accessor!
	accessor vec;

	vecinit(int nels_, accessor& vec_) :
		nels(nels_),
		vec(vec_)
	{}

	void operator()(sycl::item<1> item) const {
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
	auto d_vec = sycl::buffer<int>(nels);

	/* init queue */

	sycl::queue q({sycl::property::queue::enable_profiling()});

	std::cout << "Platform name: " << q.get_device().get_platform().get_info<sycl::info::platform::name>() << std::endl;
	std::cout << "Device name: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

	/* enqueue vecinit */
	const int gws = nels;

	std::cout << "Submit ..." << std::endl;
	auto init_evt = q.submit([&](sycl::handler& hand) {
		vecinit::accessor vec = d_vec.get_access<sycl::access::mode::discard_write>(hand);

		auto kernel = vecinit(nels, vec);
		hand.parallel_for(sycl::range<1>(gws), kernel);
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

	verify_init(nels, d_vec.get_host_access(sycl::read_only));

	std::cout << "OK." << std::endl;

	return 0;
}
