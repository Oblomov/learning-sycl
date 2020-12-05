#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include "sample-common.h"

struct vecinit
{
	using accessor = cl::sycl::accessor<int, 1, cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>;

	int nels;
	accessor vec;

	vecinit(cl::sycl::handler& hand, cl::sycl::buffer<int>& buf) :
		nels(buf.get_count()),
		vec(buf.get_access<cl::sycl::access::mode::discard_write>(hand))
	{}

	void operator()(cl::sycl::item<1> item) const {
		int i = item.get_id(0);
		if (i < nels) {
			vec[i] = 17;
		}
	}
};

template<typename T>
struct reduce
{
	using accessor_in   = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,       cl::sycl::access::target::global_buffer>;
	using accessor_out  = cl::sycl::accessor<T, 1, cl::sycl::access::mode::write,      cl::sycl::access::target::global_buffer>;
	using accessor_lmem = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

	int nels;
	accessor_in in;
	accessor_out out;
	accessor_lmem lmem;

	reduce(cl::sycl::handler& hand, cl::sycl::buffer<int>& in_, cl::sycl::buffer<int>& out_, int lws) :
		nels(in_.get_count()),
		in(in_.get_access<cl::sycl::access::mode::read>(hand)),
		out(out_.get_access<cl::sycl::access::mode::write>(hand)),
		lmem(accessor_lmem(lws, hand))
	{}

	reduce(cl::sycl::handler& hand, cl::sycl::buffer<int>& out_, int lws) :
		nels(out_.get_count()),
		in(out_.get_access<cl::sycl::access::mode::read>(hand)),
		out(out_.get_access<cl::sycl::access::mode::write>(hand)),
		lmem(accessor_lmem(lws, hand))
	{}

	void operator()(cl::sycl::nd_item<1> item) const {
		int gi = item.get_global_id(0);
		T acc = T(0);
		while (gi < nels) {
			acc += in[gi];
			gi += item.get_global_range(0);
		}
		const int li = item.get_local_id(0);
		lmem[li] = acc;

		int active = item.get_local_range(0)/2;
		while (active > 0) {
			item.barrier(cl::sycl::access::fence_space::local_space);
			if (li < active) {
				acc += lmem[li + active];
				lmem[li] = acc;
			}
			active /= 2;
		}
		if (li == 0) {
			out[item.get_group(0)] = acc;
		}
	}
};

int main(int argc, char *argv[]) try {

	if (argc != 4)
		throw std::invalid_argument("syntax: ./sample nels lws nwg_cu");

	const int nels = std::atoi(argv[1]);
	if (nels <= 0) throw std::invalid_argument("nels <= 0");
	const int lws = std::atoi(argv[2]);
	if (lws <= 0) throw std::invalid_argument("lws <= 0");
	const int nwg_cu = std::atoi(argv[3]);
	if (nwg_cu <= 0) throw std::invalid_argument("nwg_cu <= 0");

	/* init queue */

	env_device_selector sel;
	cl::sycl::queue q(sel, {cl::sycl::property::queue::enable_profiling()});

	auto q_dev = q.get_device();
	auto dev_cus = q_dev.get_info<cl::sycl::info::device::max_compute_units>();

	std::cout << "Host? " << (q.is_host() ? "true" : "false") << std::endl;
	std::cout << "Platform name: " << q_dev.get_platform().get_info<cl::sycl::info::platform::name>() << std::endl;
	std::cout << "Device name: " << q_dev.get_info<cl::sycl::info::device::name>() << std::endl;
	std::cout << "Device CUs: " << dev_cus << std::endl;

	/* allocate memory */
	int nwg = nwg_cu*dev_cus;
	auto d_vec = cl::sycl::buffer<int>(nels);
	auto d_red = cl::sycl::buffer<int>(nwg);

	/* enqueue vecinit */
	std::cout << "Submit init ..." << std::endl;
	auto init_evt = q.submit([&](cl::sycl::handler& hand) {
		hand.parallel_for(cl::sycl::range<1>(nels), vecinit(hand, d_vec));
	});

	std::cout << "Submit reduce (" + std::to_string(nwg) + "/" + std::to_string(lws) + ") ..." << std::endl;
	auto reduce_pass1_evt = q.submit([&](cl::sycl::handler& hand) {
		hand.parallel_for(cl::sycl::nd_range<1>(lws*nwg, lws), reduce<int>(hand, d_vec, d_red, lws));
	});

	auto reduce_pass2_evt = q.submit([&](cl::sycl::handler& hand) {
		hand.parallel_for(cl::sycl::nd_range<1>(lws, lws), reduce<int>(hand, d_red, lws));
	});


	q.wait();

	const double init_runtime = event_runtime_ms(init_evt);
	const double reduce_pass1_runtime = event_runtime_ms(reduce_pass1_evt);
	const double reduce_pass2_runtime = event_runtime_ms(reduce_pass2_evt);

	std::cout << "init runtime: " << init_runtime  << "ms" << std::endl;
	std::cout << "init bw: " << nels*sizeof(int)/init_runtime/1.0e6  << "GB/s" << std::endl;

	std::cout << "reduce pass1 runtime: " << reduce_pass1_runtime  << "ms" << std::endl;
	std::cout << "reduce pass2 runtime: " << reduce_pass2_runtime  << "ms" << std::endl;

	/* verify */

	int sum = d_red.get_access<cl::sycl::access::mode::read>()[0];
	const int expected = 17*nels;
	if (sum != expected)
		throw std::runtime_error("wrong sum: " + std::to_string(sum) + " != " + std::to_string(expected));

	std::cout << "OK." << std::endl;

	return 0;
} catch (cl::sycl::exception& e) {
	std::cerr << e.what() << std::endl;
	throw e;
}
