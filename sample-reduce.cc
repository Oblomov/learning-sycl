#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include "sample-common.h"

struct vecinit
{
	using accessor = sycl::accessor<int, 1, sycl::access::mode::discard_write, sycl::access::target::device>;

	int nels;
	accessor vec;

	vecinit(sycl::handler& hand, sycl::buffer<int>& buf) :
		nels(buf.size()),
		vec(buf.get_access<sycl::access::mode::discard_write>(hand))
	{}

	void operator()(sycl::item<1> item) const {
		int i = item.get_id(0);
		if (i < nels) {
			vec[i] = 17;
		}
	}
};

template<typename T>
struct reduce
{
	using accessor_in   = sycl::accessor<T, 1, sycl::access::mode::read,       sycl::access::target::device>;
	using accessor_out  = sycl::accessor<T, 1, sycl::access::mode::write,      sycl::access::target::device>;
	using accessor_lmem = sycl::local_accessor<T, 1>;

	int nels;
	accessor_in in;
	accessor_out out;
	accessor_lmem lmem;

	reduce(sycl::handler& hand, sycl::buffer<int>& in_, sycl::buffer<int>& out_, int lws) :
		nels(in_.size()),
		in(in_.get_access<sycl::access::mode::read>(hand)),
		out(out_.get_access<sycl::access::mode::write>(hand)),
		lmem(accessor_lmem(lws, hand))
	{}

	reduce(sycl::handler& hand, sycl::buffer<int>& out_, int lws) :
		nels(out_.size()),
		in(out_.get_access<sycl::access::mode::read>(hand)),
		out(out_.get_access<sycl::access::mode::write>(hand)),
		lmem(accessor_lmem(lws, hand))
	{}

	void operator()(sycl::nd_item<1> item) const {
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
			item.barrier(sycl::access::fence_space::local_space);
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

	sycl::queue q(env_device_selector, {sycl::property::queue::enable_profiling()});

	auto q_dev = q.get_device();
	auto dev_cus = q_dev.get_info<sycl::info::device::max_compute_units>();

	std::cout << "Platform name: " << q_dev.get_platform().get_info<sycl::info::platform::name>() << std::endl;
	std::cout << "Device name: " << q_dev.get_info<sycl::info::device::name>() << std::endl;
	std::cout << "Device CUs: " << dev_cus << std::endl;

	/* allocate memory */
	int nwg = nwg_cu*dev_cus;
	auto d_vec = sycl::buffer<int>(nels);
	auto d_red = sycl::buffer<int>(nwg);

	/* enqueue vecinit */
	std::cout << "Submit init ..." << std::endl;
	auto init_evt = q.submit([&](sycl::handler& hand) {
		hand.parallel_for(sycl::range<1>(nels), vecinit(hand, d_vec));
	});

	std::cout << "Submit reduce (" + std::to_string(nwg) + "/" + std::to_string(lws) + ") ..." << std::endl;
	auto reduce_pass1_evt = q.submit([&](sycl::handler& hand) {
		hand.parallel_for(sycl::nd_range<1>(lws*nwg, lws), reduce<int>(hand, d_vec, d_red, lws));
	});

	auto reduce_pass2_evt = q.submit([&](sycl::handler& hand) {
		hand.parallel_for(sycl::nd_range<1>(lws, lws), reduce<int>(hand, d_red, lws));
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

	int sum = d_red.get_host_access(sycl::read_only)[0];
	const int expected = 17*nels;
	if (sum != expected)
		throw std::runtime_error("wrong sum: " + std::to_string(sum) + " != " + std::to_string(expected));

	std::cout << "OK." << std::endl;

	return 0;
} catch (sycl::exception& e) {
	std::cerr << e.what() << std::endl;
	throw e;
}
