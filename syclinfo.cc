#include <iostream>
#include <sycl/sycl.hpp>

// AdaptiveCpp doesn't define ostream for the backend_id unless this file is included
#ifdef SYCL_IMPLEMENTATION_ACPP
#include <hipSYCL/runtime/serialization/serialization.hpp>
#endif

int main(int argc, char *argv[])
{

	auto platforms = sycl::platform::get_platforms();
	const size_t np = platforms.size();

	for (size_t i = 0; i < np; ++i) {
		const auto& p = platforms[i];
		std::cout << "Platform #" << i
#if SYCL_LANGUAGE_VERSION >= 202001
			<< " [" << p.get_backend() <<"]\t"
#endif
			<< ": " << p.get_info<sycl::info::platform::name>() << std::endl;

		auto devices = p.get_devices();
		const size_t nd = devices.size();
		for (size_t j = 0; j < nd; ++j) {
			const auto& d = devices[j];
			const std::string connector(j == nd-1 ? "`" : "+");
			std::cout << " " << connector << "-- Device #" << j << ": " << d.get_info<sycl::info::device::name>() << std::endl;
		}
	}
}
