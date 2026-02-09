#include <iostream>
#include <vector>
#include <cmath>

// Определяем тип в зависимости от CMake-опции
#ifdef USE_DOUBLE
	#if USE_DOUBLE
		using type = double;
    	constexpr const char* type_name = "double";

    #else
		using type = float;
    	constexpr const char* type_name = "float";
	#endif

#else 
	using type = double;
	constexpr const char* type_name = "double";
#endif

int main() {
    
    int size = 10000000;

	type step = 2 * M_PI / size;

	std::vector<type> sin_seq;

	type sum = 0.0;
	type elem = 0.0;


	for (int i = 0; i < size; i++){
		sin_seq.push_back(std::sin( static_cast<type>(i) * step));
		elem = static_cast<type>(i) * step;

	}

	for (int i = 0; i < size; i++){
		sum += sin_seq[i];
	}

	std::cout << "Сумма :" << sum << std::endl;
	std::cout << type_name << std::endl;
	std::cout << "last elem  " << elem << std::endl;

    return 0;
}