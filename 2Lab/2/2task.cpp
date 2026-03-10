#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <fstream>

constexpr double PI = 3.14159265358979323846;
constexpr double a = -4.0;
constexpr double b = 4.0;
constexpr int nsteps = 40000000;

double func(double x)
{
    return std::exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int threads_num)
{
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel num_threads(threads_num)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumloc = 0.0;

        for (int i = lb; i <= ub; i++)
            sumloc += func(a + h * (i + 0.5));
        #pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}

double run_serial()
{
    auto start = std::chrono::high_resolution_clock::now();

    double res = integrate(func, a, b, nsteps);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    return diff.count();
}

std::vector<double> run_parallel(std::vector<int> threads)
{
    
    std::vector<double> times;


    for (int j = 0; j < threads.size(); j++){
        auto start = std::chrono::high_resolution_clock::now();

        double res = integrate_omp(func, a, b, nsteps, threads[j]);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;

        times.push_back(diff.count());
        
        std::cout << "Время выполнения для " << threads[j] << " потоков = " << diff.count() << std::endl;
    }

    
    return times;
}


int main(int argc, char **argv){

    std::vector<int> threads = {1, 2, 4, 7, 8, 16, 20, 40};

    double Ts = run_serial();
    
    std::cout << " Время выполнения последовательной = " << Ts << std::endl;


    // run_serial();
    std::vector<double> times = run_parallel(threads);

    std::vector<double> speedup;


    for (int i = 0; i < times.size(); i++){
        double su = Ts / times[i];
        speedup.push_back(su);
        
        std::cout << "Коэф. ускорения на " << threads[i] << " потоках = " << su << std::endl;
    }

    // Запись результатов в файл results.txt
    std::ofstream outfile("results.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < threads.size(); i++) {
            outfile << speedup[i] << std::endl;
        }
        outfile.close();
        std::cout << "\nРезультаты успешно записаны в файл results.txt" << std::endl;
    } else {
        std::cerr << "Ошибка: не удалось создать файл results.txt" << std::endl;
        return 1;
    }

    return 0;
}