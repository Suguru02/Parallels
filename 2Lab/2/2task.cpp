#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <numeric>

constexpr double PI = 3.14159265358979323846;
constexpr double a = -4.0;
constexpr double b = 4.0;
const std::vector<int> NSTEPS_SIZES = {40000000, 80000000};
const std::vector<int> THREADS = {1, 2, 4, 7, 8, 16, 20, 40};
const int NUM_RUNS = 100;

double func(double x){
    return std::exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n){
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int threads_num){
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

std::vector<double> run_serial(const std::vector<int>& sizes, int runs = NUM_RUNS){
    std::vector<double> avg_times;
    
    for (int n : sizes) {
        std::vector<double> run_times;
        run_times.reserve(runs);
        
        for (int run = 0; run < runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            double res = integrate(func, a, b, n);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> diff = end - start;
            run_times.push_back(diff.count());
        }
        
        double avg = std::accumulate(run_times.begin(), run_times.end(), 0.0) / runs;
        avg_times.push_back(avg);
        std::cout << "Среднее время (послед.) для n=" << n 
                  << " за " << runs << " запусков: " << avg << " с" << std::endl;
    }
    return avg_times;
}

std::vector<std::vector<double>> run_parallel(const std::vector<int>& sizes, 
                                               const std::vector<int>& threads, 
                                               int runs = NUM_RUNS){
    std::vector<std::vector<double>> all_times;
    
    for (int n : sizes) {
        std::vector<double> avg_times_for_threads;
        
        for (int t : threads) {
            std::vector<double> run_times;
            run_times.reserve(runs);
            
            for (int run = 0; run < runs; run++) {
                auto start = std::chrono::high_resolution_clock::now();
                double res = integrate_omp(func, a, b, n, t);
                auto end = std::chrono::high_resolution_clock::now();
                
                std::chrono::duration<double> diff = end - start;
                run_times.push_back(diff.count());
            }
            
            double avg = std::accumulate(run_times.begin(), run_times.end(), 0.0) / runs;
            avg_times_for_threads.push_back(avg);
            std::cout << "  n=" << n << " | Потоков: " << t 
                      << " | Среднее время: " << avg << " с" << std::endl;
        }
        all_times.push_back(avg_times_for_threads);
    }
    return all_times;
}

int main(int argc, char **argv){
    std::cout << "=== Запуск последовательной версии ===" << std::endl;
    std::vector<double> serial_times = run_serial(NSTEPS_SIZES, NUM_RUNS);

    std::cout << "\n=== Запуск параллельной версии ===" << std::endl;
    auto parallel_times = run_parallel(NSTEPS_SIZES, THREADS, NUM_RUNS);

    std::ofstream outfile("results.txt");
    if (!outfile.is_open()) {
        std::cerr << "Ошибка: не удалось создать файл results.txt" << std::endl;
        return 1;
    }

    outfile << "NSteps\tThreads\tSerial_Time\tParallel_Time\tSpeedup\n";
    
    for (int size_idx = 0; size_idx < NSTEPS_SIZES.size(); size_idx++) {
        int n = NSTEPS_SIZES[size_idx];
        double serial_time = serial_times[size_idx];
        
        std::cout << "\n=== Результаты для n=" << n << " ===" << std::endl;
        
        for (int t_idx = 0; t_idx < THREADS.size(); t_idx++) {
            double parallel_time = parallel_times[size_idx][t_idx];
            double speedup = (parallel_time > 1e-9) ? serial_time / parallel_time : 0;
            
            outfile << n << "\t" << THREADS[t_idx] << "\t" 
                    << serial_time << "\t" << parallel_time << "\t" << speedup << "\n";
            
            std::cout << "Потоков: " << THREADS[t_idx] 
                      << " | Ускорение: " << speedup << std::endl;
        }
    }
    
    outfile.close();

    return 0;
}
