#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <numeric>

void matrix_vector_product(const std::vector<double>& a, const std::vector<double>& b, 
                           std::vector<double>& c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        c[i] = 0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_omp(const std::vector<double>& a, const std::vector<double>& b, 
                               std::vector<double>& c, int m, int n, int threads_num)
{
    #pragma omp parallel num_threads(threads_num)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        
        for (int i = lb; i <= ub; i++) {
            c[i] = 0;
            for (int j = 0; j < n; j++) 
                c[i] += a[i * n + j] * b[j];
        }
    }
}

std::vector<double> run_serial(const std::vector<int>& sizes, int runs = 1)
{
    std::vector<double> avg_times;
    
    for (int m : sizes) {
        std::vector<double> a(m * m);
        std::vector<double> b(m);
        std::vector<double> c(m);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++)
                a[i * m + j] = i + j;
        }
        for (int j = 0; j < m; j++)
            b[j] = j;

        std::vector<double> run_times;
        run_times.reserve(runs);
        
        for (int run = 0; run < runs; run++) {
            std::fill(c.begin(), c.end(), 0.0);
            
            auto start = std::chrono::high_resolution_clock::now();
            matrix_vector_product(a, b, c, m, m);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> diff = end - start;
            run_times.push_back(diff.count());
        }
        
        double avg = std::accumulate(run_times.begin(), run_times.end(), 0.0) / runs;
        avg_times.push_back(avg);
        std::cout << "Среднее время (послед.) для " << m << "x" << m 
                  << " за " << runs << " запусков: " << avg << " с" << std::endl;
    }
    return avg_times;
}

std::vector<std::vector<double>> run_parallel(const std::vector<int>& sizes, 
                                               const std::vector<int>& threads, 
                                               int runs = 100)
{
    std::vector<std::vector<double>> all_times;
    
    for (int m : sizes) {
        std::vector<double> a(m * m);
        std::vector<double> b(m);
        std::vector<double> c(m);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                a[i * m + j] = i + j;
        for (int j = 0; j < m; j++)
            b[j] = j;

        std::vector<double> avg_times_for_sizes;
        
        for (int t : threads) {
            std::vector<double> run_times;
            run_times.reserve(runs);
            
            for (int run = 0; run < runs; run++) {
                std::fill(c.begin(), c.end(), 0.0);
                
                auto start = std::chrono::high_resolution_clock::now();
                matrix_vector_product_omp(a, b, c, m, m, t);
                auto end = std::chrono::high_resolution_clock::now();
                
                std::chrono::duration<double> diff = end - start;
                run_times.push_back(diff.count());
            }
            
            double avg = std::accumulate(run_times.begin(), run_times.end(), 0.0) / runs;
            avg_times_for_sizes.push_back(avg);
            std::cout << "  Потоков: " << t << " | Среднее время: " << avg << " с" << std::endl;
        }
        all_times.push_back(avg_times_for_sizes);
    }
    return all_times;
}

int main(int argc, char **argv) {
    std::vector<int> sizes = {20000, 40000};
    std::vector<int> threads = {1, 2, 4, 7, 8, 16, 20, 40};
    const int NUM_RUNS = 100;

    std::cout << "=== Запуск последовательной версии ===" << std::endl;
    std::vector<double> serial_times = run_serial(sizes);

    std::cout << "\n=== Запуск параллельной версии ===" << std::endl;
    auto parallel_times = run_parallel(sizes, threads, NUM_RUNS);

    std::ofstream outfile("results.txt");
    if (!outfile.is_open()) {
        std::cerr << "Ошибка: не удалось создать файл results.txt" << std::endl;
        return 1;
    }

    outfile << "# Size\tThreads\tSerial_Time\tParallel_Time\tSpeedup\n";
    
    for (int size_idx = 0; size_idx < sizes.size(); size_idx++) {
        int m = sizes[size_idx];
        double serial_time = serial_times[size_idx];
        
        std::cout << "\n=== Результаты для матрицы " << m << "x" << m << " ===" << std::endl;
        
        for (int t_idx = 0; t_idx < threads.size(); t_idx++) {
            double parallel_time = parallel_times[size_idx][t_idx];
            double speedup = (parallel_time > 0) ? serial_time / parallel_time : 0;
            
            outfile << m << "\t" << threads[t_idx] << "\t" 
                    << serial_time << "\t" << parallel_time << "\t" << speedup << "\n";
            
            std::cout << "Потоков: " << threads[t_idx] 
                      << " | Ускорение: " << speedup << std::endl;
        }
    }
    
    outfile.close();
    return 0;
}
