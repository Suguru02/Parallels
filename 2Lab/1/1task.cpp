
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>

void matrix_vector_product(const std::vector<double>& a, const std::vector<double>& b, std::vector<double> c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        c[i] = 0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];

    }
}


double run_serial(int m, int n)
{

    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }
    for (int j = 0; j < n; j++)
        b[j] = j;

    auto start = std::chrono::high_resolution_clock::now();

    matrix_vector_product(a, b, c, m, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    return diff.count();
}


void matrix_vector_product_omp(const std::vector<double>& a, const std::vector<double>& b, std::vector<double> c, int m, int n, int threads_num)
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
            for (int j = 0; j < n; j++) c[i] += a[i * n + j] * b[j];
        }
    }
}


std::vector<double> run_parallel(int m, int n, std::vector<int> threads){
    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    
    std::vector<double> times;


    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }
    
    for (int j = 0; j < n; j++)
        b[j] = j;

    for (int j = 0; j < threads.size(); j++){

        auto start = std::chrono::high_resolution_clock::now();

        matrix_vector_product_omp(a, b, c, m, n, threads[j]);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;

        times.push_back(diff.count());
        
        std::cout << "Время выполнения для " << threads[j] << " потоков = " << diff.count() << std::endl;
    
    }

    return times;

}


int main(int argc, char **argv){

    int n = 20000;
    std::vector<int> threads = {1, 2, 4, 7, 8, 16, 20, 40};

    double Ts = run_serial(n, n);
    
    std::cout << " Время выполнения последовательной = " << Ts << std::endl;


    // run_serial();
    std::vector<double> times = run_parallel(n,n,threads);

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