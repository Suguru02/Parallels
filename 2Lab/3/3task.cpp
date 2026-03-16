#include <iostream>
#include <omp.h>
#include <cmath>
#include <string>
#include <chrono>
#include <vector>
#include <fstream>


#define MAX_ITERATION 10000
#define EPSILON 1e-5
#define N 1000


void solve_linear_system_v1(const std::vector<double>& A, std::vector<double> x, std::vector<double>& b, int threads_num)
{
    double tau = 0.01;
    {
        for (int itr = 0; itr < MAX_ITERATION; itr++) {
            std::vector<double> Ax(N, 0.0);
            #pragma omp parallel for num_threads(threads_num)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    Ax[i] += A[i * N + j] * x[j];
                }
            }

            #pragma omp parallel for num_threads(threads_num)
            for (int i = 0; i < N; i++) {
                x[i] = x[i] - tau * (Ax[i] - b[i]);
            }

    
            double error_norm_sq = 0.0;
            #pragma omp parallel for reduction(+:error_norm_sq) num_threads(threads_num)
            for (int i = 0; i < N; i++) {
                double diff = tau * (Ax[i] - b[i]);
                error_norm_sq += diff * diff;
            }
            
            if (sqrt(error_norm_sq) < EPSILON) {
                break;
            }
        }  
    }
}

void solve_linear_system_v2(const std::vector<double>& A, std::vector<double> x, std::vector<double>& b, int threads_num)
{
    double tau = 0.01;
    #pragma omp parallel num_threads(threads_num)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = N / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (N - 1) : (lb + items_per_thread - 1);

        for (int itr = 0; itr < MAX_ITERATION; itr++) {
            std::vector<double> Ax(N, 0.0);
            for (int i = lb; i < ub; i++) {
                for (int j = 0; j < N; j++) {
                    Ax[i] += A[i * N + j] * x[j];
                }
            }

            #pragma omp barrier
            for (int i = lb; i < ub; i++) {
                x[i] = x[i] - tau * (Ax[i] - b[i]);
            }

            #pragma omp barrier

            double error_norm_sq = 0.0;
            #pragma omp for reduction(+:error_norm_sq)
            for (int i = 0; i < N; i++) {
                double diff = tau * (Ax[i] - b[i]);
                error_norm_sq += diff * diff;
            }
            #pragma omp barrier
            
            #pragma omp single
            if (sqrt(error_norm_sq) < EPSILON) {
                break;
            }

            #pragma omp barrier
        }  
    }
}

void solve_linear_system_v3(const std::vector<double>& A, std::vector<double>& x, std::vector<double>& b, int threads_num, std::string type, int chunk_size)
{
    bool converged = false;
    double tau = 0.01;
    #pragma omp parallel num_threads(threads_num)
    {   
        std::vector<double> Ax(N, 0.0);
        for (int itr = 0; itr < MAX_ITERATION && !converged; itr++) {
            
            // === Этап 1: Вычисление Ax ===
            if (type == "static") {
                #pragma omp for schedule(static, chunk_size)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            } else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            } else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            }
            #pragma omp barrier


            if (type == "static") {
                #pragma omp for schedule(static, chunk_size)
                for (int i = 0; i < N; i++)
                    x[i] = x[i] - tau * (Ax[i] - b[i]);
            } else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size)
                for (int i = 0; i < N; i++)
                    x[i] = x[i] - tau * (Ax[i] - b[i]);
            } else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size)
                for (int i = 0; i < N; i++)
                    x[i] = x[i] - tau * (Ax[i] - b[i]);
            }
            #pragma omp barrier
    

            double error_norm_sq = 0.0;
            if (type == "static") {
                #pragma omp for schedule(static, chunk_size) reduction(+:error_norm_sq)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    error_norm_sq += diff * diff;
                }
            } else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size) reduction(+:error_norm_sq)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    error_norm_sq += diff * diff;
                }
            } else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size) reduction(+:error_norm_sq)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    error_norm_sq += diff * diff;
                }
            }
            
            #pragma omp single
            {
                if (sqrt(error_norm_sq) < EPSILON) {
                    converged = true;
                }
            }

            #pragma omp barrier 

            if (converged) {
                break;
            }
        }  
    }
}


std::vector<double> run_parallel(std::vector<int>& threads, const std::vector<double> A, std::vector<double> x, std::vector<double>& b)
{
    
    std::vector<double> times;


    for (int i = 0; i < threads.size(); i++){
        auto start = std::chrono::high_resolution_clock::now();

        // solve_linear_system_v1(A, x, b, threads[i]);
        solve_linear_system_v2(A, x, b, threads[i]);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;

        times.push_back(diff.count());
        
        std::cout << "Время выполнения для " << threads[i] << " потоков = " << diff.count() << std::endl;
    }

    
    return times;
}

std::vector<double> run_parallel_schedule(std::vector<int>& chunks, const std::vector<double> A, std::vector<double> x, std::vector<double>& b)
{
    
    std::vector<double> times;


    for (int i = 0; i < chunks.size(); i++){
        auto start = std::chrono::high_resolution_clock::now();

        solve_linear_system_v3(A, x, b, 16, "guided", chunks[i]);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;

        times.push_back(diff.count());
        
        std::cout << "Время выполнения для " << chunks[i] << " чанков = " << diff.count() << std::endl;
    }

    
    return times;
}


int main(){

    std::vector<double> A(N * N), b(N);
    std::vector<double> x(N, 0.0);

    std::vector<int> threads = {1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<int> chunks = {8, 16, 32, 64};


// Инициализация 
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (i==j){
                A[i * N + j] = 2.0;
            }
            else {
                A[i * N + j] = 1.0;
            }
        }
    }

    for (int i = 0; i < N; i++){
        b[i] = static_cast<double>(N+1);
    }

    // std::vector<double>times = run_parallel(threads, A, x, b);
    std::vector<double>times = run_parallel_schedule(chunks, A, x, b);

    std::vector<double> speedup;


    // Вариант подсчета и вывода для ускорения

    // for (int i = 1; i < times.size(); i++){
    //     double su = times[0] / times[i];
    //     speedup.push_back(su);
        
    //     std::cout << "Коэф. ускорения на " << threads[i] << " потоках = " << su << std::endl;
    // }

    // // Запись результатов в файл results.txt
    // std::ofstream outfile("results.txt");
    // if (outfile.is_open()) {
    //     for (int i = 0; i < threads.size(); i++) {
    //         outfile << speedup[i] << std::endl;
    //     }
    //     outfile.close();
    //     std::cout << "\nРезультаты успешно записаны в файл results.txt" << std::endl;
    // } else {
    //     std::cerr << "Ошибка: не удалось создать файл results.txt" << std::endl;
    //     return 1;
    // }

    
    //  Вариант вывода для замера времени с разными чанками

     std::ofstream outfile("results.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < chunks.size(); i++) {
            outfile << times[i] << std::endl;
        }
        outfile.close();
        std::cout << "\nРезультаты успешно записаны в файл results.txt" << std::endl;
    } else {
        std::cerr << "Ошибка: не удалось создать файл results.txt" << std::endl;
        return 1;
    }


    return 0;
}
