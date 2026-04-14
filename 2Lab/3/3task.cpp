#include <iostream>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <inttypes.h> 
#include <cmath>
#include <string>
#include <chrono>
#include <fstream>
#include <numeric>
#include <cstdlib>

#define EPS 1e-5
#define MAX_ITER 10000
#define N 1000
#define NUM_RUNS 100

using TYPE = std::vector<double>;


void init(TYPE& A, TYPE& b){
    A.resize(N * N);
    b.resize(N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            if(i == j) 
                A[i * N + j] = 2.0;
            else 
                A[i * N + j] = 1.0;
        }
    } 

    for (int j = 0; j < N; j++)
        b[j] = static_cast<double>(N+1);
}

void solve_linear_system_v1(const TYPE& A, TYPE& x, const TYPE& b, int num_threads){   
    double tau = 0.01;
    TYPE Ax(N);
    for (int iter = 0; iter < MAX_ITER; iter++) {
        std::fill(Ax.begin(), Ax.end(), 0.0);
        
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                Ax[i] += A[i * N + j] * x[j];
            }
        }

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            x[i] = x[i] - tau * (Ax[i] - b[i]);
        }

        double global_diff_norm = 0.0;
        #pragma omp parallel for reduction(+:global_diff_norm) num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            double diff = tau * (Ax[i] - b[i]);
            global_diff_norm += diff * diff;
        }
        
        if (std::sqrt(global_diff_norm) < EPS) {
            break;
        }
    }
}

void solve_linear_system_v2(const TYPE& A, TYPE& x, const TYPE& b, int threads_num){
    double tau = 0.01;
    #pragma omp parallel num_threads(threads_num)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = N / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? N : (lb + items_per_thread);

        for (int itr = 0; itr < MAX_ITER; itr++) {
            TYPE Ax(N, 0.0);
            
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
            {
                if (std::sqrt(error_norm_sq) < EPS) {
                    itr = MAX_ITER;
                }
            }
            #pragma omp barrier
        }  
    }
}

void solve_linear_system_v3(const TYPE& A, TYPE& x, const TYPE& b, 
                           int num_threads, std::string type, int chunk_size){   
    bool converged = false;
    double tau = 0.01;
    #pragma omp parallel num_threads(num_threads)
    {   
        TYPE Ax(N);
        for (int iter = 0; iter < MAX_ITER && !converged; iter++) {
            std::fill(Ax.begin(), Ax.end(), 0.0);
            
            if (type == "static") { 
                #pragma omp for schedule(static, chunk_size)
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            } 
            else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size)
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            } 
            else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size)
                for (int i = 0; i < N; i++) {
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
    
            double global_diff_norm = 0.0;
            if (type == "static") {
                #pragma omp for schedule(static, chunk_size) reduction(+:global_diff_norm)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    global_diff_norm += diff * diff;
                }
            } else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size) reduction(+:global_diff_norm)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    global_diff_norm += diff * diff;
                }
            } else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size) reduction(+:global_diff_norm)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    global_diff_norm += diff * diff;
                }
            }
            
            #pragma omp single
            {
                if (std::sqrt(global_diff_norm) < EPS) {
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


double run_parallel_standard(const TYPE& A, TYPE& x, const TYPE& b, 
                            int num_threads, 
                            void (*solver)(const TYPE&, TYPE&, const TYPE&, int)){   
    auto start = std::chrono::high_resolution_clock::now();
    solver(A, x, b, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}


double run_parallel_schedule(const TYPE& A, TYPE& x, const TYPE& b, 
                            int num_threads, std::string type, int chunk_size){   
    auto start = std::chrono::high_resolution_clock::now();
    solve_linear_system_v3(A, x, b, num_threads, type, chunk_size);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}


int main(int argc, char** argv){
    int version = std::atoi(argv[1]);

    TYPE A, b, x(N, 0.0);
    init(A, b);
    
    std::vector<int> threads = {1, 2, 4, 7, 8, 16, 20, 40};
    const std::vector<std::string> schedules = {"static", "dynamic", "guided"};
    const std::vector<int> chunks = {1, 4, 8, 16, 32, 64, 128, 256};
    
    const int fixed_threads_v3 = 8;

    printf("Linear equation (N=%d)\n", N);
    printf("Selected version: %d\n\n", version);

    printf("=== Замер базового времени (1 поток) ===\n");
    std::vector<double> serial_runs;
    serial_runs.reserve(NUM_RUNS);
    for (int run = 0; run < NUM_RUNS; run++) {
        std::fill(x.begin(), x.end(), 0.0);
        double t = run_parallel_standard(A, x, b, 1, solve_linear_system_v1);
        serial_runs.push_back(t);
    }
    double serial_time = std::accumulate(serial_runs.begin(), serial_runs.end(), 0.0) / NUM_RUNS;
    printf("Базовое время (1 поток): %.6f с\n\n", serial_time);

    std::string result_file = "results" + std::to_string(version) + ".txt";
    FILE* file = fopen(result_file.c_str(), "w");
    
    if (!file) {
        std::cerr << "Ошибка: не удалось создать файл " << result_file << std::endl;
        return 1;
    }

    if (version == 1) {
        fprintf(file, "Average Results after %d operations:\n", NUM_RUNS);
        fprintf(file, "%10s %15s %15s %15s\n", "Threads(I)", "Time(sec)(T_i)", "Gain(S)", "Gain Modified(S)");
        
        for (int t : threads) {
            double total_time = 0.0;
            for (int r = 0; r < NUM_RUNS; r++) {
                std::fill(x.begin(), x.end(), 0.0);
                total_time += run_parallel_standard(A, x, b, t, solve_linear_system_v1);
            }
            double avg_time = total_time / NUM_RUNS;
            double gain = serial_time / avg_time;
            double gain_modified = gain / t;
            
            fprintf(file, "%10d %15.6f %15.6f %15.6f\n", t, avg_time, gain, gain_modified);
            printf("Threads %2d: %.4f sec | Speedup: %.2f\n", t, avg_time, gain);
        }
        
    } else if (version == 2) {
        fprintf(file, "%10s %15s %15s %15s\n", "Threads(I)", "Time(sec)(T_i)", "Gain(S)", "Gain Modified(S)");
        
        for (int t : threads) {
            double total_time = 0.0;
            for (int r = 0; r < NUM_RUNS; r++) {
                std::fill(x.begin(), x.end(), 0.0);
                total_time += run_parallel_standard(A, x, b, t, solve_linear_system_v2);
            }
            double avg_time = total_time / NUM_RUNS;
            double gain = serial_time / avg_time;
            double gain_modified = gain / t;
            
            fprintf(file, "%10d %15.6f %15.6f %15.6f\n", t, avg_time, gain, gain_modified);
            printf("Threads %2d: %.4f sec | Speedup: %.2f\n", t, avg_time, gain);
        }
        
    } else if (version == 3) {
        fprintf(file, "%-10s %15s %15s\n", "Type schedule", "Chunks", "Time");
        
        for (const std::string& schedule : schedules) {
            for (int chunk : chunks) {
                double total_time = 0.0;
                
                for (int r = 0; r < NUM_RUNS; r++) {
                    std::fill(x.begin(), x.end(), 0.0);
                    total_time += run_parallel_schedule(A, x, b, fixed_threads_v3, schedule, chunk);
                }
                double avg_time = total_time / NUM_RUNS;
                
                fprintf(file, "%-10s %15d %15.6f\n", schedule.c_str(), chunk, avg_time);
                printf("Type: %-8s Chunks: %4d | Time: %.6f sec\n", 
                       schedule.c_str(), chunk, avg_time);
            }
            fprintf(file, "\n");
            printf("\n");
        }
    }
    
    fclose(file);
    return 0;
}
