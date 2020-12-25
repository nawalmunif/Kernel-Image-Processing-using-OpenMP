#include <iostream>;
#include <omp.h>;
#include <time.h>
using namespace std;

int convolution(int InputImage[6][6], int Kernel[3][3], int OuputImage[6][6]) {
    int sum = 0;
    int i, j, k, m, rows, cols;
    rows = 6;
    cols = 6;
    for (int p = 0; p < 6; ++p) {
        for (int h = 0; h < 6; ++h)
            OuputImage[p][h] = { 0 };
    };
    
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    omp_set_nested(1);
#pragma omp parallel shared(InputImage,Kernel) private(i,j,k,m) reduction(+:sum)
#pragma omp for
    for (i = 1; i < rows - 1; i++) {
        for (j = 1; j < cols - 1; j++) {
            sum = 0;
            for (k = -1; k < 2; k++) {
                for (m = -1; m < 2; m++) 
                    sum = sum + InputImage[i + k][j + m] * Kernel[k + 1][m + 1];
                    OuputImage[i][j] = sum;
                
                
                
            }printf(" \n Thread %d has i = %d and j= %d for Output Pixel[i][j] = %d \n", omp_get_thread_num(), i, j, OuputImage[i][j]);
        }
    }


    cout << '\n' << "Output Image Matrix of size 6x6 is:" << '\n';
    for (int l = 0; l < 6; ++l) {
        for (int t = 0; t < 6; ++t)
            cout << OuputImage[l][t] << " ";
        cout << endl;
    }
    return 0;
}

int main()

{
    clock_t Start_time = clock();
    ///------------------Sharpen Kernel Matrix 3x3-----------------------////////////

    int Kernel[3][3] = { {0, 0, 0} , {0, -5, 0} , {0, 0, 0} };

    cout << "Kernel Matrix of 3x3 is:" << '\n';
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            cout << Kernel[i][j] << " ";
        cout << endl;
    };

    //////--------------------------Input Image Matrix of size 6x6----------------------////////


    int Inputdata[6][6] = { 12, 222, 31, 71, 112, 0,
        141, 23, 7, 67, 89, 77,
        36, 87, 90, 44, 51, 8,
        3, 11, 9, 31, 22, 189,
        99, 100, 101, 102, 9, 1,
        35, 41, 221, 22, 11, 17 };

    cout << '\n' << "Input Image Matrix of size 6x6 is:" << '\n';
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j)
            cout << Inputdata[i][j] << " ";
        cout << endl;
    }

    int OutputImage[6][6];
    convolution(Inputdata, Kernel, OutputImage);

    printf(" \n Time taken for execution using OpenMP is: %.2fs\n", (double)(clock() - Start_time) / CLOCKS_PER_SEC);
    
    return 0;
}
