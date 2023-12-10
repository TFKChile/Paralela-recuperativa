#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>

#define FILAS 7121
#define COLUMNAS 10681

using namespace std;
using namespace cv;

//se crea la matriz
int** crearMatriz() {
    int** matriz = new int*[FILAS];
    for (int i = 0; i < FILAS; i++) {
        matriz[i] = new int[COLUMNAS];
    }
    return matriz;
}

//se libera la matriz
void liberarMatriz(int** matriz) {
    for (int i = 0; i < FILAS; i++) {
        delete[] matriz[i];
    }
    delete[] matriz;
}

//creamos la imagen pasandole las matrices corregidas sin el * 
void crearImagen(int** verde, int** azul, int** rojo) {
    // Crear una imagen vacía con 4 canales (incluyendo alfa)
    Mat image(FILAS, COLUMNAS, CV_8UC4);

    for (int i = 0; i < FILAS; i++) {
        for (int j = 0; j < COLUMNAS; j++) {
            // Comprobar que los valores están en el rango [0, 255]columnas
            uchar valorVerde = static_cast<uchar>(max(0, min(255, verde[i][j])));
            uchar valorAzul = static_cast<uchar>(max(0, min(255, azul[i][j])));
            uchar valorRojo = static_cast<uchar>(max(0, min(255, rojo[i][j])));
            // Asignar los valores de color al píxel (BGR + Alfa(255))
            image.at<Vec4b>(i, j) = Vec4b(valorAzul, valorVerde, valorRojo, 255);
        }
    }

    // Guardar la imagen
    imwrite("ImagenGalaxia.png", image);
}

void leerArchivo(const char* nombreArchivo, int** matriz) {

    FILE* file = fopen(nombreArchivo, "r");
    if (file == NULL) {
        fprintf(stderr, "Error al abrir el archivo %s\n", nombreArchivo);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < FILAS; i++) {
        for (int j = 0; j < COLUMNAS; j++) {
            char valor[10];
            if (fscanf(file, "%s", valor) != 1) {
                fprintf(stderr, "Error al leer el archivo %s\n", nombreArchivo);
                fclose(file);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            if (strcmp(valor, "*") == 0) {
                matriz[i][j] = -1; // Marcar con un valor especial para identificar el '*'
            } else {
                matriz[i][j] = atof(valor);
            }
        }
    }

    fclose(file);
}

void calcularValores(int** matrizVerde, int** matrizAzul, int** matrizRojo, int** matrizPromedio) {
    for (int i = 0; i < FILAS; i++) {
        for (int j = 0; j < COLUMNAS; j++) {
            // Asegúrese de que todos los valores necesarios estén disponibles antes de calcular
            // usamos las formulas proporcionada para calcular el * de las matrices
            if (matrizVerde[i][j] == -1 && matrizRojo[i][j] != -1 && matrizAzul[i][j] != -1) {
                matrizVerde[i][j] = (matrizPromedio[i][j] - 0.3 * matrizRojo[i][j] - 0.11 * matrizAzul[i][j]) / 0.59;
            }
            if (matrizAzul[i][j] == -1 && matrizRojo[i][j] != -1 && matrizVerde[i][j] != -1) {
                matrizAzul[i][j] = (matrizPromedio[i][j] - 0.3 * matrizRojo[i][j] - 0.59 * matrizVerde[i][j]) / 0.11;
            }
            if (matrizRojo[i][j] == -1 && matrizVerde[i][j] != -1 && matrizAzul[i][j] != -1) {
                matrizRojo[i][j] = (matrizPromedio[i][j] - 0.59 * matrizVerde[i][j] - 0.11 * matrizAzul[i][j]) / 0.3;
            }
        }
    }
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int** verde = crearMatriz();
    int** azul = crearMatriz();
    int** rojo = crearMatriz();
    int** promedio = crearMatriz();

    // Cada proceso maneja una parte del trabajo son 5 procesos
    if (world_rank == 0) {
        leerArchivo("verde.txt", verde);
        calcularValores(verde, azul, rojo, promedio);
        MPI_Send(&(verde[0][0]), FILAS * COLUMNAS, MPI_INT, 4, 0, MPI_COMM_WORLD); //se envia a world_rank 4
    } else if (world_rank == 1) {
        leerArchivo("azul.txt", azul);
        calcularValores(verde, azul, rojo, promedio);
        MPI_Send(&(azul[0][0]), FILAS * COLUMNAS, MPI_INT, 4, 1, MPI_COMM_WORLD);
    } else if (world_rank == 2) {
        leerArchivo("rojo.txt", rojo);
        calcularValores(verde, azul, rojo, promedio);
        MPI_Send(&(rojo[0][0]), FILAS * COLUMNAS, MPI_INT, 4, 2, MPI_COMM_WORLD);
    } else if (world_rank == 3) {
        leerArchivo("promedio.txt", promedio);
        calcularValores(verde, azul, rojo, promedio);
        MPI_Send(&(promedio[0][0]), FILAS * COLUMNAS, MPI_INT, 4, 3, MPI_COMM_WORLD);
    } else if (world_rank == 4) {
        MPI_Recv(&(verde[0][0]), FILAS * COLUMNAS, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(azul[0][0]), FILAS * COLUMNAS, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(rojo[0][0]), FILAS * COLUMNAS, MPI_INT, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(promedio[0][0]), FILAS * COLUMNAS, MPI_INT, 3, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        crearImagen(verde, azul, rojo); //se crea la imagen
    }

    // Liberar memoria
    liberarMatriz(verde);
    liberarMatriz(azul);
    liberarMatriz(rojo);
    liberarMatriz(promedio);

    MPI_Finalize();
    return 0;
}