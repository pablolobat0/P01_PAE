#include <npp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// Función para cargar imágenes PGM
bool loadPGM(const std::string &filename, std::vector<unsigned char> &image, int &width, int &height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    std::string magic;
    file >> magic;
    if (magic != "P5") return false;

    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.ignore(); // Ignorar un espacio en blanco

    image.resize(width * height);
    file.read(reinterpret_cast<char*>(image.data()), width * height);
    return true;
}

// Función para guardar imágenes PGM
bool savePGM(const std::string &filename, const std::vector<unsigned char> &image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), width * height);
    return true;
}

int main() {
    // Cargar imagen PGM
    int width, height;
    std::vector<unsigned char> hostImage;
    if (!loadPGM("input.pgm", hostImage, width, height)) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return EXIT_FAILURE;
    }
    struct timeval ex_start, ex_finish, init_start, init_finish;
    // Definir un kernel de convolución (Ejemplo: Detector de bordes Sobel)
    const int kernelSize = 3;


    gettimeofday(&init_start, NULL);
    float h_kernel[kernelSize * kernelSize] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    // Alojar memoria en GPU para imagen y kernel
    Npp8u *d_src, *d_dst;
    Npp32f *d_kernel;
    CHECK_CUDA(cudaMalloc((void**)&d_src, width * height * sizeof(Npp8u)));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, width * height * sizeof(Npp8u)));
    CHECK_CUDA(cudaMalloc((void**)&d_kernel, kernelSize * kernelSize * sizeof(Npp32f)));

    // Copiar datos a la GPU
    CHECK_CUDA(cudaMemcpy(d_src, hostImage.data(), width * height * sizeof(Npp8u), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(Npp32f), cudaMemcpyHostToDevice));

    // Definir ROI (Región de Interés)
    NppiSize srcSize = { width, height }; //  Toda la imagen
    NppiSize maskSize = { kernelSize, kernelSize }; //el kernel
    NppiPoint anchor = { kernelSize / 2, kernelSize / 2 }; //  Punto central del kernel

    gettimeofday(&init_finish, NULL);

    gettimeofday(&ex_start, NULL);
    // Ejecutar convolución con NPP
    nppiFilter_8u_C1R(
        d_src, width * sizeof(Npp8u),
        srcSize,
        {0, 0}, // ROI offset
        d_dst, width * sizeof(Npp8u),
        maskSize,
        d_kernel,
        anchor
    );

    gettimeofday(&ex_finish, NULL);
    // Copiar resultado de vuelta a CPU
    std::vector<unsigned char> hostOutput(width * height);
    CHECK_CUDA(cudaMemcpy(hostOutput.data(), d_dst, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost));

    // Guardar la imagen de salida
    if (!savePGM("output.pgm", hostOutput, width, height)) {
        std::cerr << "Error al guardar la imagen." << std::endl;
        return EXIT_FAILURE;
    }
    
    time = (init_finish.tv_sec - init_start.tv_sec +
            (init_finish.tv_usec - init_start.tv_usec) / 1.e6);

    printf("Reserva de memoria: %.10lf\n", time);

    time = (ex_finish.tv_sec - ex_start.tv_sec +
            (ex_finish.tv_usec - ex_start.tv_usec) / 1.e6);

    printf("Tiempo de Ejecucion: %.10lf\n", time);




    // Liberar memoria GPU
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);

    std::cout << "Convolución completada. Imagen guardada como output.pgm" << std::endl;
    return 0;
}
