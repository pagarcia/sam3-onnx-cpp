#include <iostream>
#include <exception>
#include <string>

int runOnnxTestImage(int argc, char** argv);
int runOnnxTestVideo(int argc, char** argv);

static void printMainUsage()
{
    std::cout
        << "\nUSAGE\n"
        << "  Segment <mode> [options]\n\n"
        << "Modes:\n"
        << "  --onnx_test_image   interactive SAM3 image demo\n"
        << "  --onnx_test_video   interactive SAM3 video tracker demo\n\n"
        << "Examples\n"
        << "  Segment --onnx_test_image --image sample.png\n"
        << "  Segment --onnx_test_image --image sample.png --box 100,80,420,350 --save_overlay out.png\n"
        << "  Segment --onnx_test_video --video clip.mp4 --prompt seed_points\n"
        << "  Segment --onnx_test_video --video clip.mp4 --box 120,80,520,430 --max_frames 30\n"
        << std::endl;
}

int main(int argc, char** argv)
{
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    try {
        if (argc < 2) {
            char* fakeArgv[2] = {argv[0], const_cast<char*>("--onnx_test_image")};
            return runOnnxTestImage(2, fakeArgv);
        }

        const std::string mode = argv[1];
        if (mode == "--onnx_test_image") {
            return runOnnxTestImage(argc, argv);
        }
        if (mode == "--onnx_test_video") {
            return runOnnxTestVideo(argc, argv);
        }

        std::cerr << "[ERROR] Unknown mode: " << mode << '\n';
        printMainUsage();
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] Unhandled exception: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "[ERROR] Unhandled non-standard exception.\n";
        return 1;
    }
}
