#include <stdio.h>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>

using namespace MNN;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./eval_mnn/build/run models/mnist_cnn.mnn input.jpg [forwardType] [precision] [thread]\n");
        return 0;
    }

    return 0;
}
