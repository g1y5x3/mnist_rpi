#include <stdio.h>
#include <MNN/Interpreter.hpp>
// #include <MNN/expr/Module.hpp>
// #include <MNN/expr/Executor.hpp>
// #include <MNN/expr/ExprCreator.hpp>
// #include <MNN/expr/Executor.hpp>

// using namespace MNN;
// using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    // if (argc < 3) {
    //     MNN_PRINT("Usage: ./run ../../models/mnist_cnn.mnn ../../data/mnist_test_1000.npz\n");
    //     return 0;
    // }

    // create interpreter which holds the model weights
    std::shared_ptr<MNN::Interpreter> model;
    model.reset(MNN::Interpreter::createFromFile("../../models/mnist_cnn.mnn"));
    if (model == nullptr) {
        MNN_ERROR("Invalid Model\n");
        return 0;
    }
    model->setCacheFile(".cachefile");

    // create session which holds iniference data
    
    MNN::ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_CPU;
    sConfig.numThread = 4;
    // TODO: specify backend
    auto session = model->createSession(sConfig);

    // const std::string model_file = "models/mnist_cnn.mnn";  // model file with path
    // const std::vector<std::string> input_names{"input"};
    // const std::vector<std::string> output_names{"output"};

    // Module::Config mdconfig; // default module config
    // std::unique_ptr<Module> module;

    // // module.reset(Module::load(input_names, output_names, model_file.c_str(), rtMgr, &mdconfig));

    return 0;
}
