

#include "enforce.h"
#include "var_desc.h"
#include "program_desc.h"
#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include "framework.pb-c.h"
#include "protobuf-c.h"
#include <fstream>
#include <iostream>
#include "log.h"
using std::string;

static const std::string g_googlenet_combine = "/Users/xiebaiyuan/PaddleProject/quali/models/googlenet_combine";
static const std::string g_googlenet = "/Users/xiebaiyuan/PaddleProject/quali/models/googlenet";

char *Get_binary_data(const std::string &filename) {

    FILE *file = fopen(filename.c_str(), "rb");

    PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                          filename.c_str());
    fseek(file, 0, SEEK_END);
    int64_t size = ftell(file);
    DLOG<<"size of "<<filename.c_str()<<"  = "<<size;

    PADDLE_MOBILE_ENFORCE(size > 0, "size is too small");
    rewind(file);
    auto *data = new char[size];
    size_t bytes_read = fread(data, 1, static_cast<size_t>(size), file);
    PADDLE_MOBILE_ENFORCE(bytes_read == size,
                          "read binary file bytes do not match with fseek");
    fclose(file);
    return data;
}

const size_t SIZE_UINT_64 = sizeof(uint64_t);
const size_t SIZE_UINT_32 = sizeof(uint32_t);

static size_t ReadBuffer(const char *file_name, uint8_t **out) {
    FILE *fp;
    DLOG << "*file_name"<< (*file_name);

    fp = fopen(file_name, "rb");
    PADDLE_MOBILE_ENFORCE(fp != nullptr, " %s open failed !", file_name);
    fseek(fp, 0, SEEK_END);
    size_t size = static_cast<size_t>(ftell(fp));

    rewind(fp);
    DLOG << "model size: " << size;

    *out = reinterpret_cast<uint8_t *>(malloc(size));

    size_t cur_len = 0;
    size_t nread;
    while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
        cur_len += nread;
    }
    fclose(fp);
    return cur_len;
}

std::shared_ptr<ProgramDesc> loadParams(const std::string model_path) {
    PaddleMobile__Framework__Proto__ProgramDesc *c_program;
    uint8_t *buf = nullptr;
    DLOG << "model_filename.c_str()"<< model_path.c_str();

    size_t read_size = ReadBuffer(model_path.c_str(), &buf);

    DLOG << "read_size :"<< read_size;

    PADDLE_MOBILE_ENFORCE(buf != nullptr, "read from __model__ is null");

    c_program = paddle_mobile__framework__proto__program_desc__unpack(
            nullptr, read_size, buf);
    //
    PADDLE_MOBILE_ENFORCE(c_program != nullptr, "program is null");
    //

    //std::cout<<"n_ops:  = "<<(*c_program->blocks)->n_ops<<std::endl;

  //  DLOG << "n_ops: " << (*c_program->blocks)->n_ops;
    //
    auto originProgramDesc = std::make_shared<ProgramDesc>(c_program);
    DLOG << "originProgramDesc.get()->Blocks().size() "<< originProgramDesc.get()->Blocks().size();

   // std::cout<<"originProgramDesc = "<<originProgramDesc.get()->Blocks().size()<<std::endl;

    return originProgramDesc;

}

void LoadWithDump(const paddle_mobile::framework::VarDesc &var_desc, char *dataP, FILE *out_file) {
    // 1. version
    uint32_t version = *reinterpret_cast<uint32_t *>(dataP);

    // write version
    fwrite(&version, SIZE_UINT_32, 1, out_file);

    dataP += SIZE_UINT_32;

    // 2 Lod information
    auto *lod_level_ptr = new uint64_t();
    memcpy(lod_level_ptr, dataP, SIZE_UINT_64);

    uint64_t lod_level = 0;
    // write lod Information
    fwrite(&lod_level, SIZE_UINT_64, 1, out_file);
    delete lod_level_ptr;


    dataP += SIZE_UINT_64;


//    auto &lod = *tensor->mutable_lod();
//    lod.resize(lod_level);


    for (uint64_t i = 0; i < lod_level; ++i) {
        uint64_t size = *reinterpret_cast<uint64_t *>(dataP);
        // write lod size
        fwrite(&size, SIZE_UINT_64, 1, out_file);
        (dataP) += SIZE_UINT_64;

        std::vector<size_t> tmp(size / sizeof(size_t));
        for (unsigned long &k : tmp) {
            k = *reinterpret_cast<size_t *>(dataP);
            (dataP) += sizeof(size_t);
        }
        // write lod size vector
        fwrite(&tmp, sizeof(size_t), tmp.size(), out_file);

        //   lod[i] = tmp;
    }

    // 3. tensor version
    uint32_t tensor_version = *reinterpret_cast<uint32_t *>(dataP);
    // write tensor version
    fwrite(&tensor_version, SIZE_UINT_32, 1, out_file);
    (dataP) += SIZE_UINT_32;

    // 4. tensor desc
    int32_t size = *reinterpret_cast<int32_t *>(dataP);
    // write tensor desc
    fwrite(&size, sizeof(int32_t), 1, out_file);
    (dataP) += sizeof(int32_t);

    std::unique_ptr<char[]> buf(new char[size]);
    for (int m = 0; m < size; ++m) {
        buf.get()[m] = (dataP)[m];
    }

    fwrite(buf.get(), sizeof(char), static_cast<size_t>(size), out_file);
    (dataP) += (sizeof(char) * size);

    const paddle_mobile::framework::TensorDesc &desc = var_desc.Tensor_desc();
    int memory_size = 1;
    for (auto l : desc.Dims()) {
        memory_size *= l;
    }


    //tensor->Resize(paddle_mobile::framework::make_ddim(desc.Dims()));

    void *memory = nullptr;
    int type_size = 0;
    switch (desc.DataType()) {
        case paddle_mobile::framework::VARTYPE_TYPE_FP16:
            type_size = 2;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_FP32:
            type_size = 4;
            // memory = tensor->mutable_data<float>();
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_FP64:
            type_size = 8;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_INT32:
            type_size = 4;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_INT64:
            type_size = 8;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_BOOL:
            type_size = 1;
            break;
        default:
            break;
    }
    size_t tensorSize = sizeof(char) * memory_size * type_size;

    memory = new char[tensorSize];

    for (int n = 0; n < tensorSize; ++n) {
        static_cast<char *>(memory)[n] = (dataP)[n];
    }
    dataP += tensorSize;

    // for float 32
    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::min();

    for (int k = 0; k < memory_size; ++k) {
        min_value = std::min(min_value, static_cast<float *> (memory)[k]);
        max_value = std::max(max_value, static_cast<float *> (memory)[k]);
    }

    fwrite(&min_value, sizeof(float), 1, out_file);
    fwrite(&max_value, sizeof(float), 1, out_file);

    for (int g = 0; g < memory_size; ++g) {
        float value = static_cast<float *> (memory)[g];
        auto factor = (uint8_t) round((value - min_value) / (max_value - min_value) * 255);
        fwrite(&factor, sizeof(uint8_t), 1, out_file);
    }

}

void quantificate_combined(const std::string &model_path, const std::string &param_path, const std::string &param_min_path) {
//    paddle_mobile::Loader<paddle_mobile::CPU,paddle_mobile::Precision::FP32 > loader;
    //auto program = loader.Load(model_path, param_path, optimize);

    auto program = loadParams(model_path);

    char *origin_data = Get_binary_data(param_path);
    char *data = origin_data;
    FILE *out_file = fopen(param_min_path.c_str(), "wb");

    for (const auto &block : program->Blocks()) {
        for (const auto &var_desc : block->Vars()) {
            //   auto var = program.scope->Var(var_desc->Name());
            if (var_desc->Persistable()) {
                //     auto tensor = var->template GetMutable<paddle_mobile::framework::LoDTensor>();
                if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
                    continue;
                }
                LoadWithDump(*var_desc, data, out_file);
            }
        }
    }
    fclose(out_file);
    delete origin_data;

}

void quantificate_seperated(const std::string model_dir, const std::string param_min_path) {
    //  paddle_mobile::Loader<paddle_mobile::CPU,paddle_mobile::Precision::FP32 > loader;
    // auto program = loader.Load(model_dir, optimize);

    auto program = loadParams(model_dir + "/__model__");

    std::string shell_command = "mkdir " + param_min_path;
    system(shell_command.c_str());

    for (const auto &block : program->Blocks()) {
        DLOG<<"block->Vars().size(): "<<block->Vars().size();
        for (const auto &var_desc : block->Vars()) {
//                auto var = program.scope->Var(var_desc->Name());
            if (var_desc->Persistable()) {
//                    auto tensor = var->template GetMutable<paddle_mobile::framework::LoDTensor>();
                DLOG<<"var_desc->Name(): "<<var_desc->Name();

                if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
                    continue;
                }
                std::string file_name = param_min_path + "/" + var_desc->Name();

                FILE *out_file = fopen(file_name.c_str(), "wb");
                char *origin_data =
                        Get_binary_data(model_dir + "/" + var_desc->Name());
                char *data = origin_data;
                LoadWithDump(*var_desc, data, out_file);
                delete origin_data;
                fclose(out_file);
            }
        }
    }

}

int main() {
    std::string filename = "params_min";
    std::string model_path = g_googlenet_combine + "/model";
    std::string param_path = g_googlenet_combine + "/params";
    std::string dirname = "param_min_dir";
    std::string model_dir = g_googlenet;
    //quantificate_combined(model_path, param_path,filename);
    quantificate_seperated(model_dir, dirname);

    return 0;
}






