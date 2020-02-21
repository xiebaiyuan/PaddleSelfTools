//
// Created by 谢柏渊 on 2018/7/25.
//

#ifndef QUALI_PROGRAM_DESC_H
#define QUALI_PROGRAM_DESC_H


#include <memory>
#include "framework.pb-c.h"
#include "BlockDesc.h"
#include <vector>

class ProgramDesc {
public:
//    friend class Node;
//
//    friend class ProgramOptimize;

    explicit ProgramDesc(PaddleMobile__Framework__Proto__ProgramDesc *desc);

    const std::vector<std::shared_ptr<BlockDesc>> Blocks();


private:
    std::vector<std::shared_ptr<BlockDesc>> blocks_;
};

#endif //QUALI_PROGRAM_DESC_H
