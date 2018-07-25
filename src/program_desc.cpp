//
// Created by 谢柏渊 on 2018/7/25.
//

#include "program_desc.h"
#include <vector>

ProgramDesc::ProgramDesc(PaddleMobile__Framework__Proto__ProgramDesc *desc) {
    for (int i = 0; i < desc->n_blocks; ++i) {
        blocks_.emplace_back(std::make_shared<BlockDesc>(desc->blocks[i]));
    }
}

const std::vector<std::shared_ptr<BlockDesc>> ProgramDesc::Blocks() {
    return blocks_;
}



