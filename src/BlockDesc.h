//
// Created by 谢柏渊 on 2018/7/25.
//

#ifndef QUALI_BLOCKDESC_H
#define QUALI_BLOCKDESC_H

#include "var_desc.h"

class BlockDesc {
public:
    friend class Node;
    friend class ProgramOptimize;
    BlockDesc() {}
    BlockDesc(PaddleMobile__Framework__Proto__BlockDesc *desc);

    const int &ID() const { return index_; }

    const bool &MultiThread() const { return multi_thread_; }

    const int &Parent() const { return parent_index_; }

    bool operator==(const BlockDesc &in_block) const {
        return this->ID() == in_block.ID() && this->Parent() == in_block.Parent();
    }

    bool operator<(const BlockDesc &in_block) const {
        return this->ID() < in_block.ID() && this->Parent() < in_block.Parent();
    }

    std::vector<std::shared_ptr<paddle_mobile::framework::VarDesc>> Vars() const;

private:
    int index_;
    bool multi_thread_;
    int parent_index_;
    std::vector<std::shared_ptr<paddle_mobile::framework::VarDesc>> vars_;
};


#endif //QUALI_BLOCKDESC_H
