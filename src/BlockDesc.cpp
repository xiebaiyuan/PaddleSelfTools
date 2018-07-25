//
// Created by 谢柏渊 on 2018/7/25.
//
#include "BlockDesc.h"
#include <memory>
#include <vector>

#include "framework.pb-c.h"

std::vector<std::shared_ptr<paddle_mobile::framework::VarDesc>> BlockDesc::Vars() const {
    return vars_;
}

BlockDesc::BlockDesc(PaddleMobile__Framework__Proto__BlockDesc *desc)
        : index_(desc->idx), parent_index_(desc->idx) {
    for (int i = 0; i < desc->n_vars; ++i) {
        PaddleMobile__Framework__Proto__VarDesc *var_desc = desc->vars[i];
        vars_.emplace_back(
                std::shared_ptr<paddle_mobile::framework::VarDesc>(new paddle_mobile::framework::VarDesc(var_desc)));
    }

    std::sort(vars_.begin(), vars_.end(),
              [](std::shared_ptr<paddle_mobile::framework::VarDesc> left,
                 std::shared_ptr<paddle_mobile::framework::VarDesc> right) {
                  return left->Name() < right->Name();
              });

//        for (int j = 0; j < desc->n_ops; ++j) {
//            PaddleMobile__Framework__Proto__OpDesc *op_desc = desc->ops[j];
//            ops_.emplace_back(new OpDesc(op_desc));
//        }

}
