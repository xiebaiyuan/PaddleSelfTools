import pickle, pprint

merged_weight_bias = pickle.load(open(
    "/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/modeltools/mobilenet/datas/all_inputs_outputs/merged_wegiths_bias.pkl",
    'r'))

keys = merged_weight_bias.keys()
keys.sort(cmp=None, key=str.lower)

print keys

bin_2_2_dw_0 = merged_weight_bias['conv2_2_dw_0.bin']
# print bin_2_2_dw_0
bin_2_2_dw_1 = merged_weight_bias['conv2_2_dw_1.bin']
# print bin_2_2_dw_1

# ld.print_stride(bin_2_2_dw_0, 100000)
# pprint.pprint(bin_2_2_dw_0)

# for chw in bin_2_2_dw_0:
#     # print n
#     for hw in chw:
#         for h in hw:
#             for i in h:
#                 print i
# print bin_2_2_dw_1

for i in bin_2_2_dw_1:
    print i
