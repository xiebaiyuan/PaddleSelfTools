# coding=utf-8
import numpy as np

# fromfile = np.fromfile('/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg.npfile', 'f')
# fromfile = np.fromfile('/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs/0000_0.9946-35960_36550-0377100ea28c4800ca76c01539327637.jpg_p0_0.970266780694.jpg.npfile', 'f')
fromfile = np.fromfile(
    '/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs2/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg.conv1BeforeRelu.npfile',
    'f')

print '第1层 conv add :-------------------------------  '

stride = len(fromfile) / 20
if stride > 0:
    stride = stride
else:
    stride = 1
for i in range(0, len(fromfile), stride):
    print fromfile[i]

print '前20个数: '
for i in range(0, 20):
    print fromfile[i]

print len(fromfile)
print fromfile

fromfile = np.fromfile(
    '/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs2/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg.conv2BeforeRelu.npfile',
    'f')

print '第2层 conv add :-------------------------------  '

stride = len(fromfile) / 20
if stride > 0:
    stride = stride
else:
    stride = 1
for i in range(0, len(fromfile), stride):
    print fromfile[i]

print '前20个数: '
for i in range(0, 20):
    print fromfile[i]
print len(fromfile)
print fromfile



fromfile = np.fromfile(
    '/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs2/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg.Beforepool.npfile',
    'f')
print 'pool前 :-------------------------------  '

stride = len(fromfile) / 20
if stride > 0:
    stride = stride
else:
    stride = 1
for i in range(0, len(fromfile), stride):
    print fromfile[i]
print '前20个数: '
for i in range(0, 20):
    print fromfile[i]
print len(fromfile)
print fromfile


fromfile = np.fromfile(
    '/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs2/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg.Afterpool.npfile',
    'f')
print 'pool后 :-------------------------------  '

stride = len(fromfile) / 20
if stride > 0:
    stride = stride
else:
    stride = 1
for i in range(0, len(fromfile), stride):
    print fromfile[i]
print '前20个数: '
for i in range(0, 20):
    print fromfile[i]
print len(fromfile)
print fromfile

fromfile = np.fromfile(
    '/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs2/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg.fc7.npfile',
    'f')

print '第fc输出层:-------------------------------  '

stride = len(fromfile) / 20
if stride > 0:
    stride = stride
else:
    stride = 1
for i in range(0, len(fromfile), stride):
    print fromfile[i]
print '前20个数: '
for i in range(0, 20):
    print fromfile[i]
print len(fromfile)
print fromfile

fromfile = np.fromfile(
    '/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg.npfile',
    'f')

print '最终 :-------------------------------  '

stride = len(fromfile) / 20
if stride > 0:
    stride = stride
else:
    stride = 1
for i in range(0, len(fromfile), stride):
    print fromfile[i]
print '前20个数: '
for i in range(0, 20):
    print fromfile[i]
print len(fromfile)
print fromfile
