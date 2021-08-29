# 基于 opencv 和 tensorflow 的车牌识别项目

## Python 环境配置

| 主要包名称 | 版本 |
| :- | :-: |
| python | 3.6.13 |
| opencv-contrib-python | 3.4.4.19 | 
| tensorflow | 1.12.0 |
| scikit-learn | 0.24.2 |

> 推荐使用 **Anaconda** 或 **Miniconda** 进行环境的搭建

## 主要识别流程

- **车牌定位**：使用 opencv 库函数进行形态学操作，初步定位车牌位置，得到预选区域
- **车牌筛选**：通过训练好的卷积神经网络，对预选区域进行进一步筛选，得到较准确的车牌图片
- **字符分割**：对车牌图片再次进行形态学操作，然后将车牌图片上的字符分割开
- **字符识别**：最后再用卷积神经网络识别字符，输出结果

> **车牌筛选** 和 **字符识别** 使用不同的卷积神经网络

## 操作步骤

- 首先搭建 Python 3.6.13 的基础环境
- 然后通过 pip 导入 requirements.txt 中所需的 Python 包
- 运行 cnn_plate.py 和 cnn_char.py 进行模型的训练
- 调整 lpr_main.py 中模型的路径，最后运行，输出结果

## 说明

本项目参考了 GitHub 作者 ***simple2048*** 的代码，并在此基础上进行了修改，提高了识别的准确率、增大了适用范围和精简了代码，参考源地址：https://github.com/simple2048/CarPlateIdentity
