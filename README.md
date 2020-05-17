# qssim
一个判断图像相似度的方法。

应用场景：你从某个绅士论坛上获取了某个小姐姐/哥哥的图包合集，而你已经收集了不少这个小姐姐/哥哥的图包，你希望能将两者中重复的图包剔除。

追根究底，需要判断两张图像的相似度，而使用传统的SSIM，需要计算所有图像两两之间的协方差，这会导致不小的内存和时间开销，并且还需要确保图像总是存储在当前使用的计算机上。本项目使用两张图像的差值哈希的交并比代替协方差判断两张图像在`结构`上的相似度。

***
![procedure](https://github.com/wujf98/qssim/raw/master/doc/procedure.jpg)
***
![formula](https://github.com/wujf98/qssim/raw/master/doc/formula.jpg)
***

优势：
1. 存储：图像的差值哈希一旦被计算出来，便可单独存储，形成图像的`指纹`库，用于之后的比较，也允许了原始图像的转移；
2. 内存&时间：显而易见，计算两个固定长度的二进制串的交并比所需的内存和时间远小于计算两张图像的协方差。

建议：
1. 考虑插值哈希的计算和使用过程，不难推知本项目的方法在原始图像和原始图像经过裁剪后得到的图像之间的判断效果并不好，即不支持模板匹配；但对缩放和水印的鲁棒性较好。原始图像与对原始图像不同处理下得到的图像的判断效果总结如下：不处理(或微小处理，如jpg编码转png编码等等) > 缩放、水印 >> 裁剪、旋转。
2. 最终判断图像是否相似的MQSSIM的阈值可设置在0.7~0.95之间，0.7的阈值可以在缩放和水印下有不错的鲁棒性，但如果你的图像的`长宽比`比较悬殊，如1:10，你可能还需要进一步调小阈值，实际上，两张完全不相似的图像的MQSSIM很难达到0.5甚至0.4以上。
