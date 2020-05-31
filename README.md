# qssim
一个判断图像相似度的方法。

应用场景：你从某个绅士论坛上获取了某个小姐姐/哥哥的图包合集，而你已经收集了不少这个小姐姐/哥哥的图包，你希望能将两者中重复的图包剔除。

追根究底，需要判断两张图像的相似度，而使用传统的SSIM(公式如下图所示)，需要计算所有图像两两之间的协方差，这会导致不小的内存和时间开销，并且还需要确保图像总是存储在当前使用的计算机上。

***
![formula_ssim](https://github.com/wujf98/qssim/raw/master/docs/formula_ssim.jpg)

***
本项目使用两张图像的差值哈希的交并比代替协方差判断两张图像在`结构`上的相似度，并将改进后的指标姑且称为`QSSIM`。

***
本项目中差值哈希的计算方法如下图所示。概括来说就是将图像灰度化、重采样到一个较小的尺寸，计算中心某个区域的像素与其相邻的八个方向上像素的差值，并将差值按规则映射为0或1(差值大于0置为1，小于等于0置为0)，组成所谓的差值哈希二进制串。具体到本项目，图像灰度化、重采样到10×10大小，中心的8×8个像素在每一个方向上可获得一个64位的差值哈希二进制串，八个方向共计512位，用以表示图像的整体结构信息。

***
![d_hash](https://github.com/wujf98/qssim/raw/master/docs/d_hash.jpg)

***
在计算两张图像的QSSIM时，为了提高判断的准确率，将图像分块，计算两张图像各个对应区域的QSSIM，最后计算平均值，即`MQSSIM`。与SSIM/MSSIM不同，QSSIM/MQSSIM不再需要图像的直接输入，取而代之的是图像(或图像各个区域)的`指纹`信息，所谓的`指纹`信息，在本项目中指的是图像(或图像各个区域)的均值、方差和差值哈希二进制串。两张图像的MQSSIM的计算公式如下图所示。

***
![formula_mqssim](https://github.com/wujf98/qssim/raw/master/docs/formula_mqssim.jpg)

***
本项目方法的流程如下图所示。最后输出的是一个介于0~1之间的数，这个数值越接近1，说明两张图片越相似。需要注意的是，图像(或图像各个区域)的指纹信息一旦被计算，便可单独存储，反复用于之后的比较。

***
![procedure](https://github.com/wujf98/qssim/raw/master/docs/procedure.jpg)

***
优势：
1. 存储：图像的差值哈希一旦被计算出来，便可单独存储，形成图像的`指纹`库，用于之后的比较，也允许了原始图像的转移；
2. 内存&时间：显而易见，计算两个固定长度的二进制串的交并比所需的内存和时间远小于计算两张图像的协方差。

建议：
1. 考虑差值哈希的计算和使用过程，不难推知本项目的方法在原始图像和原始图像经过裁剪后得到的图像之间的判断效果并不好，即不支持模板匹配；但对缩放和水印的鲁棒性较好。原始图像与对原始图像不同处理下得到的图像的判断效果总结如下：不处理(或微小处理，如jpg编码转png编码等等) > 缩放、水印 >> 裁剪、旋转。
2. 最终判断图像是否相似的MQSSIM的阈值可设置在0.7~0.95之间，0.7的阈值可以在缩放和水印下有不错的鲁棒性，但如果你的图像的`长宽比`比较悬殊，如1:10，你可能还需要进一步调小阈值，实际上，两张完全不相似的图像的MQSSIM很难达到0.5甚至0.4以上。
3. 回到最初的应用场景，对每一个待加入仓库的图包，可以考虑只生成每个图包中少量图像的指纹信息与仓库进行比较，这不仅可以缩短生成指纹的时间，也可缩短比较指纹的时间；如果你希望对待加入仓库的所有图像与仓库进行指纹比较，可以考虑使用其他工具(如格式工厂)生成相比待加入仓库图像在尺寸上更小的副本，生成这个副本的指纹所需要的的时间开销会小得多；此外，你也可以考虑放弃分块，只对一整张图像生成指纹信息，也可以加快生成指纹与比较指纹的速度。
