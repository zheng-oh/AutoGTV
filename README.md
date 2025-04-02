# AutoGTV：基于深度学习的肺癌放疗靶区智能勾画引擎

当前目录有一个test文件夹，里面有imgae、gtv文件夹，image为原始CT图序列，gtv为预测结果序列

## docker
#### 测试样本目录为：./test/image/patient001/*.dcm，名称为patient001，patient001可以自定义修改

docker run -e SAMPLE_NAME=patient001 -v ./test/image/patient001/:/workspace/test/image/ -v ./test/gtv/:/workspace/test/gtv
 autogtv