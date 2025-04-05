# AutoGTV：基于深度学习的肺癌放疗靶区智能勾画引擎

#### 1. docker 镜像拉取
```
docker pull zxingcc/autogtv:latest
```

#### 2. 准备测试样本，目录格式为：
    test/
    ├── image/
    │   └── patient001/
    │       └── *.dcm (多个DICOM文件)
    └── gtv/
        └── (模型预测结果的输出目录)
⚠️名称为patient001，patient001可以自定义修改，测试另外的数据依然需要放到test/image下

#### 3. docker运行测试

```
docker run -e SAMPLE_NAME=patient001 -v ./test/image/patient001/:/workspace/test/image/ -v ./test/gtv/:/workspace/test/gtv zxingcc/autogtv
```

#### 4.可视化结果

```
python visualize.py  --samplename patient001
```





