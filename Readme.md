# Tianchi - BigData #
-----

该仓库托管一些我之前参加天池大数据竞赛的代码。有关打比赛的内容，欢迎访问我的博客 [Snoopy_Yuan的博客 - 天池赛](http://blog.csdn.net/Snoopy_Yuan/article/category/6924508) 或 [PnYuan- Homepages - 天池赛](https://pnyuan.github.io/blog/categories/%E5%A4%A9%E6%B1%A0%E5%A4%A7%E6%95%B0%E6%8D%AE%E7%AB%9E%E8%B5%9B/) 。

here is a repository for my code during **Tianchi big data competition**, for more, welcome to my blog [Snoopy_Yuan的博客 - 天池赛](http://blog.csdn.net/Snoopy_Yuan/article/category/6924508)  or [PnYuan_Homepages - 天池赛](https://pnyuan.github.io/blog/categories/%E5%A4%A9%E6%B1%A0%E5%A4%A7%E6%95%B0%E6%8D%AE%E7%AB%9E%E8%B5%9B/) .

## [移动推荐算法赛](https://tianchi.aliyun.com/getStart/introduction.htm?spm=5176.100066.333.1.jRXOx1&raceId=231522) ##

代码内容包括：

|content|code|toolkit|
|-------|----|-------|
|数据加载测试 (data loading test)|[python](./Mobile_Recommendation/data_preanalysis/time_test_of_data_loading.py)|pandas|
|数据可视化分析 (data visualization analysis)|[python](./Mobile_Recommendation/data_preanalysis/data_analysis.py)|pandas, matplotlib|
|基于简单规则的预测 (rule-based prediction)|[python](./Mobile_Recommendation/rule_based/rule_example.py)|pandas|
|特征构建 (feature data construction)|[python](./Mobile_Recommendation/feature_construct/)|pandas|
|基于LR/RF/GBDT/XGBoost模型的预测 (model-based prediction)|[python](./Mobile_Recommendation/model_based/)|pandas, sklearn, matplotlib|

-----

- 初始化环境（自行安装uv）

```
uv sync
```

- 数据加载测试

```bash
uv run python Mobile_Recommendation/data_preanalysis/time_test_of_data_loading.py
```

- 数据可视化分析

```bash
uv run python Mobile_Recommendation/data_preanalysis/data_analysis.py
```

- 基于简单规则的预测

```bash
uv run python Mobile_Recommendation/rule_based/rule_example.py
```

- 特征构建

```bash
uv run python Mobile_Recommendation/feature_construct/divide_data_set.py
uv run python Mobile_Recommendation/feature_construct/feature_construct_part_1.py
uv run python Mobile_Recommendation/feature_construct/feature_construct_part_2.py
uv run python Mobile_Recommendation/feature_construct/feature_construct_part_3.py
```

-

```bash
uv run python Mobile_Recommendation/model_based/k_means_preprocessing.py

uv run python Mobile_Recommendation/model_based/lr_on_subsample.py

uv run python Mobile_Recommendation/model_based/rf_on_subsample.py

uv run python Mobile_Recommendation/model_based/gbdt_on_subsample.py

uv run python Mobile_Recommendation/model_based/xgb_test.py
```