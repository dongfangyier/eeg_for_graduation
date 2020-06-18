# eeg_for_graduation
### 为毕业设计准备的数据分析部分，尚未完成。

#### TO DO LIST
+ 调整all_analyze模块，尝试提高分类准确率
+ 调整cnn_lstm模块，尝试深度学习

#### 工程结构
>+ all_analyze 
>   + all_inone.py 把所有计算好的特征合并到一个csv
>   + anova_all_inone.py 对特征做方差分析
>   + features_classify.py 分类
>   + get_c_v_result.py 分类用到的具体模型
>   + get_dataframe.py 读取计算好的特征文件
>   + learn_curve_all_inone.py 学习曲线
>   + main.py 入口脚本
>   + mrmr_classify_all_inone.py mrmr特征选择后的分类
>   + mrmr_selectFeatures_all_inone.py mrmr特征选择
>   + sklearn_select_features_all_inone.py 特征选择
>
>+ analyze_features 存放特征选择、分类结果的文件夹
>+ cal_feature_data 存放计算特征的结果
>+ calculate_features 计算特征的脚本
>   + main.py 计算特征的入口函数
>   + eeg_*.py 计算对应特征的脚本
>
>+ cnn_lstm 深度学习的相关代码
>   + deep_learn.py 深度学习的具体模型代码
>   + main.py 入口脚本
>   + preprocess_data.py 数据预处理的代码

+ check_file.py 读取文件的工具
+ CONST.py 一些固定路径和基本信息
+ pyeeg.py 计算特征（dfa）的用到的工具
+ test.py 测试用