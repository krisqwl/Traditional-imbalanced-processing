<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
# Traditional-imbalanced-processing

## 样本类别分布不均衡的概念

* 所谓的不平衡指的是不同类别的样本量异非常大。样本类别分布不平衡主要出现在分类相关的建模问题上。样本类别分布不均衡从数据规模上可以分为大数据分布不均衡和小数据分布不均衡两种。
	* 大数据分布不均衡。这种情况下整体数据规模大，只是其中的少样本类的占比较少。但是从每个特征的分布来看，小样本也覆盖了大部分或全部的特征。例如1000万条记录的数据集中，其中占比50万条的少数分类样本便于属于这种情况。
	* 小数据分布不均衡。这种情况下整体数据规模小，并且占据少量样本比例的分类数量也少，这会导致特征分布的严重不平衡。例如拥有1000条数据样本的数据集中，其中占有10条样本的分类，其特征无论如何拟合也无法实现完整特征值的覆盖，此时属于严重的数据样本分布不均衡。
* 样本分布不均衡将导致样本量少的分类所包含的特征过少，并很难从中提取规律；即使得到分类模型，也容易产生过度依赖于有限的数据样本而导致过拟合的问题，当模型应用到新的数据上时，模型的准确性和鲁棒性将很差。
* 样本分布不平衡主要在于不同类别间的样本比例差异，通常来看，如果不同分类间的样本量差异达到超过10倍就需要引起警觉并考虑处理该问题。

## 非平衡数据集容易出现的场景

* 异常检测场景。大多数企业中的异常个案都是少量的，比如恶意刷单、黄牛订单、信用卡欺诈、电力窃电、设备故障等，这些数据样本所占的比例通常是整体样本中很少的一部分，以信用卡欺诈为例，刷实体信用卡的欺诈比例一般都在0.1%以内。
* 客户流失场景。大型企业的流失客户相对于整体客户通常是少量的，尤其对于具有垄断地位的行业巨擘，例如电信、石油、网络运营商等更是如此。
* 罕见事件的分析。罕见事件与异常检测类似，都属于发生个案较少；但不同点在于异常检测通常都有是预先定义好的规则和逻辑，并且大多数异常事件都对会企业运营造成负面影响，因此针对异常事件的检测和预防非常重要；但罕见事件则无法预判，并且也没有明显的积极和消极影响倾向。例如由于某网络大V无意中转发了企业的一条趣味广告导致用户流量明显提升便属于此类。
* 发生频率低的事件。这种事件是预期或计划性事件，但是发生频率非常低。例如每年1次的双11盛会一般都会产生较高的销售额，但放到全年来看这一天的销售额占比很可能只有1%不到，尤其对于很少参与活动的公司而言，这种情况更加明显。这种属于典型的低频事件。

## 典型采样算法

* 欠采样
	* RandomUnderSampler  
	* NearMiss-1
	* NearMiss-2
	* NearMiss-3
* 过采样
	* SMOTE
	* Borderline-SMOTE1
	* Borderline-SMOTE2
	* SVMSMOTE
	* ADASYN
	* SMOTENC
* 混合采样
	* SMOTETomek

### 欠采样

* RandomUnderSampler
	* 从majority class中随机删除一部分样本点，使整个样本达到均衡。
* NearMiss
	* Version1:计算majority class中所有点的距离最近的三个minority class样本点的平均欧氏距离，从距离最小的majority class的样本点中依次选择，使整个样本达到均衡。
	* Version2:计算majority class中所有点的距离最远的三个minority class样本点的平均欧氏距离，从距离最小的majority class的样本点中依次选择，使整个样本达到均衡。
	* Version3:为每个minority class样本点选择给定数量的majority class样本点(KNN)，最后去除重复点，以保证每个minority class样本点都被一些近邻的minority class样本点环绕。

### 过采样

* SMOTE
	* $x_{new}=x_i+(\hat{x_i}−x_i) \times\delta,\delta \in [0,1]􏲉$ 
* Borderline-SMOTE
	* Version1:
		* $synthetic_j=p_i'+r_j \times dif_j,j=1,2,...,s<k,r_j \in (0,1)$ 
		* 边界点的knn从minority class中选取
	* Version2:
		* 边界点的knn从minority class和majority class中共同选取，当选取的点属于majority class时，$r_j \in (0,0.5)$
* SVMSMOTE
	* 基于SVM算法首先找到属于minority class中的支持向量，遍历支持向量。
	* 如果某$sv_i$周围的majority class样本点更多(KNN)，则：$s_{i\_new}=sv_i+(sv_i-x_{im}) \times \lambda$
	* 如果某$sv_i$周围的minority class样本点更多(KNN)，则：$s_{i\_new}=sv_i+(x_{im}-sv_i) \times \lambda$
* ADASYN
	* 基于SMOTE的改进算法。
	* 根据minority class样本点的knn中的majority class样本点个数来自适应地确定该minority class样本点应该生成的人造样本的数量。
* SMOTENC
	* 基于SMOTE的改进算法，针对数据集中存在离散特征的情况。

### 混合采样

* SMOTETomek
	* Tomek links对：具有最小距离的异类样本点对
	* 对原始数据集过采样(SMOTE)之后，找出其中所有的Tomek links对并删除，使得所有样本点的最近邻点都属于同类

### 各类算法优缺点
* 过采样和欠采样均存在的一个问题是他们都会改变样本的原始分布，从而可能带来训练的偏差。
* 欠抽样（RandomUnderSampler、NearMiss）
	* 优点：简单，快捷，容易理解，不涉及复杂算法与技术，仅仅依靠简单随机抽样。
	* 缺点：可能丢失多数类样本中的某些重要信息，导致分类不准确。
* 过采样
	* SMOTE
		* 优点：通过对人造样本添加噪声，有助于打破简单随机过采样导致数据集过于紧密的缺点，并以显着改善分类性能的方式增加原始数据集。
		* 缺点：原始数据集过度泛化，SMOTE为每个原始少数类生成相同数量的人造数据样本，而不考虑相邻样本点的关系，这会增加类之间发生重叠的概率。
	* Borderline-SMOTE、SVMSMOTE、ADASYN 
		* 优点：能一定程度改善SMOTE的的缺点，使得人造样本的分布更加均衡。
		* 缺点：不能完全消除过拟合的可能。
	* SMOTENC
		* 优点：能够对存在离散特征的情况做出有针对性的处理。
* SMOTETomek(混合采样)
	* 优点：可以改善SMOTE的缺点，利用Tomek Links的性质，直接地去除重叠效应，使得样本分布更为理想和稳健。
