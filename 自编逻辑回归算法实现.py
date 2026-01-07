import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = "电信用户流失.csv"
df = pd.read_csv(DATA_PATH)
print(f"原始数据: {df.shape[0]}个样本, {df.shape[1]}个特征")

#数据预处理
#处理TotalCharges列：转换为数值类型，错误值设为NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_count = df['TotalCharges'].isnull().sum()
if missing_count > 0:
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)   #使用中位数填充缺失值

#创建新特征：月费与总费用的比例
df['MonthlyToTotalRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
#创建新特征：在网时长与月费的比例
df['TenureMonthlyRatio'] = df['tenure'] / (df['MonthlyCharges'] + 1)

services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
service_cols = [col for col in services if col in df.columns]
for col in service_cols:
#映射：No和No internet service为0，Yes为1
    df[col] = df[col].map({'No': 0, 'No internet service': 0, 'Yes': 1})
#总服务数量
df['ServiceCount'] = df[service_cols].sum(axis=1)

#分类特征编码
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']
#创建标签编码器
le = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

#对目标变量进行编码
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

#特征选择和数据集准备
df = df.drop(columns=['customerID'])

#计算特征与目标变量的相关性
correlation = df.corr()['Churn'].abs().sort_values(ascending=False)

#选择相关性大于0.05的特征
selected_features = correlation[correlation > 0.05].index.tolist()
if 'Churn' in selected_features:
    selected_features.remove('Churn')

X = df[selected_features]  #特征矩阵
feature_names = selected_features
y = df['Churn']  #目标变量
print(f"处理完成: X={X.shape}, y={y.shape}")

#数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集流失率: {y_train.mean():.2%}, 测试集流失率: {y_test.mean():.2%}")  #流失率

#特征标准化
#创建标准化器
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  #对训练集进行拟合和转换
X_test_scaled = scaler.transform(X_test)  #对测试集进行转换

learning_rate = 0.1  #学习率
n_iterations = 800  #迭代次数
l2_lambda = 0.05  #L2正则化系数
class_weight_ratio = 1.5  #类别权重比例
cost_missed = 300  #漏报成本
cost_false = 50  #误报成本
best_threshold = 0.45  #初始阈值

#训练集进一步分割
n_samples, n_features = X_train_scaled.shape
split_idx = int(n_samples * 0.9)
X_train_split, X_val = X_train_scaled[:split_idx], X_train_scaled[split_idx:]
y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]

#模型参数初始化
limit = np.sqrt(6 / (n_features + 1))
weights = np.random.uniform(-limit, limit, n_features)
bias = 0

#处理类别不平衡
pos_ratio = np.mean(y_train_split)  #正样本比例
neg_ratio = 1 - pos_ratio  #负样本比例

#计算正样本权重
weight_pos = min(2.5, max(1.0, (neg_ratio / pos_ratio) * class_weight_ratio))
#为每个样本分配权重
sample_weights = np.where(y_train_split == 1, weight_pos, 1.0)

best_val_loss = float('inf')
patience = 20
patience_counter = 0
best_weights = weights.copy()  #保存最佳权重
best_bias = bias  #保存最佳偏置
loss_history = []
accuracy_history = []

def sigmoid(z):
    z = np.clip(z, -50, 50)  #限制z的范围防止数值溢出
    return 1 / (1 + np.exp(-z))

#模型训练循环
print("开始训练逻辑回归模型...")
for i in range(n_iterations):
#前向传播
    linear_output = np.dot(X_train_split, weights) + bias
    y_pred = sigmoid(linear_output)

#计算损失
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
#交叉熵损失
    base_loss = -np.mean(y_train_split * np.log(y_pred_clipped) +
                         (1 - y_train_split) * np.log(1 - y_pred_clipped))
#L2正则化项
    l2_reg = (l2_lambda / (2 * len(y_train_split))) * np.sum(weights ** 2)
    loss = base_loss + l2_reg
    loss_history.append(loss)

#计算准确率
    y_pred_class = (y_pred > best_threshold).astype(int)
    accuracy = accuracy_score(y_train_split, y_pred_class)
    accuracy_history.append(accuracy)

#反向传播
    error = y_pred - y_train_split

#计算梯度
    dw = np.dot(X_train_split.T, error * sample_weights) / len(y_train_split)  # 权重梯度
    db = np.sum(error * sample_weights) / len(y_train_split)  # 偏置梯度
    dw += (l2_lambda / len(y_train_split)) * weights  # 添加L2正则化梯度

#学习率衰减
    current_lr = learning_rate * np.exp(-0.001 * i)

    weights -= current_lr * dw
    bias -= current_lr * db

    if i % 10 == 0:
    #验证集预测
        val_linear_output = np.dot(X_val, weights) + bias
        val_y_pred = sigmoid(val_linear_output)
        val_y_pred_clipped = np.clip(val_y_pred, epsilon, 1 - epsilon)

        val_loss = -np.mean(y_val * np.log(val_y_pred_clipped) +
                            (1 - y_val) * np.log(1 - val_y_pred_clipped))
        val_loss += (l2_lambda / (2 * len(y_val))) * np.sum(weights ** 2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = weights.copy()
            best_bias = bias
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            weights = best_weights
            bias = best_bias
            print(f"早停在迭代 {i}")
            break

    if i % 100 == 0:
        print(f"迭代 {i:4d}: 损失 = {loss:.6f}, 准确率 = {accuracy:.4f}")

#训练结果输出
final_loss = loss_history[-1]  #最终损失
final_accuracy = accuracy_history[-1]  #最终准确率
print(f"训练完成! 最终损失 = {final_loss:.6f}, 最终准确率 = {final_accuracy:.4f}")

# 特征重要性分析
feature_importance = np.abs(weights)
print("\n基于业务成本优化阈值...")

y_proba_val = sigmoid(np.dot(X_val, weights) + bias)

best_threshold = 0.5
min_cost = float('inf')
threshold_analysis = []

#遍历不同的阈值
for th in np.arange(0.1, 0.9, 0.02):
    y_pred_val = (y_proba_val >= th).astype(int)

#计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val).ravel()

#计算业务成本
    total_cost = fn * cost_missed + fp * cost_false  #总成本
    avg_cost = total_cost / len(y_val)  #户均成本

#计算性能指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  #精确率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  #召回率
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    threshold_analysis.append({
        'threshold': th,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_cost': avg_cost,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    })

    if avg_cost < min_cost:
        min_cost = avg_cost
        best_threshold = th

print(f"找到最优阈值: {best_threshold:.3f}")
print(f"对应户均成本: ${min_cost:.2f}")

print("\n  阈值敏感性分析:")
print(f"  {'阈值':<8} {'精确率':<8} {'召回率':<8} {'F1':<8} {'户均成本':<10}")
for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    filtered = [item for item in threshold_analysis if abs(item['threshold'] - th) < 0.01]
    if filtered:
        item = filtered[0]
        marker = " ← 最优" if abs(th - best_threshold) < 0.01 else ""
        print(
            f"  {th:<8.2f} {item['precision']:<8.2%} {item['recall']:<8.2%} "
            f"{item['f1']:<8.4f} ${item['avg_cost']:<9.2f}{marker}")

def predict_proba(X):
    linear_output = np.dot(X, weights) + bias
    return sigmoid(linear_output)

def predict(X, threshold=best_threshold):
    probabilities = predict_proba(X)
    return (probabilities >= threshold).astype(int)

#模型评估
#测试集预测
y_pred = predict(X_test_scaled)
y_pred_proba = predict_proba(X_test_scaled)  # 预测概率

#计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"\n模型评估结果:")
print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"\n混淆矩阵:")
print(f"              预测未流失   预测流失")
print(f"  实际未流失     {cm[0, 0]:5d}      {cm[0, 1]:5d}")
print(f"  实际流失       {cm[1, 0]:5d}      {cm[1, 1]:5d}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
detection_rate = tp / (tp + fn)
false_alarm_rate = fp / (fp + tn)
total_cost = fn * 300 + fp * 50
avg_cost_per_customer = total_cost / len(y_test)

print(f"\n业务评估:")
print(f"流失检测率: {detection_rate:.2%}")
print(f"误报率: {false_alarm_rate:.2%}")
print(f"总业务成本: ${total_cost}")
print(f"户均成本: ${avg_cost_per_customer:.2f}")

os.makedirs('images_cost_optimized', exist_ok=True)

#1.训练过程图
plt.figure(figsize=(15, 5))

#子图:训练损失变化
plt.subplot(1, 3, 1)
plt.plot(loss_history)
plt.title('训练损失变化')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.grid(True, alpha=0.3)

#子图2:训练准确率变化
plt.subplot(1, 3, 2)
plt.plot(accuracy_history, label='准确率')
plt.title('训练准确率变化')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.grid(True, alpha=0.3)

#子图3：阈值-指标关系
plt.subplot(1, 3, 3)
thresholds = np.arange(0.1, 0.9, 0.05)
precisions = []
recalls = []
costs = []
for th in thresholds:
    y_pred_th = (y_pred_proba >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_th).ravel()
    precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    costs.append((fn * 300 + fp * 50) / len(y_test))

plt.plot(thresholds, precisions, 'b-', label='精确率', linewidth=2)
plt.plot(thresholds, recalls, 'r-', label='召回率', linewidth=2)
plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'最优阈值={best_threshold:.2f}')
plt.title('阈值-指标关系')
plt.xlabel('阈值')
plt.ylabel('指标值')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images_cost_optimized/训练过程图.png', dpi=150, bbox_inches='tight')
plt.show()

#2.业务成本分析图
analysis_df = pd.DataFrame(threshold_analysis)
plt.figure(figsize=(12, 8))

#子图1：业务成本-阈值关系
plt.subplot(2, 2, 1)
plt.plot(analysis_df['threshold'], analysis_df['avg_cost'], 'g-', linewidth=2, marker='o')
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'最优阈值={best_threshold:.2f}')
plt.xlabel('阈值')
plt.ylabel('户均成本 ($)')
plt.title('业务成本-阈值关系')
plt.legend()
plt.grid(True, alpha=0.3)

#子图2：精确率-召回率曲线
plt.subplot(2, 2, 2)
plt.plot(analysis_df['recall'], analysis_df['precision'], 'b-', linewidth=2)
optimal_idx = analysis_df['avg_cost'].idxmin()
plt.plot(analysis_df.loc[optimal_idx, 'recall'],
         analysis_df.loc[optimal_idx, 'precision'], 'ro', markersize=10, label=f'最优点')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.legend()
plt.grid(True, alpha=0.3)

#子图3：阈值-指标关系
plt.subplot(2, 2, 3)
plt.plot(analysis_df['threshold'], analysis_df['precision'], 'b-', label='精确率', linewidth=2)
plt.plot(analysis_df['threshold'], analysis_df['recall'], 'r-', label='召回率', linewidth=2)
plt.plot(analysis_df['threshold'], analysis_df['f1'], 'g-', label='F1分数', linewidth=2)
plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'最优阈值')
plt.xlabel('阈值')
plt.ylabel('指标值')
plt.title('阈值-指标关系')
plt.legend()
plt.grid(True, alpha=0.3)

#子图4：最优阈值下业务指标
plt.subplot(2, 2, 4)
metrics = ['精确率', '召回率', 'F1分数', '户均成本']
optimal_metrics = [
    analysis_df.loc[optimal_idx, 'precision'],
    analysis_df.loc[optimal_idx, 'recall'],
    analysis_df.loc[optimal_idx, 'f1'],
    analysis_df.loc[optimal_idx, 'avg_cost'] / 100  # 转换为百元单位
]
colors = ['blue', 'red', 'green', 'orange']
bars = plt.bar(metrics, optimal_metrics, color=colors, alpha=0.7)
plt.ylabel('指标值')
plt.title('最优阈值下业务指标')
plt.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, optimal_metrics):
    if bar.get_height() < 1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2%}', ha='center', va='bottom', fontsize=10)
    else:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'${val * 100:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('images_cost_optimized/业务成本分析.png', dpi=150, bbox_inches='tight')
plt.show()

#3.混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测未流失', '预测流失'],
            yticklabels=['实际未流失', '实际流失'])
plt.title('混淆矩阵热图')
plt.tight_layout()
plt.savefig('images_cost_optimized/混淆矩阵.png', dpi=150, bbox_inches='tight')
plt.show()

#4.ROC曲线
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, 'b-', label=f'ROC曲线 (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'r--', label='随机分类器', linewidth=1)
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC曲线')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images_cost_optimized/ROC曲线.png', dpi=150, bbox_inches='tight')
plt.show()

#5.特征重要性
plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=True).tail(15)  #取前15个重要特征
colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('特征重要性')
plt.title('Top 15 特征重要性')
plt.grid(True, alpha=0.3, axis='x')

#添加特征重要性数值
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
             va='center', fontsize=9)

plt.tight_layout()
plt.savefig('images_cost_optimized/特征重要性.png', dpi=150, bbox_inches='tight')
plt.show()

print("项目执行成功!")
print("输出文件:")
print("images_cost_optimized/训练过程图.png")
print("images_cost_optimized/业务成本分析.png")
print("images_cost_optimized/混淆矩阵.png")
print("images_cost_optimized/ROC曲线.png")
print("images_cost_optimized/特征重要性.png")
print("\n模型参数:")
print(f"权重形状: {weights.shape}")
print(f"偏置项: {bias:.6f}")
print(f"训练迭代次数: {len(loss_history)}")
print(f"最优阈值: {best_threshold:.3f}")
print(f"业务成本参数: 漏报${cost_missed}, 误报${cost_false}")
print(f"   精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}, 户均成本: ${avg_cost_per_customer:.2f}")