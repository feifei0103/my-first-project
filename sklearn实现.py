import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import warnings
import os
import joblib

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('ggplot')

DATA_PATH = "电信用户流失.csv"
df = pd.read_csv(DATA_PATH)
print(f"数据集: {df.shape[0]}个样本, {df.shape[1]}个特征")
print(f"标签分布: 未流失={df['Churn'].value_counts().get('No', 0)}, 流失={df['Churn'].value_counts().get('Yes', 0)}")
print(f"流失比例: {df['Churn'].value_counts(normalize=True).get('Yes', 0):.2%}")

df_processed = df.drop(columns=['customerID'])

#处理TotalCharges列：转换为数值类型，错误值设为NaN
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
missing_count = df_processed['TotalCharges'].isnull().sum()
if missing_count > 0:
    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)   #使用中位数填充缺失值

categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

#使用LabelEncoder对分类特征进行编码
le = LabelEncoder()
for col in categorical_cols:
    df_processed[col] = le.fit_transform(df_processed[col])
df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})

#分离特征和目标变量
X = df_processed.drop(columns=['Churn'])  #特征矩阵
y = df_processed['Churn']   #目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集流失率: {y_train.mean():.2%}, 测试集流失率: {y_test.mean():.2%}")

#创建标准化器
scaler = StandardScaler()
#对训练集进行拟合和转换
X_train_scaled = scaler.fit_transform(X_train)
#对测试集进行转换
X_test_scaled = scaler.transform(X_test)
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

#计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_multiplier = 1.5
class_weight_dict = {0: class_weights[0], 1: class_weights[1] * weight_multiplier}

#定义要训练的模型
models = {
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(probability=True, random_state=42, class_weight='balanced')
}

results = {}
for name, model in models.items():
    print(f"训练{name}...", end='')
    model.fit(X_train_scaled, y_train)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

#定义各模型的最佳阈值
    thresholds = {
        "Logistic Regression": 0.35,
        "Decision Tree": 0.35,
        "Random Forest": 0.45,
        "Gradient Boosting": 0.40,
        "SVM": 0.40
    }
    threshold = thresholds.get(name, 0.35)
    #根据阈值将概率转换为类别预测
    y_pred = (y_proba >= threshold).astype(int)

    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba,
        'threshold': threshold
    }
    print(f" 准确率: {results[name]['accuracy']:.4f}, 召回率: {results[name]['recall']:.4f}")

#创建结果汇总DataFrame
results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Threshold'])
for name, result in results.items():
    new_row = pd.DataFrame({
        'Model': [name],
        'Accuracy': [f"{result['accuracy']:.4f}"],
        'Precision': [f"{result['precision']:.4f}"],
        'Recall': [f"{result['recall']:.4f}"],
        'F1-Score': [f"{result['f1']:.4f}"],
        'ROC-AUC': [f"{result['roc_auc']:.4f}"],
        'Threshold': [f"{result['threshold']}"]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)

#计算混淆矩阵的各组成部分
    tn, fp, fn, tp = confusion_matrix(y_test, result['y_pred']).ravel()
    #计算业务指标
    detection_rate = tp / (tp + fn)
    false_alarm_rate = fp / (fp + tn)
    total_cost = fn * 300 + fp * 50
    avg_cost_per_customer = total_cost / len(y_test)

    print(f"\n{name} - 业务评估")
    print(f"流失检测率: {detection_rate:.2%} ({tp}/{tp + fn})")
    print(f"误报率: {false_alarm_rate:.2%} ({fp}/{fp + tn})")
    print(f"总业务成本: ${total_cost}")
    print(f"户均成本: ${avg_cost_per_customer:.2f}")

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

#根据F1分数选择最佳模型
best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
best_result = results[best_model_name]

print(f"\nBest Model (by F1-Score): {best_model_name} (F1: {best_result['f1']:.4f})")
print(f"\n{best_model_name} Detailed Classification Report:")
print(classification_report(y_test, best_result['y_pred'], target_names=['Not Churn', 'Churn']))

os.makedirs('images', exist_ok=True)

#1.模型准确率和召回率对比图
plt.figure(figsize=(14, 6))
models_list = list(results.keys())

#子图1：准确率对比
plt.subplot(1, 2, 1)
accuracy_scores = [results[m]['accuracy'] for m in models_list]
bars = plt.bar(models_list, accuracy_scores, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
               edgecolor='black', linewidth=1.5)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.6, 0.9)
plt.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, accuracy_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{score:.4f}', ha='center', va='bottom',
             fontsize=9)
plt.xticks(rotation=45, fontsize=10)

#子图2：召回率对比
plt.subplot(1, 2, 2)
recall_scores = [results[m]['recall'] for m in models_list]
bars = plt.bar(models_list, recall_scores, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
               edgecolor='black', linewidth=1.5)
plt.title('Model Recall Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.ylim(0.4, 0.9)
plt.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, recall_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{score:.4f}', ha='center', va='bottom',
             fontsize=9)
plt.xticks(rotation=45, fontsize=10)

plt.tight_layout()
plt.savefig('images/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

#2.最佳模型的混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_result['y_pred'])
#使用热图绘制混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Churn', 'Predicted Churn'],
            yticklabels=['Actual Not Churn', 'Actual Churn'])
plt.title(f'{best_model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

#3.ROC曲线对比图
plt.figure(figsize=(10, 8))
for name, result in results.items():
    if result['y_proba'] is not None and not np.isnan(result['y_proba']).any():
    #计算ROC曲线的假正率和真正率
        fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
        auc_score = result['roc_auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

#添加随机分类器的对角线
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

for name, result in results.items():
    model_filename = name.replace(" ", "_") + ".pkl"
    model_path = f'models/{model_filename}'
    joblib.dump(result['model'], model_path)

print("项目执行成功")
print("输出文件:")
print("images/model_comparison.png - 模型准确率对比")
print("images/confusion_matrix.png - 混淆矩阵")
print("images/roc_curves.png - ROC曲线对比")
print("models/ - 训练好的模型文件")
