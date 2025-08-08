# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from datetime import datetime

# 修复日期格式问题的函数
def fix_date(x):
    if isinstance(x, str):
        try:
            # 尝试多种日期格式
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d']:
                try:
                    return datetime.strptime(x, fmt)
                except ValueError:
                    continue
            # 如果所有格式都失败，返回None
            return None
        except:
            return None
    return x

# 加载训练数据
train_data = pd.read_csv('F:/二手车价格预测/used_car_train_20200313/used_car_train_20200313.csv')

# 数据预处理
# 处理日期
train_data['regDate'] = train_data['regDate'].apply(fix_date)
train_data['creatDate'] = train_data['creatDate'].apply(fix_date)

# 特征工程
# 计算车龄
train_data['car_age'] = (train_data['creatDate'] - train_data['regDate']).dt.days / 365.25

# 计算功率与年龄的比率
train_data['power_age_ratio'] = train_data['power'] / (train_data['car_age'] + 1)

# 计算功率与公里数的比率
train_data['power_km_ratio'] = train_data['power'] / (train_data['kilometer'] + 1)

# 定义分类特征
cat_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']

# 准备特征
features = ['power', 'kilometer', 'car_age', 'power_age_ratio', 'power_km_ratio'] + cat_features
for feature in cat_features:
    train_data[feature] = train_data[feature].astype('category')

# 分割训练集和验证集
X = train_data[features]
y = train_data['price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化和训练CatBoost模型
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          early_stopping_rounds=50,
          verbose=100)

# 输出验证集性能
val_predictions = model.predict(X_val)
rmse = np.sqrt(np.mean((y_val - val_predictions) ** 2))
mae = np.mean(np.abs(y_val - val_predictions))
r2 = 1 - np.sum((y_val - val_predictions) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

print(f'Validation RMSE: {rmse:.2f}')
print(f'Validation MAE: {mae:.2f}')
print(f'Validation R2: {r2:.4f}')

# 加载测试数据
test_data = pd.read_csv('used_car_testB_20200421/used_car_testB_20200421.csv')

# 对测试数据进行相同的预处理
test_data['regDate'] = test_data['regDate'].apply(fix_date)
test_data['creatDate'] = test_data['creatDate'].apply(fix_date)

# 特征工程
test_data['car_age'] = (test_data['creatDate'] - test_data['regDate']).dt.days / 365.25
test_data['power_age_ratio'] = test_data['power'] / (test_data['car_age'] + 1)
test_data['power_km_ratio'] = test_data['power'] / (test_data['kilometer'] + 1)

# 转换分类特征
for feature in cat_features:
    test_data[feature] = test_data[feature].astype('category')

# 预测测试集
test_predictions = model.predict(test_data[features])

# 保存预测结果
submission = pd.DataFrame({
    'SaleID': test_data['SaleID'],
    'price': test_predictions
})
submission.to_csv('catboost_predictions.csv', index=False)

print('预测完成，结果已保存到catboost_predictions.csv')