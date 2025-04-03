import tensorflow as tf
import csv
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold


class IDS_CNN:
    def __init__(self):
        self.feature = None
        self.label = None
        self.model = None
        self.early_stopping_patience = 10
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 设置GPU内存增长
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU 内存增长已启用")
            except RuntimeError as e:
                print(f"GPU 设置错误: {e}")

        # 加载预处理器状态
        try:
            if not os.path.exists('preprocessor_state.npz'):
                raise FileNotFoundError("找不到preprocessor_state.npz. 请先运行handle2.py")
            if not os.path.exists('preprocessing_params.npy'):
                raise FileNotFoundError("找不到preprocessing_params.npy. 请先运行handle2.py")

            print("加载预处理器状态...")
            # 加载编码器状态和攻击类型列表
            preprocessor_state = np.load('preprocessor_state.npz', allow_pickle=True)
            self.attack_list = preprocessor_state['attack_list']
            self.protocol_classes = preprocessor_state['protocol_classes']
            self.service_classes = preprocessor_state['service_classes']
            self.flag_classes = preprocessor_state['flag_classes']
            self.continuous_features = preprocessor_state['continuous_features']

            # 加载标准化参数
            self.preprocessing_params = np.load('preprocessing_params.npy', allow_pickle=True).item()

            print(f"成功加载预处理器状态:")
            print(f"- 攻击类型数量: {len(self.attack_list)}")
            print(f"- 协议类型数量: {len(self.protocol_classes)}")
            print(f"- 服务类型数量: {len(self.service_classes)}")
            print(f"- 标志类型数量: {len(self.flag_classes)}")

        except Exception as e:
            print(f"加载预处理器状态时出错: {e}")
            raise

    def build_model(self):
        num_classes = len(self.attack_list)
        if num_classes == 0:
            raise ValueError("Attack types not loaded. Please ensure preprocessor_state.npz exists and is valid.")

        print(f"\nBuilding model...")
        print(f"- Input features: 41")
        print(f"- Output classes: {num_classes}")

        # 创建学习率调度器
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )

        inputs = tf.keras.layers.Input(shape=(41,))

        # 特征处理层
        x = tf.keras.layers.BatchNormalization()(inputs)

        # 第一个密集块
        x = tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # 第二个密集块
        x = tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # 第三个密集块
        x = tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # 输出层
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # 编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()]
        )

        return self.model

    def apply_smote(self, X, y, max_samples_per_class=None):
        print("\nUsing SMOTE to balance dataset...")
        try:
            # Convert one-hot encoding back to category labels
            print("Preparing data...")
            y_labels = np.argmax(y, axis=1)

            # Count samples for each category
            unique_labels, counts = np.unique(y_labels, return_counts=True)

            print("\nOriginal data category distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"- {self.attack_list[label]}: {count} samples")

            # Find categories with more than 5 samples
            valid_labels = unique_labels[counts >= 5]
            if len(valid_labels) < len(unique_labels):
                invalid_labels = [self.attack_list[i] for i in unique_labels[counts < 5]]
                print(f"Warning: The following categories have too few samples and will be excluded from SMOTE:")
                for label in invalid_labels:
                    print(f"- {label}")

            # Only apply SMOTE to categories with sufficient samples
            print("\nStarting data processing...")
            mask = np.isin(y_labels, valid_labels)
            X_valid = X[mask]
            y_valid = y_labels[mask]

            # 创建采样策略
            sampling_strategy = {}
            if max_samples_per_class is not None:
                for label, count in zip(valid_labels, counts[counts >= 5]):
                    if count < max_samples_per_class:
                        sampling_strategy[label] = max_samples_per_class

            # 创建和设置SMOTE
            print("Initialize SMOTE...")
            smote = SMOTE(
                sampling_strategy=sampling_strategy if sampling_strategy else 'auto',
                random_state=42,
                k_neighbors=min(5, min(counts[counts >= 5]) - 1)
            )

            # 执行SMOTE
            print("\n Generate synthetic samples...")
            X_resampled, y_resampled = smote.fit_resample(X_valid, y_valid)

            # 转换回one-hot编码
            print("\n Convert data format...")
            y_resampled = tf.keras.utils.to_categorical(y_resampled, num_classes=len(self.attack_list))

            # 显示平衡后的类别分布
            y_balanced_labels = np.argmax(y_resampled, axis=1)
            unique_balanced, counts_balanced = np.unique(y_balanced_labels, return_counts=True)

            print("\n Balanced data category distribution:")
            for label, count in zip(unique_balanced, counts_balanced):
                print(f"- {self.attack_list[label]}: {count} sample")

            print(f"\n Data balancing completed:")
            print(f"- Original dataset size: {len(X_valid)}")
            print(f"- Data set size after balancing: {len(X_resampled)}")

            return X_resampled, y_resampled

        except Exception as e:
            print(f"SMOTE处理出错: {e}")
            print("使用原始数据继续训练...")
            return X, y

    def compute_safe_class_weights(self, y_labels):
        """Calculate class weights, handle cases with very few samples"""
        unique_labels, counts = np.unique(y_labels, return_counts=True)
        total_samples = np.sum(counts)
        weights = total_samples / (len(unique_labels) * counts)

        # Limit weight range to avoid extreme values
        weights = np.clip(weights, 0.1, 10.0)
        return dict(zip(unique_labels, weights))

    def load_data(self):
        """加载和预处理数据"""
        print("\n开始加载数据...")
        self.feature, self.label = [], []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "kddcup.data.corrected.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到预处理后的数据文件: {file_path}")

        print(f"从文件加载数据: {file_path}")

        try:
            if os.path.getsize(file_path) == 0:
                raise ValueError("数据文件为空")

            total_lines = sum(1 for _ in open(file_path)) - 1
            print(f"总样本数: {total_lines}")

            with open(file_path, 'r', encoding='utf-8') as data_from:
                next(data_from)
                csv_reader = csv.reader(data_from)

                for row in tqdm(csv_reader, total=total_lines, desc="加载数据", ncols=100):
                    try:
                        if len(row) < 42:
                            continue

                        # 处理特征
                        feature_row = [float(x) for x in row[:41]]
                        self.feature.append(feature_row)

                        # 处理标签
                        attack_type = row[41].strip()
                        if attack_type in self.attack_list:
                            label_index = np.where(self.attack_list == attack_type)[0][0]
                            label = np.zeros(len(self.attack_list))
                            label[label_index] = 1
                            self.label.append(label)
                        else:
                            self.feature.pop()

                    except Exception as e:
                        print(f"\n处理数据行时出错: {e}")
                        continue

            self.feature = np.array(self.feature)
            self.label = np.array(self.label)

            print(f"\n数据加载完成:")
            print(f"- 特征形状: {self.feature.shape}")
            print(f"- 标签形状: {self.label.shape}")

            # 打印标签分布
            labels = np.argmax(self.label, axis=1)
            unique, counts = np.unique(labels, return_counts=True)
            print("\n原始数据集中攻击类型分布:")
            for label, count in zip(unique, counts):
                print(f"- {self.attack_list[label]}: {count} 样本 ({count/len(labels)*100:.2f}%)")

        except Exception as e:
            print(f"\n加载数据时出现严重错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    def plot_training_history(self, history):
        print("\n生成训练历史图表...")
        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # AUC曲线
        plt.subplot(1, 3, 3)

        # 打印所有可用的历史记录键，帮助诊断
        print("可用的历史记录键:", list(history.history.keys()))

        # 寻找AUC相关的键
        auc_keys = [key for key in history.history.keys() if 'auc' in key.lower()]

        if auc_keys:
            # 选择第一个AUC相关的键
            auc_key = auc_keys[0]
            val_auc_key = f'val_{auc_key}' if f'val_{auc_key}' in history.history else None

            plt.plot(history.history[auc_key], label='Training AUC')
            if val_auc_key:
                plt.plot(history.history[val_auc_key], label='Validation AUC')

            plt.title('Model AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
        else:
            # 如果没有找到AUC相关的键，打印警告并关闭图表
            print("警告：未找到AUC指标。可用的指标包括:", list(history.history.keys()))
            plt.close()
            return

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("训练历史图表已保存为 training_history.png")

    def predict_network_traffic(self, traffic_data):
        try:
            # 创建特征数组
            features = np.zeros(41)

            # 编码分类特征
            features[1] = self.protocol_encoder.transform([traffic_data['protocol_type']])[0]
            features[2] = self.service_encoder.transform([traffic_data['service']])[0]
            features[3] = self.flag_encoder.transform([traffic_data['flag']])[0]

            # 处理连续特征
            for idx in self.continuous_features:
                try:
                    features[idx] = float(traffic_data[idx])
                except:
                    features[idx] = 0.0

            # 标准化特征
            features = (features - self.preprocessing_params['mean']) / \
                      self.preprocessing_params['std']

            # 使用模型预测
            prediction = self.model.predict(features.reshape(1, -1))
            attack_type = self.attack_list[np.argmax(prediction)]
            confidence = np.max(prediction)

            return {
                'attack_type': attack_type,
                'confidence': confidence,
                'is_attack': attack_type.lower() != 'normal',
                'raw_predictions': prediction[0]
            }

        except Exception as e:
            print(f"预测过程中出错: {e}")
            return None

    def train(self, epochs=5, batch_size=1024):
        print("\n=== 开始模型训练 ===")
        try:
            self.load_data()

            if self.feature is None or len(self.feature) == 0:
                print("没有加载到数据，无法进行训练。")
                return

            # Use K-fold cross validation
            n_splits = 5
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.feature)):
                print(f"\nStarting training fold {fold + 1}/{n_splits}...")

                # Prepare training and validation data
                X_train, X_val = self.feature[train_idx], self.feature[val_idx]
                y_train, y_val = self.label[train_idx], self.label[val_idx]

                # Apply SMOTE to balance training data, limit maximum samples per class to 10000
                X_train, y_train = self.apply_smote(X_train, y_train, max_samples_per_class=10000)

                # Calculate class weights
                y_train_labels = np.argmax(y_train, axis=1)
                class_weight_dict = self.compute_safe_class_weights(y_train_labels)

                # Build new model
                self.build_model()

                # 设置回调
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=self.early_stopping_patience,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        f'best_model_fold_{fold + 1}.keras',
                        monitor='val_loss',
                        save_best_only=True,
                        mode='min'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                ]

                # 动态调整batch_size以适应内存
                current_batch_size = batch_size
                max_retries = 3
                training_successful = False

                for attempt in range(max_retries):
                    try:
                        # 训练模型
                        history = self.model.fit(
                            X_train, y_train,
                            epochs=epochs,
                            batch_size=current_batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            class_weight=class_weight_dict,
                            verbose=1
                        )
                        training_successful = True
                        break
                    except Exception as e:
                        print(f"训练过程出错: {e}")
                        # 每次失败都减半batch_size
                        current_batch_size = max(current_batch_size // 2, 32)
                        print(f"尝试使用较小的batch_size: {current_batch_size}")

                if not training_successful:
                    print("多次尝试训练失败，跳过当前折")
                    continue

                # 评估当前折
                evaluation_results = self.model.evaluate(X_val, y_val, verbose=0)
                val_loss = evaluation_results[0]  # 损失值
                val_acc = evaluation_results[1]  # 准确率
                val_precision = evaluation_results[2]  # 精确率
                val_recall = evaluation_results[3]  # 召回率
                val_auc = evaluation_results[4]  # AUC

                fold_results.append((val_loss, val_acc, val_precision, val_recall, val_auc))

                # 如果是最佳模型则保存
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.model.save('best_model.keras')

                # 生成当前折的评估报告
                predictions = self.model.predict(X_val)
                y_pred = np.argmax(predictions, axis=1)
                y_true = np.argmax(y_val, axis=1)

                # 找出实际出现在数据中的类别
                unique_classes = np.unique(y_true)
                present_attack_types = [self.attack_list[i] for i in unique_classes]

                # 保存分类报告
                try:
                    report = classification_report(y_true, y_pred,
                                                   target_names=present_attack_types,
                                                   labels=unique_classes)
                    with open(f'classification_report_fold_{fold + 1}.txt', 'w') as f:
                        f.write(report)
                except Exception as e:
                    print(f"Error saving category report: {e}")

                # 绘制混淆矩阵
                try:
                    plt.figure(figsize=(15, 12))
                    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=present_attack_types,
                                yticklabels=present_attack_types)
                    plt.title(f'Fold {fold + 1} Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f'confusion_matrix_fold_{fold + 1}.png')
                    plt.close()
                except Exception as e:
                    print(f"Error generating confusion matrix: {e}")

                # 绘制训练历史
                try:
                    self.plot_training_history(history)
                except Exception as e:
                    print(f"绘制训练历史时出错: {e}")

            # 输出所有折的平均结果
            mean_val_loss = np.mean([x[0] for x in fold_results])
            mean_val_acc = np.mean([x[1] for x in fold_results])
            mean_val_precision = np.mean([x[2] for x in fold_results])
            mean_val_recall = np.mean([x[3] for x in fold_results])
            mean_val_auc = np.mean([x[4] for x in fold_results])

            print("\n=== Cross Validation Results ===")
            print(f"Mean validation loss: {mean_val_loss:.4f}")
            print(f"Mean validation accuracy: {mean_val_acc:.4f}")
            print(f"Mean validation precision: {mean_val_precision:.4f}")
            print(f"Mean validation recall: {mean_val_recall:.4f}")
            print(f"Mean validation AUC: {mean_val_auc:.4f}")

            print("\n=== Training Completed ===")
            print("Best model saved as 'best_model.keras'")

        except Exception as e:
            print(f"\nError during training:{e}")
            import traceback
            traceback.print_exc()

def main():
    try:
        print("=== 入侵检测系统模型训练 ===")
        model = IDS_CNN()
        model.train(epochs=5, batch_size=1024)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()