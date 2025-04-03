import numpy as np
import csv
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm


class KDDPreprocessor:
    def __init__(self):
        self.protocol_encoder = LabelEncoder()
        self.service_encoder = LabelEncoder()
        self.flag_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # 定义连续特征的索引
        self.continuous_features = [0, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18,
                                    19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

        # 初始化攻击类型列表
        self.attack_list = []

        # 添加用于存储预处理参数的变量
        self.preprocessing_params = {
            'mean': None,
            'std': None
        }

    def count_lines(self, file_path):
        print("Calculate the total number of file lines...")
        with open(file_path, 'r') as f:
            total = sum(1 for _ in tqdm(f, desc="Count rows", ncols=100))
        return total

    def fit_encoders(self, data_path):
        print("\n Start training the encoder...")
        protocol_types = set()
        services = set()
        flags = set()

        total_lines = self.count_lines(data_path)

        with open(data_path, 'r') as data_from:
            csv_reader = csv.reader(data_from)
            for row in tqdm(csv_reader, total=total_lines, desc="Training encoder",
                            ncols=100, colour='green'):
                if len(row) > 3:  # 确保行有足够的列
                    protocol_types.add(row[1])
                    services.add(row[2])
                    flags.add(row[3])

        # 训练编码器
        self.protocol_encoder.fit(list(protocol_types))
        self.service_encoder.fit(list(services))
        self.flag_encoder.fit(list(flags))

        print(f"\n编码器训练完成:")
        print(f"- 发现 {len(protocol_types)} 种协议类型")
        print(f"- 发现 {len(services)} 种服务")
        print(f"- 发现 {len(flags)} 种标志")

    def collect_attack_types(self, source_file):
        print("\nStarting to collect attack types...")
        attack_types = set()

        total_lines = self.count_lines(source_file)

        with open(source_file, 'r') as data_from:
            csv_reader = csv.reader(data_from)
            for row in tqdm(csv_reader, total=total_lines, desc="Collecting attack types",
                            ncols=100, colour='blue'):
                if row:  # Ensure row is not empty
                    attack_type = row[-1].strip()
                    if attack_type:
                        attack_types.add(attack_type)

        self.attack_list = sorted(list(attack_types))
        print(f"\nAttack types collection completed:")
        print(f"- Found {len(self.attack_list)} unique attack types")
        print("- Attack types list:", self.attack_list)

    def calculate_scaling_params(self, source_file):
        """Calculate standardization parameters"""
        print("\nCalculating standardization parameters...")

        # Create arrays to store continuous feature values
        total_lines = self.count_lines(source_file)
        continuous_data = []

        with open(source_file, 'r') as data_from:
            csv_reader = csv.reader(data_from)
            for row in tqdm(csv_reader, total=total_lines, desc="Collecting continuous feature data",
                            ncols=100, colour='yellow'):
                if len(row) > max(self.continuous_features):
                    try:
                        # Only collect continuous feature values
                        feature_values = []
                        for idx in self.continuous_features:
                            try:
                                feature_values.append(float(row[idx]))
                            except ValueError:
                                feature_values.append(0.0)
                        continuous_data.append(feature_values)
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue

        continuous_data = np.array(continuous_data)

        # 计算均值和标准差
        print("计算均值和标准差...")
        continuous_mean = np.mean(continuous_data, axis=0)
        continuous_std = np.std(continuous_data, axis=0)
        continuous_std = np.where(continuous_std == 0, 1, continuous_std)  # 避免除以零

        # 创建完整的均值和标准差数组
        full_mean = np.zeros(41)  # 41个特征
        full_std = np.ones(41)  # 41个特征

        # 将连续特征的均值和标准差放入完整数组
        for i, idx in enumerate(self.continuous_features):
            full_mean[idx] = continuous_mean[i]
            full_std[idx] = continuous_std[i]

        # 保存预处理参数
        self.preprocessing_params['mean'] = full_mean
        self.preprocessing_params['std'] = full_std

        print("标准化参数计算完成")

    def preprocess(self, source_file, handled_file):
        """主预处理函数"""
        print("\n=== 开始数据预处理 ===")
        print(f"源文件: {source_file}")
        print(f"目标文件: {handled_file}")

        try:
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"找不到源文件: {source_file}")

            # 添加列名
            feature_names = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'attack_type'
            ]

            # 首先收集所有唯一值并训练编码器
            self.fit_encoders(source_file)
            self.collect_attack_types(source_file)

            # 计算标准化参数
            self.calculate_scaling_params(source_file)

            # 准备输出目录
            os.makedirs(os.path.dirname(handled_file), exist_ok=True)

            # 处理并保存数据
            print("\n处理并保存数据...")
            total_lines = self.count_lines(source_file)
            processed_count = 0
            error_count = 0

            with open(handled_file, 'w', newline='', encoding='utf-8') as data_to:
                csv_writer = csv.writer(data_to)
                csv_writer.writerow(feature_names)  # 写入列名

                with open(source_file, 'r') as data_from:
                    csv_reader = csv.reader(data_from)
                    for row in tqdm(csv_reader, total=total_lines, desc="处理数据",
                                    ncols=100, colour='cyan'):
                        try:
                            if len(row) < len(feature_names) - 1:  # -1 因为最后一列是标签
                                error_count += 1
                                continue

                            # 创建特征数组
                            features = np.zeros(41)

                            # 编码分类特征
                            features[1] = self.protocol_encoder.transform([row[1]])[0]
                            features[2] = self.service_encoder.transform([row[2]])[0]
                            features[3] = self.flag_encoder.transform([row[3]])[0]

                            # 处理连续特征
                            for idx in self.continuous_features:
                                try:
                                    features[idx] = float(row[idx])
                                except:
                                    features[idx] = 0.0

                            # 标准化特征
                            features = (features - self.preprocessing_params['mean']) / \
                                       self.preprocessing_params['std']

                            # 准备输出行
                            output_row = list(features)
                            output_row.append(row[-1].strip())  # 添加攻击类型标签

                            # 写入数据
                            csv_writer.writerow(output_row)
                            processed_count += 1

                        except Exception as e:
                            print(f"\n处理行时出错: {e}")
                            error_count += 1
                            continue

            # 保存预处理器状态
            print("\n保存预处理器状态...")
            self.save_preprocessor_state('preprocessor_state.npz')

            # 保存预处理参数
            print("保存预处理参数...")
            np.save('preprocessing_params.npy', self.preprocessing_params)

            print("\n=== 预处理完成! ===")
            print(f"总行数: {total_lines}")
            print(f"成功处理: {processed_count}")
            print(f"处理失败: {error_count}")

        except Exception as e:
            print(f"\n预处理过程中出现严重错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    def save_preprocessor_state(self, filename):
        print(f"正在保存预处理器状态到 {filename}")
        np.savez(filename,
                 protocol_classes=self.protocol_encoder.classes_,
                 service_classes=self.service_encoder.classes_,
                 flag_classes=self.flag_encoder.classes_,
                 attack_list=np.array(self.attack_list),
                 continuous_features=np.array(self.continuous_features))
        print("预处理器状态保存成功")


def main():
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 设置文件路径
        source_file = os.path.join(current_dir, 'kddcup.data.corrected')
        handled_file = os.path.join(current_dir, 'kddcup.data.corrected.csv')

        print("=== KDD Cup 数据预处理工具 ===")
        print(f"当前目录: {current_dir}")
        print(f"源文件路径: {source_file}")
        print(f"目标文件路径: {handled_file}")

        # 创建预处理器实例并处理数据
        preprocessor = KDDPreprocessor()
        preprocessor.preprocess(source_file, handled_file)

    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()