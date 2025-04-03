import pandas as pd
import os
from tkinter import filedialog, Tk, messagebox


def convert_to_csv(input_file=None):
    """
    将各种格式文件转换为CSV

    :param input_file: 输入文件路径（可选）
    :return: 转换后的CSV文件路径
    """
    root = Tk()
    root.withdraw()

    try:
        # 如果没有传入文件，则打开文件选择对话框
        if not input_file:
            input_file = filedialog.askopenfilename(
                title="选择要转换的文件",
                filetypes=[
                    ("Excel files", "*.xlsx *.xls"),
                    ("CSV files", "*.csv *.cvs"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )

        if not input_file:
            print("未选择文件")
            return None

        # 根据文件扩展名选择读取方法
        file_extension = os.path.splitext(input_file)[1].lower()

        # 读取文件
        if file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
        elif file_extension in ['.csv', '.cvs', '.txt']:
            # 对于CSV和文本文件，尝试不同的分隔符
            try:
                # 尝试逗号分隔
                df = pd.read_csv(input_file)
            except:
                try:
                    # 尝试制表符分隔
                    df = pd.read_csv(input_file, sep='\t')
                except:
                    # 最后尝试空白分隔
                    df = pd.read_csv(input_file, delim_whitespace=True)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")

        # 生成CSV文件名（使用原始文件名）
        csv_file = os.path.splitext(input_file)[0] + '.csv'

        # 保存为CSV
        print("正在转换为CSV文件...")
        df.to_csv(csv_file, index=False, encoding='utf-8')

        print(f"转换完成！\nCSV文件保存在: {csv_file}")
        messagebox.showinfo("成功", "文件转换完成！")

        return csv_file

    except Exception as e:
        error_msg = f"转换过程中出错：{str(e)}"
        print(error_msg)
        messagebox.showerror("错误", error_msg)
        return None

    finally:
        root.destroy()


def batch_convert(folder_path=None):
    """
    批量转换文件夹中的所有文件

    :param folder_path: 文件夹路径（可选）
    """
    root = Tk()
    root.withdraw()

    try:
        # 如果没有传入文件夹，则打开文件夹选择对话框
        if not folder_path:
            folder_path = filedialog.askdirectory(title="选择要转换的文件夹")

        if not folder_path:
            print("未选择文件夹")
            return

        # 支持的文件扩展名
        supported_extensions = ['.xlsx', '.xls', '.csv', '.cvs', '.txt']

        # 遍历文件夹
        converted_files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # 检查文件扩展名
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in supported_extensions:
                try:
                    csv_file = convert_to_csv(file_path)
                    if csv_file:
                        converted_files.append(csv_file)
                except Exception as e:
                    print(f"转换 {filename} 时出错: {e}")

        print(f"共转换 {len(converted_files)} 个文件")
        messagebox.showinfo("完成", f"共转换 {len(converted_files)} 个文件")

    except Exception as e:
        error_msg = f"批量转换过程中出错：{str(e)}"
        print(error_msg)
        messagebox.showerror("错误", error_msg)

    finally:
        root.destroy()


if __name__ == "__main__":
    # 可以选择单文件转换或批量转换
    choice = input("选择操作模式 (1: 单文件转换, 2: 批量转换): ")

    if choice == '1':
        convert_to_csv()
    elif choice == '2':
        batch_convert()
    else:
        print("无效的选择")