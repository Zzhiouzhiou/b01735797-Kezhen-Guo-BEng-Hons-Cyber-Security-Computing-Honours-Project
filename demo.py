import os
import tkinter as tk
import re
import threading
import queue
import psutil
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from network_monitor import NetworkMonitor

# Attack type normalization
def normalize_attack_type(value):
    return re.sub(r'\W+', '', str(value).strip().lower())

class UniversalThreatDetector:
    def __init__(self, model_path='best_model.keras', preprocessing_params_path='preprocessing_params.npy'):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessing_params = np.load(preprocessing_params_path, allow_pickle=True).item()
        preprocessor_state = np.load('preprocessor_state.npz', allow_pickle=True)
        self.attack_types = preprocessor_state['attack_list']
        self.protocol_classes = preprocessor_state['protocol_classes']
        self.service_classes = preprocessor_state['service_classes']
        self.flag_classes = preprocessor_state['flag_classes']
        self.continuous_features = [0, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18,
                                    19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    def encode_categorical_feature(self, value, classes):
        value = str(value).strip()
        return np.where(classes == value)[0][0] if value in classes else 0

    def preprocess_network_data(self, data):
        processed_data = np.zeros((len(data), 41))
        for i, row in data.iterrows():
            processed_data[i, 1] = self.encode_categorical_feature(row['protocol_type'], self.protocol_classes)
            processed_data[i, 2] = self.encode_categorical_feature(row['service'], self.service_classes)
            processed_data[i, 3] = self.encode_categorical_feature(row['flag'], self.flag_classes)
            for j in self.continuous_features:
                processed_data[i, j] = float(row.iloc[j]) if j < len(row) else 0.0
        mean = self.preprocessing_params['mean']
        std = self.preprocessing_params['std']
        return (processed_data - mean) / np.where(std == 0, 1, std)

    def detect_threats(self, data):
        features = self.preprocess_network_data(data)
        predictions = self.model.predict(features, verbose=0)
        results = []
        for pred in predictions:
            label_index = np.argmax(pred)
            attack_type = self.attack_types[label_index]
            results.append({
                'threat_detected': normalize_attack_type(attack_type) != 'normal',
                'attack_type': attack_type,
                'confidence': float(np.max(pred))
            })
        return results

class IDS_App:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Network IDS")
        self.root.geometry("1000x700")
        self.detector = UniversalThreatDetector()
        self.monitor = NetworkMonitor(callback=self.process_data)
        self.results = []
        self.monitoring = False
        self.setup_ui()

    def setup_ui(self):
        control = ttk.Frame(self.root)
        control.pack(fill='x', padx=10, pady=5)
        self.start_btn = ttk.Button(control, text="Start Monitoring", command=self.toggle_monitor)
        self.start_btn.pack(side='left')
        self.status = ttk.Label(control, text="Status: Idle")
        self.status.pack(side='left', padx=10)

        self.tree = ttk.Treeview(self.root, columns=('Time', 'Src IP', 'Dst IP', 'Protocol', 'Type', 'Conf'), show='headings')
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)
        self.tree.pack(fill='both', expand=True, padx=10, pady=5)
        self.tree.tag_configure('threat', foreground='red')
        self.tree.tag_configure('normal', foreground='green')

        chart_frame = ttk.LabelFrame(self.root, text="Threat Distribution")
        chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
        fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def toggle_monitor(self):
        if not self.monitoring:
            self.monitor.set_interface(self.monitor.available_interfaces[0]['name'])
            self.monitor.start_capture()
            self.monitoring = True
            self.start_btn.configure(text="Stop Monitoring")
            self.status.configure(text="Status: Monitoring", foreground='green')
        else:
            self.monitor.stop_capture()
            self.monitoring = False
            self.start_btn.configure(text="Start Monitoring")
            self.status.configure(text="Status: Stopped", foreground='red')
            self.update_chart()

    def process_data(self, data):
        results = self.detector.detect_threats(data)
        now = datetime.now().strftime("%H:%M:%S")
        for i, res in enumerate(results):
            tag = 'normal' if not res['threat_detected'] else 'threat'
            src = data.iloc[i].get('src_ip', 'N/A')
            dst = data.iloc[i].get('dst_ip', 'N/A')
            proto = data.iloc[i].get('protocol_type', 'N/A')
            self.tree.insert('', 'end', values=(now, src, dst, proto, res['attack_type'], f"{res['confidence']:.2%}"), tags=(tag,))
            self.results.append(res)

    def update_chart(self):
        self.ax.clear()
        dist = {}
        for r in self.results:
            if r['threat_detected']:
                atype = r['attack_type']
                dist[atype] = dist.get(atype, 0) + 1
        if dist:
            self.ax.pie(dist.values(), labels=dist.keys(), autopct='%1.1f%%')
            self.ax.set_title("Detected Threats")
        else:
            self.ax.text(0.5, 0.5, "No threats detected", ha='center')
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = IDS_App(root)
    root.mainloop()

if __name__ == '__main__':
    main()
