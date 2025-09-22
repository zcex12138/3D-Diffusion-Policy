"""
实时3D点云可视化器
支持1024*6格式的点云数据：前3维是坐标(x,y,z)，后3维是RGB颜色值
"""

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from flask import Flask, render_template_string, jsonify
import json
import threading
import time
import logging
from termcolor import cprint


class RealTime3DVisualizer:
    """
    实时3D点云可视化器类
    支持1024*6格式的点云数据：前3维是坐标(x,y,z)，后3维是RGB颜色值
    """
    
    def __init__(self, window_title="Real-time 3D Point Cloud", window_size=(1200, 800)):
        """
        初始化实时3D可视化器
        
        Args:
            window_title (str): 窗口标题
            window_size (tuple): 窗口大小 (width, height)
        """
        self.window_title = window_title
        self.window_size = window_size
        self.app = None
        self.fig = None
        self.trace = None
        self.is_running = False
        self.update_interval = 100  # 更新间隔(毫秒)
        self.current_point_cloud = None
        self.server_thread = None
        
        # 初始化Flask应用
        self._init_app()
        
    def _init_app(self):
        """初始化Flask应用和Plotly图表"""
        self.app = Flask(__name__)
        
        # 关闭Flask的HTTP请求日志
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # 创建初始的空图表
        self.fig = go.Figure()
        self.fig.update_layout(
            title=self.window_title,
            scene=dict(
                aspectmode='data',
                xaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,
                    gridcolor='grey',
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,
                    gridcolor='grey',
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,
                    gridcolor='grey',
                ),
                bgcolor='white'
            ),
            width=self.window_size[0],
            height=self.window_size[1],
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # 设置路由
        self._setup_routes()
        
    def _setup_routes(self):
        """设置Flask路由"""
        
        @self.app.route('/')
        def index():
            """主页面"""
            # 使用Plotly的内置HTML生成方法
            plotly_html = pio.to_html(self.fig, div_id='plotly-div', include_plotlyjs=False, full_html=False)
            
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ title }}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
                    #plotly-div { width: 100vw; height: 100vh; }
                    .controls { 
                        position: fixed; 
                        top: 10px; 
                        right: 10px; 
                        background: rgba(255,255,255,0.9); 
                        padding: 10px; 
                        border-radius: 5px; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                        z-index: 1000;
                    }
                    .control-btn {
                        margin: 2px;
                        padding: 5px 10px;
                        border: none;
                        border-radius: 3px;
                        cursor: pointer;
                        background: #007bff;
                        color: white;
                    }
                    .control-btn:hover { background: #0056b3; }
                    .control-btn.active { background: #28a745; }
                </style>
            </head>
            <body>
                <div class="controls">
                    <button class="control-btn" id="playPauseBtn" onclick="togglePlayPause()">暂停</button>
                    <button class="control-btn" id="resetBtn" onclick="resetView()">重置视角</button>
                    <div style="margin-top: 5px;">
                        <label>更新间隔: </label>
                        <input type="range" id="intervalSlider" min="50" max="1000" value="{{ update_interval }}" 
                               onchange="changeInterval(this.value)" style="width: 100px;">
                        <span id="intervalValue">{{ update_interval }}ms</span>
                    </div>
                </div>
                {{ plotly_html | safe }}
                <script>
                    var isPlaying = true;
                    var updateInterval = {{ update_interval }};
                    var intervalId;
                    
                    function startAutoUpdate() {
                        if (intervalId) clearInterval(intervalId);
                        intervalId = setInterval(function() {
                            if (isPlaying) {
                                fetch('/update_data')
                                    .then(response => response.json())
                                    .then(data => {
                                        if (data.point_cloud) {
                                            Plotly.restyle('plotly-div', {
                                                x: [data.point_cloud.x],
                                                y: [data.point_cloud.y], 
                                                z: [data.point_cloud.z],
                                                marker: {color: data.point_cloud.colors}
                                            });
                                        }
                                    })
                                    .catch(error => console.log('Update error:', error));
                            }
                        }, updateInterval);
                    }
                    
                    function togglePlayPause() {
                        isPlaying = !isPlaying;
                        var btn = document.getElementById('playPauseBtn');
                        btn.textContent = isPlaying ? '暂停' : '播放';
                        btn.className = isPlaying ? 'control-btn' : 'control-btn active';
                    }
                    
                    function resetView() {
                        Plotly.relayout('plotly-div', {
                            'scene.camera': {
                                'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
                            }
                        });
                    }
                    
                    function changeInterval(value) {
                        updateInterval = parseInt(value);
                        document.getElementById('intervalValue').textContent = value + 'ms';
                        if (isPlaying) {
                            startAutoUpdate();
                        }
                    }
                    
                    // 启动自动更新
                    startAutoUpdate();
                </script>
            </body>
            </html>
            """
            
            return render_template_string(
                html_template,
                title=self.window_title,
                plotly_html=plotly_html,
                update_interval=self.update_interval
            )
            
        @self.app.route('/update_data')
        def update_data():
            """更新点云数据的API端点"""
            if hasattr(self, 'current_point_cloud') and self.current_point_cloud is not None:
                point_cloud = self.current_point_cloud
                
                # 提取坐标和颜色
                x_coords = point_cloud[:, 0].tolist()
                y_coords = point_cloud[:, 1].tolist()
                z_coords = point_cloud[:, 2].tolist()
                
                # 处理颜色
                if point_cloud.shape[1] >= 6:
                    # 使用RGB颜色
                    colors = [f'rgb({int(r)},{int(g)},{int(b)})' 
                             for r, g, b in point_cloud[:, 3:6]]
                else:
                    # 使用坐标生成颜色
                    colors = self._generate_coordinate_colors(point_cloud[:, :3])
                
                return jsonify({
                    'point_cloud': {
                        'x': x_coords,
                        'y': y_coords,
                        'z': z_coords,
                        'colors': colors
                    }
                })
            else:
                return jsonify({'point_cloud': None})
                
    def _generate_coordinate_colors(self, coordinates):
        """根据坐标生成颜色"""
        # 归一化坐标到[0,1]范围
        min_coords = coordinates.min(axis=0)
        max_coords = coordinates.max(axis=0)
        normalized_coords = (coordinates - min_coords) / (max_coords - min_coords + 1e-8)
        
        colors = []
        for coord in normalized_coords:
            try:
                r, g, b = int(coord[0] * 255), int(coord[1] * 255), int(coord[2] * 255)
                colors.append(f'rgb({r},{g},{b})')
            except:
                colors.append('rgb(0,255,255)')  # 默认青色
        return colors
        
    def update_point_cloud(self, point_cloud):
        """
        更新点云数据
        
        Args:
            point_cloud (np.ndarray): 形状为(1024, 6)的点云数据
                                   前3维是坐标(x,y,z)，后3维是RGB颜色值
        """
        if point_cloud is None:
            return
            
        # 验证数据格式
        if not isinstance(point_cloud, np.ndarray):
            point_cloud = np.array(point_cloud)
            
        if point_cloud.shape[1] < 3:
            raise ValueError("点云数据至少需要3维坐标信息")
            
        # 存储当前点云数据
        self.current_point_cloud = point_cloud.copy()
        
        # 提取坐标
        x_coords = point_cloud[:, 0]
        y_coords = point_cloud[:, 1] 
        z_coords = point_cloud[:, 2]
        
        # 处理颜色
        if point_cloud.shape[1] >= 6:
            # 使用RGB颜色
            colors = [f'rgb({int(r)},{int(g)},{int(b)})' 
                     for r, g, b in point_cloud[:, 3:6]]
        else:
            # 使用坐标生成颜色
            colors = self._generate_coordinate_colors(point_cloud[:, :3])
        
        # 创建新的trace
        self.trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                opacity=0.8,
                color=colors
            ),
            name='Point Cloud'
        )
        
        # 更新图表数据
        # 清空现有数据并添加新的trace
        self.fig.data = []
        self.fig.add_trace(self.trace)
        
    def start_visualization(self, host='127.0.0.1', port=5000, debug=False, threaded=True, quiet=True):
        """
        启动可视化器
        
        Args:
            host (str): 服务器主机地址
            port (int): 服务器端口
            debug (bool): 是否开启调试模式
            threaded (bool): 是否在单独线程中运行服务器
            quiet (bool): 是否静默模式（关闭HTTP请求日志）
        """
        if quiet:
            # 关闭Flask的HTTP请求日志
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
        
        print(f"Starting 3D point cloud visualizer at http://{host}:{port}")
        print("Press Ctrl+C to stop the visualizer")
        
        if threaded:
            # 在单独线程中运行服务器
            self.server_thread = threading.Thread(
                target=self.app.run,
                kwargs={'host': host, 'port': port, 'debug': debug, 'use_reloader': False},
                daemon=True
            )
            self.server_thread.start()
            self.is_running = True
            
            # 等待服务器启动
            time.sleep(1)
            print(f"Visualizer is running at http://{host}:{port}")
            
        else:
            try:
                self.app.run(host=host, port=port, debug=debug, use_reloader=False)
            except KeyboardInterrupt:
                print("\nVisualizer stopped by user")
                
    def stop_visualization(self):
        """停止可视化器"""
        self.is_running = False
        if self.server_thread and self.server_thread.is_alive():
            # 注意：Flask的run方法没有直接的停止方法
            # 这里只是设置标志，实际停止需要外部干预
            print("Visualizer stop requested")
            
    def set_update_interval(self, interval_ms):
        """
        设置更新间隔
        
        Args:
            interval_ms (int): 更新间隔，单位毫秒
        """
        self.update_interval = interval_ms


# 便捷函数
def create_realtime_visualizer(point_cloud=None, window_title="Real-time 3D Point Cloud"):
    """
    创建并返回实时3D可视化器实例
    
    Args:
        point_cloud (np.ndarray, optional): 初始点云数据
        window_title (str): 窗口标题
        
    Returns:
        RealTime3DVisualizer: 可视化器实例
    """
    visualizer = RealTime3DVisualizer(window_title=window_title)
    
    if point_cloud is not None:
        visualizer.update_point_cloud(point_cloud)
        
    return visualizer


def visualize_realtime_pointcloud(point_cloud, host='127.0.0.1', port=5000, quiet=True):
    """
    快速启动实时点云可视化
    
    Args:
        point_cloud (np.ndarray): 点云数据，形状为(1024, 6)
        host (str): 服务器主机地址  
        port (int): 服务器端口
        quiet (bool): 是否静默模式（关闭HTTP请求日志）
    """
    visualizer = create_realtime_visualizer(point_cloud)
    visualizer.start_visualization(host=host, port=port, quiet=quiet)


# 使用示例
if __name__ == "__main__":
    # 创建示例点云数据
    np.random.seed(42)
    
    # 生成1024个点的点云数据，前3维是坐标，后3维是RGB
    points = np.random.randn(1024, 3) * 0.5  # 坐标
    colors = np.random.randint(0, 256, (1024, 3))  # RGB颜色
    point_cloud = np.concatenate([points, colors], axis=1)
    
    print("Creating sample point cloud data...")
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Coordinates range: {point_cloud[:, :3].min():.2f} to {point_cloud[:, :3].max():.2f}")
    print(f"Color range: {point_cloud[:, 3:6].min()} to {point_cloud[:, 3:6].max()}")
    
    # 创建并启动可视化器
    visualizer = create_realtime_visualizer(point_cloud, "Sample 3D Point Cloud")
    visualizer.start_visualization(port=5001, quiet=True)
    
    # 模拟实时更新点云数据
    try:
        while True:
            time.sleep(2)  # 每2秒更新一次
            
            # 生成新的点云数据
            new_points = np.random.randn(1024, 3) * 0.5
            new_colors = np.random.randint(0, 256, (1024, 3))
            new_point_cloud = np.concatenate([new_points, new_colors], axis=1)
            
            # 更新可视化器
            visualizer.update_point_cloud(new_point_cloud)
            print(f"Updated point cloud at {time.strftime('%H:%M:%S')}")
            
    except KeyboardInterrupt:
        print("\nStopping visualizer...")
        visualizer.stop_visualization()
