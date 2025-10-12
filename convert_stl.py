#!/usr/bin/env python3
"""
将ASCII格式的STL文件转换为二进制格式
"""
import struct
import re

def ascii_to_binary_stl(ascii_file, binary_file):
    """将ASCII STL文件转换为二进制STL文件"""
    
    with open(ascii_file, 'r') as f:
        content = f.read()
    
    # 提取所有顶点
    vertices = []
    facet_pattern = r'facet normal\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+outer loop\s+vertex\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+vertex\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+vertex\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+endloop\s+endfacet'
    
    matches = re.findall(facet_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        # 每个facet有3个顶点
        v1 = (float(match[3]), float(match[4]), float(match[5]))
        v2 = (float(match[6]), float(match[7]), float(match[8]))
        v3 = (float(match[9]), float(match[10]), float(match[11]))
        vertices.extend([v1, v2, v3])
    
    # 写入二进制STL文件
    with open(binary_file, 'wb') as f:
        # 写入80字节的头部（通常为空）
        f.write(b'\x00' * 80)
        
        # 写入三角形数量（4字节，小端序）
        num_triangles = len(vertices) // 3
        f.write(struct.pack('<I', num_triangles))
        
        # 写入每个三角形
        for i in range(0, len(vertices), 3):
            v1, v2, v3 = vertices[i], vertices[i+1], vertices[i+2]
            
            # 计算法向量（简单的叉积）
            edge1 = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
            edge2 = (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])
            
            # 叉积计算法向量
            normal = (
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            )
            
            # 归一化法向量
            length = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
            if length > 0:
                normal = (normal[0]/length, normal[1]/length, normal[2]/length)
            
            # 写入法向量（12字节，3个float）
            f.write(struct.pack('<fff', *normal))
            
            # 写入3个顶点（36字节，9个float）
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<fff', *v3))
            
            # 写入属性字节计数（2字节，通常为0）
            f.write(struct.pack('<H', 0))
    
    print(f"转换完成：{ascii_file} -> {binary_file}")
    print(f"三角形数量：{num_triangles}")

if __name__ == "__main__":
    ascii_file = "/home/yhx/workspace/3D-Diffusion-Policy/third_party/dphand/dphand/assets/objects/box.stl"
    binary_file = "/home/yhx/workspace/3D-Diffusion-Policy/third_party/dphand/dphand/assets/objects/box_binary.stl"
    
    ascii_to_binary_stl(ascii_file, binary_file)
