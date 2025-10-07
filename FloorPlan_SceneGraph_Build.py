import json
import networkx as nx
import math
import re


# 计算两个物品之间的欧氏距离
def calculate_distance(pos1, pos2):
    return math.sqrt((pos1['x'] - pos2['x']) ** 2 + (pos1['y'] - pos2['y']) ** 2 + (pos1['z'] - pos2['z']) ** 2)


# 读取JSON文件并构建图
def build_graph_from_json(file_path):
    # 创建一个新的图
    G = nx.Graph()

    # 读取JSON文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 添加节点
    for item_name, instances in data.items():
        for instance in instances:
            node_id = f"{item_name}|{instance['objectId']}"
            G.add_node(node_id, **instance)

    # 添加边，考虑新的关系
    for node1 in G.nodes(data=True):
        for node2 in G.nodes(data=True):
            if node1 != node2:
                dist = calculate_distance(node1[1]['position'], node2[1]['position'])

                # 添加基于空间接近性的边
                if dist < 1.0:
                    G.add_edge(node1[0], node2[0], relationship='spatial')

                # 添加容器和可拾起物品的关系
                if node1[1].get('receptacle', False) and node2[1].get('pickupable', False):
                    G.add_edge(node1[0], node2[0], relationship='receptacle-pickupable')

                # 添加可清洁与清洁工具的关系
                if node1[1].get('dirtyable', False) and node2[1].get('canClean', True):
                    G.add_edge(node1[0], node2[0], relationship='canClean-dirtyable')

                # 添加烹饪设备与可烹饪物品的关系
                if node1[1].get('heatingDevice', False) and node2[1].get('cookable', True):
                    G.add_edge(node1[0], node2[0], relationship='heatingDevice-cookable')

                # 添加切片工具与可切片物品的关系
                if node1[1].get('hasBlade', True) and node2[1].get('sliceable', True):
                    G.add_edge(node1[0], node2[0], relationship='blade-sliceable')

                # 考虑开关关系
                if node1[1].get('toggleable', True) and node2[1].get('canToggle', True):
                    G.add_edge(node1[0], node2[0], relationship='toggleable-canToggle')

    return G


# 保存图到 JSON 文件的函数
def save_graph_to_json(graph, output_file):
    # 收集节点数据，包括 objectId 和其它所有属性
    data = {
        'nodes': [
            {
                'id': node_data['objectId'], **{key: value for key, value in node_data.items() if key != 'objectId'}
                # 除objectId外的所有属性
            } for _, node_data in graph.nodes(data=True)
        ],
        'links': [
            {
                'source': graph.nodes[u]['objectId'],
                'target': graph.nodes[v]['objectId'],
                'relationship': attr['relationship']
            }
            for u, v, attr in graph.edges(data=True)
        ]
    }

    # 将数据写入 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def process_graph_data(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Process nodes
    for node in data['nodes']:
        node['id'] = re.sub(r'[^A-Za-z]+', '', node['id'])

    # Process links
    for link in data['links']:
        link['source'] = re.sub(r'[^A-Za-z]+', '', link['source'])
        link['target'] = re.sub(r'[^A-Za-z]+', '', link['target'])

    # Write processed data back to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def clean_links(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Remove links where source and target are the same
    cleaned_links = [link for link in data['links'] if link['source'] != link['target']]
    data['links'] = cleaned_links

    # Write the modified data back to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


# 主程序执行流程
if __name__ == "__main__":
    # 构建图并保存初始图
    input_json_path = 'FloorPlan1_info.json'  # 原始输入文件路径
    graph = build_graph_from_json(input_json_path)
    save_graph_to_json(graph, 'FloorPlan1_graph.json')

    # 处理节点和链接ID
    process_graph_data('FloorPlan1_graph.json', 'FloorPlan1_graph_1.json')

    # 清理相同源目标的链接
    clean_links('FloorPlan1_graph_1.json', 'FloorPlan1_graph_2.json')

    # 仅保留节点的ID属性（修正后）
    with open('FloorPlan1_graph_2.json', 'r') as file:
        data = json.load(file)

    for node in data['nodes']:
        # 先提取原始id
        node_id = node.get('id', '')
        # 清空节点后重新写入id
        node.clear()
        node.update({'id': node_id})

    with open('FloorPlan1_graph_3.json', 'w') as file:
        json.dump(data, file, indent=4)

    print("所有处理步骤已完成，最终文件为 FloorPlan1_graph_3.json")




