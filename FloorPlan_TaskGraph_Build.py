import json

# 加载 JSON 文件
with open('FloorPlan1_Final.json', 'r') as file:
    processed_data = json.load(file)

with open('FloorPlan1_graph_3.json', 'r') as file:
    graph_data = json.load(file)

# 创建 objectid 到 node 的映射
objectid_to_node = {node['id']: node for node in graph_data['nodes']}

# 创建 objectid 到 links 的映射
objectid_to_links = {}
for link in graph_data['links']:
    source = link['source']
    target = link['target']
    if source not in objectid_to_links:
        objectid_to_links[source] = []
    if target not in objectid_to_links:
        objectid_to_links[target] = []
    objectid_to_links[source].append(link)
    objectid_to_links[target].append(link)

# 整合数据
integrated_data = []
for task in processed_data:
    objectids = task['objectid'].split()

    nodes_info = [objectid_to_node[objectid] if objectid in objectid_to_node else None for objectid in objectids]

    # 处理 links_info
    if len(objectids) > 1:
        links_info = []
        for objectid in objectids:
            if objectid in objectid_to_links:
                # 筛选出 source 和 target 都在 objectids 中的 links
                valid_links = [link for link in objectid_to_links[objectid] if
                               link['source'] in objectids and link['target'] in objectids]
                links_info.extend(valid_links)
    else:
        links_info = []

    task['nodes_info'] = nodes_info
    if links_info:  # 只有当 links_info 不为空时才保存
        task['links_info'] = links_info
    integrated_data.append(task)

# 示例：打印整合后的第一个任务数据
print(json.dumps(integrated_data[0], indent=4))

# 保存整合后的数据到新的 JSON 文件
output_file = 'Integrated_FloorPlan_Data2.json'
with open(output_file, 'w') as file:
    json.dump(integrated_data, file, indent=4)

print(f"Integrated data has been saved to {output_file}")





