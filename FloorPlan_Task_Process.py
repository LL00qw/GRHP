import json
import re

def process_objectids(objectids):
    """提取objectid中的字母部分并去重"""
    unique_ids = set()
    for oid in objectids:
        name = re.match(r"([A-Za-z]+)", oid)
        if name:
            unique_ids.add(name.group(0))
    return ' '.join(unique_ids)

def process_and_save(input_path, output_path):
    """读取原始JSON，拆分任务条目，处理字段格式，保存新文件"""
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_tasks = []

    # 第一步：拆分 task_desc 和 high_descs
    for task in data:
        task_descs = task['task_desc']
        high_descs = task['high_descs']
        for desc, high in zip(task_descs, high_descs):
            new_task = {
                "floor_plan": task["floor_plan"],
                "task_desc": [desc],
                "high_descs": [high],
                "task_type": task["task_type"],
                "objectid": task["objectid"],
                "actions": task["actions"]
            }
            processed_tasks.append(new_task)

    # 第二步：合并列表为字符串，并清理 objectid
    final_data = []
    for item in processed_tasks:
        final_item = {
            'task_desc': ' '.join(item.get('task_desc', [])),
            'high_descs': ' '.join(desc for sublist in item.get('high_descs', []) for desc in sublist),
            'actions': ' '.join(item.get('actions', [])),
            'objectid': process_objectids(item.get('objectid', []))
        }
        final_data.append(final_item)

    # 保存最终结果
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(final_data, outfile, indent=4, ensure_ascii=False)

    print(f"✅ 处理完成，结果已保存到 '{output_path}'")

if __name__ == '__main__':
    # 输入和输出文件路径
    input_file = 'FloorPlan1.json'       # 原始文件
    output_file = 'FloorPlan1_Final.json'  # 最终输出文件

    process_and_save(input_file, output_file)



