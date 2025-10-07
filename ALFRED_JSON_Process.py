import os
import json
import shutil

def extract_object_ids(data):
    """递归提取所有objectId"""
    object_ids = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'objectId':
                object_ids.append(value)
            else:
                object_ids.extend(extract_object_ids(value))
    elif isinstance(data, list):
        for item in data:
            object_ids.extend(extract_object_ids(item))
    return object_ids

def extract_data_from_json(file_path):
    """从给定的JSON文件中提取所需数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        extracted_data = {
            'floor_plan': data['scene']['floor_plan'],
            'task_desc': [ann['task_desc'] for ann in data['turk_annotations']['anns']],
            'high_descs': [ann['high_descs'] for ann in data['turk_annotations']['anns']],
            'task_type': data['task_type'],
            'objectid': extract_object_ids(data),
            'actions': [action['api_action']['action'] for action in data['plan']['low_actions'] if 'api_action' in action]
        }
        return extracted_data
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def merge_json_files(source_files, target_file):
    """将多个json文件合并到一个json文件中"""
    merged_data = []
    for file_path in source_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data if isinstance(data, list) else [data])
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

def process_directory_extract(source_dir):
    """遍历目录并提取JSON数据"""
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.json'):
                full_path = os.path.join(root, file)
                data = extract_data_from_json(full_path)
                if data:
                    floor_plan = data['floor_plan'].replace(" ", "_").replace("/", "_")
                    save_path = os.path.join(root, f"{floor_plan}.json")
                    if os.path.exists(save_path):
                        with open(save_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                        existing_data.append(data)
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, indent=4, ensure_ascii=False)
                    else:
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump([data], f, indent=4, ensure_ascii=False)

def process_directory_merge(source_dir, target_dir):
    """合并相同floor_plan的JSON文件"""
    floor_plans = {}
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.startswith('FloorPlan') and file.endswith('.json'):
                full_path = os.path.join(root, file)
                floor_plan_name = file.split('.')[0]
                if floor_plan_name not in floor_plans:
                    floor_plans[floor_plan_name] = []
                floor_plans[floor_plan_name].append(full_path)

    for floor_plan, files in floor_plans.items():
        plan_dir = os.path.join(target_dir, floor_plan)
        os.makedirs(plan_dir, exist_ok=True)
        target_file = os.path.join(plan_dir, f"{floor_plan}.json")
        merge_json_files(files, target_file)
        for file in files:
            os.remove(file)

def run_pipeline(source_dir, target_dir):
    """执行完整流程"""
    print("1. 正在提取JSON数据...")
    process_directory_extract(source_dir)
    print("2. 正在合并JSON文件...")
    process_directory_merge(source_dir, target_dir)
    print("✅ 全部处理完成！")

if __name__ == '__main__':
    SOURCE_DIRECTORY = r'D:\课题组资料\Code\alfred\data\json_2.1.0\train'
    TARGET_DIRECTORY = 'data'
    run_pipeline(SOURCE_DIRECTORY, TARGET_DIRECTORY)



