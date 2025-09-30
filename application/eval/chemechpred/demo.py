# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: demo.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/30 14:52

TASKS_CONFIG = {

    "RXTS_TO_PRDS_PLUS": [
        # "UPD.CANO.AM.RXTS->MECH+UPD.CANO.AM.PRDS",
        # "UPD.CANO.AM.RXTS->CLS+UPD.CANO.AM.PRDS",
        "UPD.CANO.AM.RXTS->CLS+MECH+UPD.CANO.AM.PRDS",
        # "UPD.CANO.AM.RXTS+MECH->UPD.CANO.AM.PRDS",
        # "UPD.CANO.AM.RXTS+CLS->UPD.CANO.AM.PRDS",
        # "UPD.CANO.AM.RXTS+CLS->MECH+UPD.CANO.AM.PRDS",
        # "UPD.CANO.AM.RXTS+MECH->CLS+UPD.CANO.AM.PRDS",
        # "UPD.CANO.AM.RXTS+MECH+CLS->UPD.CANO.AM.PRDS"
    ],
}

def search_group(task_id):
    group = None
    task_id = task_id.lower()
    for grp, tasks in TASKS_CONFIG.items():
        for task_tag in tasks:
            # 转换task_tag为task_id格式进行比较
            task_tag_id = task_tag.replace('->', '_TO_').replace('.', '').lower()
            if task_tag_id == task_id:
                group = grp.lower()
                break
        if group:
            break

    if not group:
        raise ValueError(f"无法找到task_id {task_id} 对应的group")
    return group


if __name__ == '__main__':
    print(search_group('updcanoamrxts_to_cls_mech_updcanoamprds'))
