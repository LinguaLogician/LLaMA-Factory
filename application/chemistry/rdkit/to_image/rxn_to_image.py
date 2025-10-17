# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: rxn_to_image.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/8 16:53
# https://chat.deepseek.com/a/chat/s/57741cd7-361b-4f92-b536-82dcf033510e
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from io import BytesIO


def smiles_reaction_to_image(smiles_reaction, img_size=(2000, 800), save_path=None):
    """
    将Atom-Mapped SMILES反应表达式转换为图片

    参数:
    - smiles_reaction: SMILES格式的反应表达式
    - img_size: 图片尺寸，默认为(800, 300)
    - save_path: 图片保存路径，如果为None则不保存

    返回:
    - PIL Image对象
    """
    try:
        # 使用RDKit解析反应
        rxn = AllChem.ReactionFromSmarts(smiles_reaction, useSmiles=True)

        # 设置反应显示选项
        d2d = Draw.MolDraw2DCairo(img_size[0], img_size[1])
        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()

        # 获取图片数据
        img_data = d2d.GetDrawingText()

        # 转换为PIL Image
        from PIL import Image
        img = Image.open(BytesIO(img_data))

        # 如果提供了保存路径，则保存图片
        if save_path:
            img.save(save_path)
            print(f"反应图片已保存至: {save_path}")

        return img

    except Exception as e:
        print(f"错误: {e}")
        return None


def display_reaction_image(smiles_reaction, img_size=(1500, 600)):
    """
    显示反应图片（适用于Jupyter notebook）

    参数:
    - smiles_reaction: SMILES格式的反应表达式
    - img_size: 图片尺寸
    """
    img = smiles_reaction_to_image(smiles_reaction, img_size)
    if img:
        plt.figure(figsize=(img_size[0] / 100, img_size[1] / 100))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

#
# # 完整的用例示例
# if __name__ == "__main__":
#     # 示例1: 简单的酯化反应（带原子映射）
#     esterification_reaction = "[CH3:1][OH:2].[C:3](=[O:4])[OH:5]>>[CH3:1][O:2][C:3](=[O:4])[O:5]"
#
#     print("示例1: 酯化反应")
#     img1 = smiles_reaction_to_image(esterification_reaction, save_path="esterification_reaction.png")
#
#     # 示例2: Diels-Alder反应（带原子映射）
#     diels_alder_reaction = "[C:1]1=[C:2][C:3]=[C:4][C:5]=[C:6]1.[C:7]#[C:8]>>[C:1]12[C:2][C:3][C:4][C:5]1[C:6]2[C:7]=[C:8]"
#
#     print("\n示例2: Diels-Alder反应")
#     img2 = smiles_reaction_to_image(diels_alder_reaction, save_path="diels_alder_reaction.png")
#
#     # 示例3: 酰胺形成反应（带原子映射）
#     amide_formation = "[NH2:1][C:2]([H:3])([H:4])[C:5](=[O:6])[OH:7].[CH3:8][C:9](=[O:10])[OH:11]>>[NH:1][C:2]([H:3])([H:4])[C:5](=[O:6])[N:12][C:9]([CH3:8])=[O:10]"
#
#     print("\n示例3: 酰胺形成反应")
#     img3 = smiles_reaction_to_image(amide_formation, save_path="amide_formation.png")
#
#     # 在Jupyter notebook中显示图片
#     # display_reaction_image(esterification_reaction)
#
#     print("\n所有反应图片已生成完成！")


# 高级用法：批量处理多个反应
def batch_reactions_to_images(reaction_dict, output_dir="reaction_images"):
    """
    批量处理多个反应并保存为图片

    参数:
    - reaction_dict: 字典，键为反应名称，值为SMILES反应表达式
    - output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name, reaction_smiles in reaction_dict.items():
        print(f"处理反应: {name}")
        save_path = os.path.join(output_dir, f"{name}.png")
        img = smiles_reaction_to_image(reaction_smiles, save_path=save_path)
        results[name] = img

    return results


# 批量处理示例
if __name__ == "__main__":
    reactions = {
        # "esterification": "[CH3:1][OH:2].[C:3](=[O:4])[OH:5]>>[CH3:1][O:2][C:3](=[O:4])[O:5]",
        # "diels_alder": "[C:1]1=[C:2][C:3]=[C:4][C:5]=[C:6]1.[C:7]#[C:8]>>[C:1]12[C:2][C:3][C:4][C:5]1[C:6]2[C:7]=[C:8]",
        # "amide_formation": "[NH2:1][C:2]([H:3])([H:4])[C:5](=[O:6])[OH:7].[CH3:8][C:9](=[O:10])[OH:11]>>[NH:1][C:2]([H:3])([H:4])[C:5](=[O:6])[N:12][C:9]([CH3:8])=[O:10]",
        # "prediction": "CC(C)(C)OC(=O)[NH:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:10])[cH:11][cH:12][cH:13][c:14]2[CH:15]1[CH2:16][C:17](=[O:18])[OH:19]>>[NH2:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:10])[cH:11][cH:12][cH:13][c:14]2[CH:15]1[CH2:16][C:17](=[O:18])[OH:19]",
        # "ground_truth": "CC(C)(C)OC(=O)[NH:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:10])[cH:11][cH:12][cH:13][c:14]2[CH:15]1[CH2:16][C:17](=[O:18])[OH:19]>>[NH3+:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:10])[cH:11][cH:12][cH:13][c:14]2[CH:15]1[CH2:16][C:17](=[O:18])[OH:19]"
        # "predicted_rxn3": "[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH2:7][CH3:8])[c:9]1[N:10]=[C:11]=[O:12].[cH:13]1[cH:14][cH:15][c:16]([CH:17]2[CH2:18][CH2:19][CH:20]([NH:21][CH2:22][CH:23]3[CH2:24][CH2:25][CH2:26][CH2:27][CH2:28]3)[CH2:29][CH2:30]2)[cH:31][cH:32]1>>[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH2:7][CH3:8])[c:9]1[NH:10][C:11](=[O:12])[N:21]([CH:20]1[CH2:19][CH2:18][CH:17]([c:16]2[cH:15][cH:14][cH:13][cH:32][cH:31]2)[CH2:30][CH2:29]1)[CH2:22][CH:23]1[CH2:24][CH2:25][CH2:26][CH2:27][CH2:28]1",
        # "ground_truth_rxn3": "[CH3:1][CH2:2][c:3]1[cH:4][cH:5][cH:6][c:7]([CH2:8][CH3:9])[c:10]1[N:11]=[C:12]=[O:13].[cH:14]1[cH:15][cH:16][c:17]([CH:18]2[CH2:19][CH2:20][CH:21]([NH:22][CH2:23][CH:24]3[CH2:25][CH2:26][CH2:27][CH2:28][CH2:29]3)[CH2:30][CH2:31]2)[cH:32][cH:33]1>>[CH3:1][CH2:2][c:3]1[cH:4][cH:5][cH:6][c:7]([CH2:8][CH3:9])[c:10]1[NH:11][C:12](=[O:13])[N:22]([CH:21]1[CH2:20][CH2:19][CH:18]([c:17]2[cH:16][cH:15][cH:14][cH:33][cH:32]2)[CH2:31][CH2:30]1)[CH2:23][CH:24]1[CH2:25][CH2:26][CH2:27][CH2:28][CH2:29]1"
        # "ori_am_rxn": "[CH3:10][N:11]([CH3:12])[CH2:13][C:14]1([c:15]2[cH:16][cH:17][c:18]([OH:19])[cH:20][cH:21]2)[CH2:22][CH2:23][O:24][CH2:25][CH2:26]1.Cl[CH2:1][CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][S:7][CH2:8][CH2:9]1>>[CH2:1]([CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][S:7][CH2:8][CH2:9]1)[O:19][c:18]1[cH:17][cH:16][c:15]([C:14]2([CH2:13][N:11]([CH3:10])[CH3:12])[CH2:22][CH2:23][O:24][CH2:25][CH2:26]2)[cH:21][cH:20]1",
        #  "upd_am_rxn": "[CH3:10][N:11]([CH3:12])[CH2:13][C:14]1([c:15]2[cH:16][cH:17][c:18]([OH:19])[cH:20][cH:21]2)[CH2:22][CH2:23][O:24][CH2:25][CH2:26]1.[Cl:101][CH2:1][CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][S:7][CH2:8][CH2:9]1.[H-:301].[Na+]>>[CH2:1]([CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][S:7][CH2:8][CH2:9]1)[O:19][c:18]1[cH:17][cH:16][c:15]([C:14]2([CH2:13][N:11]([CH3:10])[CH3:12])[CH2:22][CH2:23][O:24][CH2:25][CH2:26]2)[cH:21][cH:20]1",
        # "ori_std_rxn": "CN(C)CC1(c2ccc(O)cc2)CCOCC1.ClCCCN1CCSCC1>>CN(C)CC1(c2ccc(OCCCN3CCSCC3)cc2)CCOCC1",
        # "upd_std_rxn": "CN(C)CC1(c2ccc(O)cc2)CCOCC1.ClCCCN1CCSCC1.[H-].[Na+]>>CN(C)CC1(c2ccc(OCCCN3CCSCC3)cc2)CCOCC1",
        # "rxn1": "CCO.O=C(O)O>[H+]>CCOC(=O)O.O",
        # "rxn2": "[CH3:1][CH2:2][OH:3].[CH3:4][C:5](=[O:6])[OH:7]>>[CH3:1][CH2:2][O:3][C:5](=[O:6])[CH3:4].[O:7][H:8]",
        # "rxn3": "C=CC=C.C=C>>C1C=CCCC1",
        # "rxn4": "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]>>[CH2:1]1[CH:2]=[CH:3][CH:4][CH:5][CH2:6]1",
        # "rxn": "[Br:101][CH2:1][c:2]1[cH:3][cH:4][c:5](-[c:6]2[cH:7][cH:8][c:9]([CH2:10][CH2:11][CH3:12])[cH:13][c:14]2[F:15])[cH:16][c:17]1[F:18].[OH:19][c:20]1[cH:21][cH:22][c:23]([F:24])[c:25]([F:26])[cH:27]1.[H-:301].[Na+]>>[CH2:1]([c:2]1[cH:3][cH:4][c:5](-[c:6]2[cH:7][cH:8][c:9]([CH2:10][CH2:11][CH3:12])[cH:13][c:14]2[F:15])[cH:16][c:17]1[F:18])[O:19][c:20]1[cH:21][cH:22][c:23]([F:24])[c:25]([F:26])[cH:27]1"
        "rxn_demo1": "[Br-:40].[CH2:43]1[O:44][CH2:45][CH2:46][CH2:47]1.[CH3:41][Mg+:42].[Cl:1][c:2]1[cH:3][c:4]([C:12](=[O:13])[NH:14][CH:15]([CH2:16][CH2:17][C:18](=[O:19])[OH:20])[CH2:21][c:22]2[cH:23][cH:24][c:25](-[c:28]3[n:29][c:30]([C:34](=[O:35])[N:36]([CH3:37])[O:38][CH3:39])[n:31]([CH3:33])[cH:32]3)[cH:26][cH:27]2)[cH:5][cH:6][c:7]1[O:8][CH:9]([CH3:10])[CH3:11]>>[Cl:1][c:2]1[cH:3][c:4]([C:12](=[O:13])[NH:14][CH:15]([CH2:16][CH2:17][C:18](=[O:19])[OH:20])[CH2:21][c:22]2[cH:23][cH:24][c:25](-[c:28]3[n:29][c:30]([C:34](=[O:35])[CH3:41])[n:31]([CH3:33])[cH:32]3)[cH:26][cH:27]2)[cH:5][cH:6][c:7]1[O:8][CH:9]([CH3:10])[CH3:11]"
    }

    batch_results = batch_reactions_to_images(reactions)
    print(f"批量处理完成，共处理 {len(batch_results)} 个反应")
