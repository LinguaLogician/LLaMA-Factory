# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: arrow_pushing_diagrams.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/13 16:14
# https://chat.deepseek.com/a/chat/s/1bd9a402-7bfc-47ad-85e6-92fc1a8b5061

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DCairo
from PIL import Image, ImageDraw
import math

# 1. 创建反应物和产物的分子对象
reactant_smiles = '[OH-].[CH3:1][Cl:2]'
product_smiles = 'C[OH:1].[Cl-:2]'

reactant_mol = Chem.MolFromSmiles(reactant_smiles)
product_mol = Chem.MolFromSmiles(product_smiles)

# 为了对齐，最好生成一个反应对象
rxn = AllChem.ChemicalReaction()
rxn.AddReactantTemplate(Chem.MolFromSmiles('[OH-]'))
rxn.AddReactantTemplate(Chem.MolFromSmiles('CCl'))
rxn.AddProductTemplate(Chem.MolFromSmiles('CO'))
rxn.AddProductTemplate(Chem.MolFromSmiles('[Cl-]'))
rxn.Initialize()

# 2. 使用RDKit绘制反应，但不画箭头
# 设置一个较大的画布
d = Draw.MolDraw2DCairo(800, 400)
d.DrawReaction(rxn)
d.FinishDrawing()
png_data = d.GetDrawingText()

# 将RDKit的绘图保存为PIL图像
with open('reaction_no_arrows.png', 'wb') as f:
    f.write(png_data)
img = Image.open('reaction_no_arrows.png')
draw = ImageDraw.Draw(img)

# 3. 定义计算箭头坐标的函数
# 这是一个关键且复杂的步骤，通常需要手动调整或通过子结构匹配自动计算
def get_atom_coords(mol, atom_idx, d):
    """获取指定原子在绘图中的坐标"""
    # 注意：这里我们直接从Drawer的坐标系统中获取
    # 在实际应用中，可能需要更精确的方法，例如使用Conformer的坐标并转换为2D像素坐标
    # 这里我们使用一个简化的方法：通过分子图的中心来估算
    # 更可靠的方法是使用 d.GetDrawCoords(atom_idx)
    conf = mol.GetConformer()
    pos = conf.GetAtomPosition(atom_idx)
    # 由于RDKit的绘图坐标系与我们的图像坐标系可能不同，这里需要转换
    # 我们假设Drawer已经为我们做好了转换，我们直接使用它的函数
    # 但请注意，在反应绘图中，直接获取单个原子的坐标比较棘手。
    # 一个替代方案是：分别绘制反应物和产物，然后手动排列，这样更容易控制坐标。

# 由于直接从反应图中获取精确原子坐标比较复杂，我们采用替代方案：
# 分别绘制反应物和产物，并手动排列，这样可以精确控制每个原子的位置。

# --- 替代方案：分别绘制分子并手动组合 ---

# 定义分子和重要的原子索引（需要提前知道）
# [OH-] : O index 0
# CH3Cl: C index 0, Cl index 1
# CH3OH: C index 0, O index 1
# [Cl-] : Cl index 0

mol_oh = Chem.MolFromSmiles('[OH-]')
mol_ch3cl = Chem.MolFromSmiles('CCl')
mol_ch3oh = Chem.MolFromSmiles('CO')
mol_cl = Chem.MolFromSmiles('[Cl-]')

# 为所有分子生成2D坐标
AllChem.Compute2DCoords(mol_oh)
AllChem.Compute2DCoords(mol_ch3cl)
AllChem.Compute2DCoords(mol_ch3oh)
AllChem.Compute2DCoords(mol_cl)

# 获取原子坐标（基于2D构象）
def get_2d_coords(mol, atom_idx):
    conf = mol.GetConformer()
    return conf.GetAtomPosition(atom_idx)

# 假设我们这样排列分子：[OH-] + CH3Cl -> CH3OH + [Cl-]
# 我们为每个分子定义一个在画布上的偏移量，以便排列它们
offset_oh = (50, 200)
offset_ch3cl = (150, 200)
offset_ch3oh = (350, 200)
offset_cl = (550, 200)

# 创建一个新的空白图像
img_combined = Image.new('RGB', (800, 400), 'white')
draw_combined = ImageDraw.Draw(img_combined)

# 绘制每个分子（这里简化了，实际需要用RDKit分别绘制每个分子并粘贴到正确位置）
# 为了示例，我们直接在一个大画布上绘制整个反应，然后基于此添加箭头。
# 我们回到最初的反应图，但手动定义箭头的起点和终点（基于像素坐标估算）。

# 4. 手动定义箭头坐标（在实际应用中，这部分需要根据原子坐标自动计算）
# 这些坐标是通过查看生成的“reaction_no_arrows.png”图像估算的像素坐标。
# 箭头1：从OH-的O的孤对电子到CH3Cl的C
start_arrow1 = (180, 180)  # O原子附近
end_arrow1 = (280, 180)    # C原子附近

# 箭头2：从C-Cl键到Cl离去基团
start_arrow2 = (320, 180)  # C-Cl键中间
end_arrow2 = (420, 180)    # Cl原子附近

# 5. 绘制弯箭头的函数
def draw_curved_arrow(draw, start, end, curvature=0.5, color='blue', width=2):
    """绘制一条弯曲的箭头。"""
    # 计算控制点以创建曲线
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    # 垂直于直线的方向
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0:
        return
    dx, dy = dx/length, dy/length

    # 控制点偏移
    offset_x = -dy * curvature * length * 0.5
    offset_y = dx * curvature * length * 0.5

    control_x = mid_x + offset_x
    control_y = mid_y + offset_y

    # 使用二次贝塞尔曲线绘制弯箭头
    # 由于PIL的ImageDraw不支持直接画贝塞尔曲线箭头，我们画一个简单的曲线代替。
    # 更复杂的实现可以使用多个点来模拟贝塞尔曲线。
    steps = 20
    points = []
    for i in range(steps + 1):
        t = i / steps
        # 二次贝塞尔曲线公式
        x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * end[0]
        y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * end[1]
        points.append((x, y))

    # 绘制曲线
    for i in range(len(points)-1):
        draw.line([points[i], points[i+1]], fill=color, width=width)

    # 绘制箭头头部（简化版，在终点画一个小三角形）
    head_length = 10
    # 箭头方向是最后一段的方向
    dx_head = points[-1][0] - points[-2][0]
    dy_head = points[-1][1] - points[-2][1]
    norm = math.sqrt(dx_head**2 + dy_head**2)
    if norm > 0:
        dx_head, dy_head = dx_head/norm, dy_head/norm
        # 计算垂直方向
        perp_x = -dy_head
        perp_y = dx_head
        # 箭头三角形的点
        arrow1 = (end[0] - dx_head*head_length + perp_x*head_length*0.5,
                  end[1] - dy_head*head_length + perp_y*head_length*0.5)
        arrow2 = (end[0] - dx_head*head_length - perp_x*head_length*0.5,
                  end[1] - dy_head*head_length - perp_y*head_length*0.5)
        draw.polygon([end, arrow1, arrow2], fill=color)

# 6. 在图像上绘制箭头
draw_curved_arrow(draw, start_arrow1, end_arrow1, curvature=0.5, color='blue')
draw_curved_arrow(draw, start_arrow2, end_arrow2, curvature=-0.5, color='red') # 反向弯曲

# 7. 保存最终图像
img.save('reaction_with_arrows.png')
print("反应机理图已保存为 'reaction_with_arrows.png'")