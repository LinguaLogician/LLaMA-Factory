# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: smiles_reaction.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/15 15:43

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import MolDrawOptions

# 定义反应
reaction_smiles = 'C1=CC=CC=C1C(=O)O.NCCO>>C1=CC=CC=C1C(=O)NCCO'
rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles)

# 创建极简绘制选项
draw_options = Draw.MolDrawOptions()
draw_options.useBWAtomPalette()  # 黑白原子
draw_options.bondLineWidth = 1.5  # 键线宽度
draw_options.minFontSize = 12     # 最小字体大小
draw_options.maxFontSize = 14     # 最大字体大小
draw_options.additionalAtomLabelPadding = 0.0  # 减少原子标签间距
draw_options.backgroundColour = (1, 1, 1)  # 白色背景

# 隐藏碳原子上的氢原子（更简洁的键线式）
for mol in rxn.GetReactants() + rxn.GetProducts():
    Chem.RemoveHs(mol)  # 移除氢原子

# 生成图像
img = Draw.ReactionToImage(rxn,
                          subImgSize=(350, 200),
                          drawOptions=draw_options)

img.save('reaction_simple_bw.png')
print("简化黑白反应图像已保存")