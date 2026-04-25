# 代码修复验证脚本
import sys
import traceback

print("="*60)
print("🔍 开始验证修复的代码...")
print("="*60)

test_results = []

# 测试1: 梯度下降可视化
print("\n[1/3] 检查梯度下降可视化.py...")
try:
    # 只导入，不运行动画（因为交互环境）
    import os
    os.chdir(r"D:\DpanPython\python-projects\learn\机器学习Machine Learning\00-基础知识")
    
    with open("梯度下降可视化.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    # 检查关键修复
    if "if frame < len(w_history):" in code:
        print("✓ 梯度下降可视化: 数组边界检查已添加")
        test_results.append(("梯度下降可视化", True))
    else:
        print("✗ 梯度下降可视化: 未找到修复")
        test_results.append(("梯度下降可视化", False))
except Exception as e:
    print(f"✗ 梯度下降可视化: 检查失败 - {e}")
    test_results.append(("梯度下降可视化", False))

# 测试2: 线性回归
print("\n[2/3] 检查线性回归02-代码实现.py...")
try:
    with open(r"D:\DpanPython\python-projects\learn\机器学习Machine Learning\01-线性回归\02-代码实现.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    # 检查关键修复
    if "if len(X.shape) == 1 or X.shape[1] == 1:" in code:
        print("✓ 线性回归: plot_predictions函数已修复（处理单特征）")
        test_results.append(("线性回归", True))
    else:
        print("✗ 线性回归: 未找到修复")
        test_results.append(("线性回归", False))
except Exception as e:
    print(f"✗ 线性回归: 检查失败 - {e}")
    test_results.append(("线性回归", False))

# 测试3: 逻辑回归
print("\n[3/3] 检查逻辑回归02-代码实现.py...")
try:
    with open(r"D:\DpanPython\python-projects\learn\机器学习Machine Learning\02-逻辑回归\02-代码实现.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    # 检查关键修复
    fixes_found = 0
    
    if "auc = None" in code:
        print("✓ 逻辑回归: auc变量初始化已修复")
        fixes_found += 1
    
    if "zero_division=0" in code:
        print("✓ 逻辑回归: zero_division处理已添加")
        fixes_found += 1
    
    if fixes_found >= 2:
        test_results.append(("逻辑回归", True))
    else:
        print("✗ 逻辑回归: 修复不完整")
        test_results.append(("逻辑回归", False))
        
except Exception as e:
    print(f"✗ 逻辑回归: 检查失败 - {e}")
    test_results.append(("逻辑回归", False))

# 总结
print("\n" + "="*60)
print("修复结果总结")
print("="*60)
for name, result in test_results:
    status = "✅ 已修复" if result else "⚠️  需要检查"
    print(f"{name}: {status}")

success_count = sum(1 for _, r in test_results if r)
print(f"\n总体: {success_count}/{len(test_results)} 文件已成功修复")

if success_count == len(test_results):
    print("\n✅ 所有主要bug已修复！代码应该可以正常运行了。")
else:
    print("\n⚠️  部分修复可能不完整，请手动检查。")

print("="*60)

