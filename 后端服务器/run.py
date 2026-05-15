#!/usr/bin/env python
"""
乡镇服务反馈平台后端启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_app
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("\n请确保已安装所有依赖:")
    print("pip install -r requirements.txt")
    sys.exit(1)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("乡镇服务反馈平台 - 后端服务器启动")
    print("="*50 + "\n")
    
    try:
        # 创建应用
        app = create_app()
        
        print("✅ 服务器启动成功！")
        print("📡 本地访问: http://localhost:5000")
        print("🔗 API测试: http://localhost:5000/api/auth/test")
        print("\n🔄 按 Ctrl+C 停止服务器")
        print("-"*50)
        
        # 启动服务器
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {str(e)}")
        print("\n🔧 常见问题排查:")
        print("1. 检查MySQL是否运行: 在服务中查看MySQL80状态")
        print("2. 检查数据库连接: mysql -u root -p")
        print("3. 检查密码是否正确: 确认root密码是 root123456")
        print("4. 检查依赖是否安装: pip install -r requirements.txt")
        
        input("\n按Enter键退出...")