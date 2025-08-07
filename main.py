#!/usr/bin/env python3
"""
Multi-Agent Data Analysis Assistant

A comprehensive data analysis system powered by multiple AI agents working together
to provide intelligent data processing, analysis, visualization, and insights.

Authors: AI Assistant
Version: 1.0.0
Date: 2024
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from mcp_agent.app import MCPApp
from orchestrator import DataAnalysisOrchestrator, AnalysisRequest
from agents import DataProcessorAgent, CodeGeneratorAgent, CodeExecutorAgent, InsightGeneratorAgent


def print_banner():
    """打印启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║               🤖 Multi-Agent Data Analysis Assistant 🤖                      ║
    ║                                                                              ║
    ║  A sophisticated AI-powered data analysis system featuring:                  ║
    ║                                                                              ║
    ║  🔹 Intelligent Data Processing & Quality Assessment                         ║
    ║  🔹 AI-Powered Code Generation for Analysis & Visualization                  ║
    ║  🔹 Safe Code Execution in Sandboxed Environment                            ║
    ║  🔹 Advanced Interactive Visualizations                                     ║
    ║  🔹 Business Intelligence & Actionable Insights                             ║
    ║  🔹 Multi-Agent Coordination & Task Management                              ║
    ║                                                                              ║
    ║  Supported Formats: CSV, Excel (.xlsx, .xls)                               ║
    ║  Powered by: MCP (Model Context Protocol) + OpenAI GPT-4                   ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


async def test_system():
    """测试系统功能"""
    print("\n🧪 Running System Tests...")
    
    try:
        # 初始化MCP应用
        print("\n1️⃣ Initializing MCP Application...")
        app = MCPApp(name="data_analysis_test")
        await app.initialize()
        print("   ✅ MCP Application initialized successfully")
        
        # 初始化协调器
        print("\n2️⃣ Initializing Multi-Agent Orchestrator...")
        orchestrator = DataAnalysisOrchestrator()
        await orchestrator.initialize()
        print("   ✅ All agents initialized successfully")
        
        # 测试各个智能体
        print("\n3️⃣ Testing Individual Agents...")
        
        # 测试数据处理智能体
        print("   🔍 Testing Data Processor Agent...")
        data_processor = DataProcessorAgent()
        await data_processor.initialize()
        print("   ✅ Data Processor Agent ready")
        
        # 测试代码生成智能体
        print("   🧠 Testing Code Generator Agent...")
        code_generator = CodeGeneratorAgent()
        await code_generator.initialize()
        print("   ✅ Code Generator Agent ready")
        
        # 测试代码执行智能体
        print("   ⚡ Testing Code Executor Agent...")
        code_executor = CodeExecutorAgent()
        await code_executor.initialize()
        print("   ✅ Code Executor Agent ready")
        
        # 测试洞察生成智能体
        print("   💡 Testing Insight Generator Agent...")
        insight_generator = InsightGeneratorAgent()
        await insight_generator.initialize()
        print("   ✅ Insight Generator Agent ready")
        
        print("\n🎉 All system tests passed successfully!")
        print("\n📋 System Components:")
        print("   • Data Processor Agent: Ready for data loading and preprocessing")
        print("   • Code Generator Agent: Ready for intelligent code generation")
        print("   • Code Executor Agent: Ready for safe code execution")
        print("   • Insight Generator Agent: Ready for business intelligence")
        print("   • Orchestrator: Ready for multi-agent coordination")
        
        return True
        
    except Exception as e:
        print(f"\n❌ System test failed: {str(e)}")
        print("\n🔧 Troubleshooting Tips:")
        print("   1. Check if all required packages are installed: pip install -r requirements.txt")
        print("   2. Verify MCP server configurations in mcp_agent.config.yaml")
        print("   3. Ensure OpenAI API key is set in mcp_agent.secrets.yaml")
        print("   4. Check if Node.js and npm are installed for MCP servers")
        return False


async def run_demo_analysis():
    """运行演示分析"""
    print("\n🎬 Running Demo Analysis...")
    
    try:
        # 创建示例数据
        import pandas as pd
        import numpy as np
        
        # 生成示例销售数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        
        demo_data = pd.DataFrame({
            'date': dates,
            'sales': np.random.normal(1000, 200, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 100,
            'customers': np.random.poisson(50, 365),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 365),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
            'marketing_spend': np.random.normal(500, 100, 365)
        })
        
        # 保存示例数据
        demo_file = "demo_sales_data.csv"
        demo_data.to_csv(demo_file, index=False)
        print(f"   📊 Created demo dataset: {demo_file}")
        print(f"   📈 Dataset shape: {demo_data.shape}")
        print(f"   📋 Columns: {list(demo_data.columns)}")
        
        # 初始化系统
        orchestrator = DataAnalysisOrchestrator()
        await orchestrator.initialize()
        
        # 运行示例分析
        demo_questions = [
            "Provide a comprehensive overview of the sales data",
            "Analyze the relationship between marketing spend and sales",
            "Show sales trends over time and identify seasonal patterns"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n🔍 Demo Analysis {i}: {question}")
            
            request = AnalysisRequest(
                user_question=question,
                file_path=demo_file,
                analysis_type="demo"
            )
            
            result = await orchestrator.process_analysis_request(request)
            
            if result['status'] == 'success':
                print(f"   ✅ Analysis completed successfully")
                
                # 显示执行计划摘要
                plan = result.get('execution_plan', {})
                if 'tasks' in plan:
                    print(f"   📋 Executed {len(plan['tasks'])} tasks")
                
                # 显示洞察摘要
                insights = result.get('insights', {})
                if insights.get('insights'):
                    insight_text = insights['insights'][:200] + "..." if len(insights['insights']) > 200 else insights['insights']
                    print(f"   💡 Key Insight: {insight_text}")
            else:
                print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        print("\n🎉 Demo analysis completed successfully!")
        print(f"\n📁 Demo file created: {demo_file}")
        print("   You can now use this file with the Streamlit interface.")
        
    except Exception as e:
        print(f"\n❌ Demo analysis failed: {str(e)}")


def print_usage_instructions():
    """打印使用说明"""
    instructions = """
    
🚀 Getting Started:

1️⃣ Install Dependencies:
   pip install -r requirements.txt

2️⃣ Configure API Keys:
   cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
   # Edit mcp_agent.secrets.yaml and add your OpenAI API key

3️⃣ Install MCP Servers:
   npm install -g @modelcontextprotocol/server-filesystem
   npm install -g @antv/mcp-server-chart
   pip install mcp-server-fetch
   pip install mcp-server-python

4️⃣ Launch the Application:
   streamlit run app.py

📚 Usage Tips:
   • Upload CSV or Excel files through the sidebar
   • Ask natural language questions about your data
   • Use the quick analysis buttons for common tasks
   • Explore the Data Explorer tab for detailed data inspection
   • Check the Visualizations tab for interactive charts
   • Review analysis history in the History tab

🔧 Troubleshooting:
   • Ensure all MCP servers are properly installed
   • Check that your OpenAI API key is valid
   • Verify file permissions for data uploads
   • Check console output for detailed error messages

📖 Example Questions to Ask:
   • "Show me a summary of this dataset"
   • "What are the correlations between variables?"
   • "Create a visualization showing trends over time"
   • "Identify any outliers or anomalies in the data"
   • "Analyze the relationship between X and Y columns"
   • "Generate insights about customer behavior patterns"
    """
    print(instructions)


async def main():
    """主函数"""
    print_banner()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            success = await test_system()
            if success:
                print("\n✅ System is ready! You can now run: streamlit run app.py")
            sys.exit(0 if success else 1)
        
        elif command == "demo":
            await run_demo_analysis()
            sys.exit(0)
        
        elif command == "help":
            print_usage_instructions()
            sys.exit(0)
        
        else:
            print(f"\n❌ Unknown command: {command}")
            print("\n📖 Available commands:")
            print("   python main.py test  - Run system tests")
            print("   python main.py demo  - Run demo analysis")
            print("   python main.py help  - Show usage instructions")
            print("   streamlit run app.py - Launch the web interface")
            sys.exit(1)
    
    # 默认行为：显示使用说明
    print("\n🎯 Welcome to the Multi-Agent Data Analysis Assistant!")
    print("\n📋 Quick Start Options:")
    print("   🧪 Test the system:     python main.py test")
    print("   🎬 Run demo analysis:   python main.py demo")
    print("   🚀 Launch web app:      streamlit run app.py")
    print("   📖 Show help:           python main.py help")
    
    # 检查基本依赖
    try:
        import streamlit
        import pandas
        import plotly
        print("\n✅ Core dependencies detected")
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("   Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # 检查配置文件
    config_file = Path("mcp_agent.config.yaml")
    secrets_file = Path("mcp_agent.secrets.yaml")
    
    if not config_file.exists():
        print(f"\n⚠️  Configuration file not found: {config_file}")
    else:
        print(f"\n✅ Configuration file found: {config_file}")
    
    if not secrets_file.exists():
        print(f"\n⚠️  Secrets file not found: {secrets_file}")
        print("   Please copy mcp_agent.secrets.yaml.example to mcp_agent.secrets.yaml")
        print("   and add your API keys")
    else:
        print(f"\n✅ Secrets file found: {secrets_file}")
    
    print("\n🎉 Ready to start! Choose one of the options above.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using the Multi-Agent Data Analysis Assistant!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)