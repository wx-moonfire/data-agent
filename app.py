import streamlit as st
import pandas as pd
import asyncio
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mcp_agent.app import MCPApp
from orchestrator import DataAnalysisOrchestrator, AnalysisRequest
from agents import DataContext

# 页面配置
st.set_page_config(
    page_title="🤖 Multi-Agent Data Analysis Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.agent-status {
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin: 0.25rem 0;
    font-weight: bold;
}

.agent-active {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.agent-inactive {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.task-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
}

.insight-box {
    background-color: #e7f3ff;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #0066cc;
    margin: 1rem 0;
}

.error-box {
    background-color: #ffe6e6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff4444;
    margin: 0.5rem 0;
}

.success-box {
    background-color: #e6ffe6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #44ff44;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


class DataAnalysisApp:
    """数据分析应用主类"""
    
    def __init__(self):
        self.orchestrator: Optional[DataAnalysisOrchestrator] = None
        self.mcp_app: Optional[MCPApp] = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """初始化会话状态"""
        if 'orchestrator_initialized' not in st.session_state:
            st.session_state.orchestrator_initialized = False
        
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        
        if 'uploaded_file_path' not in st.session_state:
            st.session_state.uploaded_file_path = None
        
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "👋 Hello! I'm your Multi-Agent Data Analysis Assistant. Upload a CSV or Excel file to get started!"}
            ]
    
    async def initialize_orchestrator(self):
        """初始化协调器"""
        if not st.session_state.orchestrator_initialized:
            with st.spinner("🚀 Initializing Multi-Agent System..."):
                try:
                    self.mcp_app = MCPApp(name="data_analysis_app")
                    await self.mcp_app.initialize()
                    
                    self.orchestrator = DataAnalysisOrchestrator()
                    await self.orchestrator.initialize()
                    
                    st.session_state.orchestrator_initialized = True
                    st.success("✅ Multi-Agent System initialized successfully!")
                    return True
                except Exception as e:
                    st.error(f"❌ Failed to initialize system: {str(e)}")
                    return False
        return True
    
    def render_header(self):
        """渲染页面头部"""
        st.markdown('<div class="main-header">🤖 Multi-Agent Data Analysis Assistant</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Welcome to the **Multi-Agent Data Analysis Assistant**! This intelligent system uses multiple specialized AI agents 
        to provide comprehensive data analysis, visualization, and business insights.
        
        **🎯 Key Features:**
        - 📊 **Smart Data Processing**: Automatic data loading, cleaning, and quality assessment
        - 🧠 **Intelligent Code Generation**: AI-powered Python code creation for analysis and visualization
        - ⚡ **Safe Code Execution**: Secure sandbox environment for running generated code
        - 📈 **Advanced Visualizations**: Interactive charts and graphs using Plotly and Matplotlib
        - 💡 **Business Insights**: AI-generated insights and actionable recommendations
        - 🤖 **Multi-Agent Coordination**: Specialized agents working together seamlessly
        """)
    
    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # 系统状态
            st.subheader("🔧 System Status")
            if st.session_state.orchestrator_initialized:
                st.markdown('<div class="agent-status agent-active">🟢 System Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="agent-status agent-inactive">🔴 System Not Ready</div>', unsafe_allow_html=True)
            
            # 数据上传
            st.subheader("📂 Data Upload")
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your data file to start analysis"
            )
            
            if uploaded_file is not None:
                self.handle_file_upload(uploaded_file)
            
            # 当前数据信息
            if st.session_state.current_data is not None:
                st.subheader("📊 Current Dataset")
                data_info = st.session_state.current_data
                st.write(f"**Shape:** {data_info.get('shape', 'Unknown')}")
                st.write(f"**Columns:** {len(data_info.get('columns', []))}")
                
                with st.expander("View Column Details"):
                    if 'columns' in data_info:
                        for col in data_info['columns']:
                            st.write(f"• {col}")
            
            # 分析历史
            st.subheader("📚 Analysis History")
            if st.session_state.analysis_history:
                st.write(f"**Total Analyses:** {len(st.session_state.analysis_history)}")
                
                if st.button("🗑️ Clear History"):
                    st.session_state.analysis_history = []
                    st.rerun()
            else:
                st.write("No analyses yet")
            
            # 快速分析选项
            if st.session_state.current_data is not None:
                st.subheader("⚡ Quick Analysis")
                
                quick_options = [
                    "📊 Data Overview and Summary",
                    "📈 Show trends and patterns",
                    "🔗 Correlation analysis",
                    "📉 Distribution analysis",
                    "🎯 Outlier detection",
                    "📋 Missing value analysis"
                ]
                
                for option in quick_options:
                    if st.button(option, key=f"quick_{option}"):
                        # 添加到聊天消息
                        question = option.split(" ", 1)[1]  # 移除emoji
                        st.session_state.chat_messages.append({"role": "user", "content": question})
                        st.rerun()
    
    def handle_file_upload(self, uploaded_file):
        """处理文件上传"""
        try:
            # 保存上传的文件
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 读取数据预览
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # 更新会话状态
            st.session_state.uploaded_file_path = file_path
            st.session_state.current_data = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'file_path': file_path,
                'preview': df.head().to_dict('records')
            }
            
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
            st.info(f"📊 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
        except Exception as e:
            st.error(f"❌ Error uploading file: {str(e)}")
    
    def render_main_content(self):
        """渲染主要内容区域"""
        # 检查系统是否已初始化
        if not st.session_state.orchestrator_initialized:
            st.warning("⚠️ Please wait for the system to initialize...")
            return
        
        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Analysis", "📊 Data Explorer", "📈 Visualizations", "📚 History"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_data_explorer()
        
        with tab3:
            self.render_visualizations()
        
        with tab4:
            self.render_analysis_history()
    
    def render_chat_interface(self):
        """渲染聊天界面"""
        st.subheader("💬 Interactive Data Analysis Chat")
        
        # 显示聊天历史
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # 聊天输入
        if prompt := st.chat_input("Ask me anything about your data...", key="chat_input"):
            # 检查是否有数据
            if st.session_state.current_data is None:
                st.error("❌ Please upload a data file first!")
                return
            
            # 添加用户消息
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # 处理分析请求
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing your request..."):
                    response = asyncio.run(self.process_analysis_request(prompt))
                    
                    if response['status'] == 'success':
                        self.display_analysis_results(response)
                        
                        # 添加助手回复
                        insights = response.get('insights', {}).get('insights', 'Analysis completed successfully!')
                        st.session_state.chat_messages.append({"role": "assistant", "content": insights})
                        
                        # 保存到历史
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'question': prompt,
                            'response': response
                        })
                    else:
                        error_msg = f"❌ Analysis failed: {response.get('error', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()
    
    async def process_analysis_request(self, user_question: str) -> Dict[str, Any]:
        """处理分析请求"""
        try:
            request = AnalysisRequest(
                user_question=user_question,
                file_path=st.session_state.uploaded_file_path,
                analysis_type="interactive"
            )
            
            result = await self.orchestrator.process_analysis_request(request)
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def display_analysis_results(self, response: Dict[str, Any]):
        """显示分析结果"""
        # 显示执行计划
        with st.expander("🧠 Execution Plan", expanded=False):
            plan = response.get('execution_plan', {})
            st.write(plan.get('plan_description', 'No plan available'))
            
            if 'tasks' in plan:
                st.write(f"**Total Tasks:** {len(plan['tasks'])}")
                for i, task in enumerate(plan['tasks'], 1):
                    st.markdown(f"**{i}.** {task.get('description', 'Unknown task')}")
        
        # 显示分析结果
        analysis_results = response.get('analysis_results', [])
        if analysis_results:
            st.subheader("📊 Analysis Results")
            
            for i, result in enumerate(analysis_results, 1):
                with st.expander(f"Result {i}: {result.get('task_description', 'Unknown')}", expanded=True):
                    if result.get('status') == 'success':
                        # 显示代码（如果有）
                        if 'code' in result:
                            st.code(result['code'], language='python')
                        
                        # 显示执行结果
                        if 'execution_result' in result:
                            exec_result = result['execution_result']
                            if exec_result.get('stdout'):
                                st.text("Output:")
                                st.text(exec_result['stdout'])
                            
                            if exec_result.get('results'):
                                st.json(exec_result['results'])
                        
                        # 显示其他结果
                        if 'results' in result:
                            st.json(result['results'])
                    else:
                        st.error(f"❌ {result.get('error', 'Unknown error')}")
        
        # 显示洞察
        insights = response.get('insights', {})
        if insights.get('status') == 'success':
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.subheader("💡 Key Insights")
            st.markdown(insights.get('insights', 'No insights available'))
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_data_explorer(self):
        """渲染数据探索器"""
        st.subheader("📊 Data Explorer")
        
        if st.session_state.current_data is None:
            st.info("📂 Please upload a data file to explore")
            return
        
        # 数据预览
        st.subheader("🔍 Data Preview")
        
        try:
            # 重新读取数据以显示
            file_path = st.session_state.uploaded_file_path
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # 基本信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # 数据表格
            st.subheader("📋 Data Table")
            
            # 分页显示
            rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
            total_pages = (len(df) - 1) // rows_per_page + 1
            page = st.selectbox("Page", range(1, total_pages + 1))
            
            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            
            st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
            
            # 列信息
            st.subheader("📊 Column Information")
            
            col_info = []
            for col in df.columns:
                col_info.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null': df[col].count(),
                    'Null': df[col].isnull().sum(),
                    'Unique': df[col].nunique()
                })
            
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True)
            
            # 统计摘要
            st.subheader("📈 Statistical Summary")
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary")
            
        except Exception as e:
            st.error(f"❌ Error exploring data: {str(e)}")
    
    def render_visualizations(self):
        """渲染可视化页面"""
        st.subheader("📈 Data Visualizations")
        
        if st.session_state.current_data is None:
            st.info("📂 Please upload a data file to create visualizations")
            return
        
        try:
            # 读取数据
            file_path = st.session_state.uploaded_file_path
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # 可视化选项
            st.subheader("🎨 Create Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Histogram", "Box Plot", "Scatter Plot", "Line Chart", "Bar Chart", "Correlation Heatmap"]
                )
            
            with col2:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                all_cols = df.columns.tolist()
            
            # 根据图表类型显示不同的选项
            if chart_type == "Histogram":
                if numeric_cols:
                    selected_col = st.selectbox("Select Column", numeric_cols)
                    if st.button("Generate Histogram"):
                        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for histogram")
            
            elif chart_type == "Box Plot":
                if numeric_cols:
                    selected_col = st.selectbox("Select Column", numeric_cols)
                    group_by = st.selectbox("Group By (Optional)", ["None"] + categorical_cols)
                    
                    if st.button("Generate Box Plot"):
                        if group_by != "None":
                            fig = px.box(df, y=selected_col, x=group_by, title=f"Box Plot of {selected_col} by {group_by}")
                        else:
                            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for box plot")
            
            elif chart_type == "Scatter Plot":
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("X-axis", numeric_cols)
                    y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
                    color_by = st.selectbox("Color By (Optional)", ["None"] + categorical_cols)
                    
                    if st.button("Generate Scatter Plot"):
                        if color_by != "None":
                            fig = px.scatter(df, x=x_col, y=y_col, color=color_by, title=f"{y_col} vs {x_col}")
                        else:
                            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for scatter plot")
            
            elif chart_type == "Line Chart":
                if numeric_cols:
                    x_col = st.selectbox("X-axis", all_cols)
                    y_col = st.selectbox("Y-axis", numeric_cols)
                    
                    if st.button("Generate Line Chart"):
                        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for line chart")
            
            elif chart_type == "Bar Chart":
                if categorical_cols and numeric_cols:
                    x_col = st.selectbox("Category Column", categorical_cols)
                    y_col = st.selectbox("Value Column", numeric_cols)
                    
                    if st.button("Generate Bar Chart"):
                        # 聚合数据
                        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                        fig = px.bar(agg_df, x=x_col, y=y_col, title=f"Average {y_col} by {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need both categorical and numeric columns for bar chart")
            
            elif chart_type == "Correlation Heatmap":
                if len(numeric_cols) >= 2:
                    if st.button("Generate Correlation Heatmap"):
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, 
                                      text_auto=True, 
                                      aspect="auto",
                                      title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation heatmap")
            
        except Exception as e:
            st.error(f"❌ Error creating visualization: {str(e)}")
    
    def render_analysis_history(self):
        """渲染分析历史页面"""
        st.subheader("📚 Analysis History")
        
        if not st.session_state.analysis_history:
            st.info("📝 No analysis history yet. Start by asking questions about your data!")
            return
        
        # 显示历史记录
        for i, entry in enumerate(reversed(st.session_state.analysis_history), 1):
            with st.expander(f"Analysis {len(st.session_state.analysis_history) - i + 1}: {entry['question'][:50]}...", expanded=False):
                st.write(f"**Timestamp:** {entry['timestamp']}")
                st.write(f"**Question:** {entry['question']}")
                
                response = entry['response']
                if response.get('status') == 'success':
                    # 显示洞察
                    insights = response.get('insights', {})
                    if insights.get('insights'):
                        st.markdown("**Insights:**")
                        st.markdown(insights['insights'])
                    
                    # 显示执行计划
                    plan = response.get('execution_plan', {})
                    if plan.get('tasks'):
                        st.markdown(f"**Tasks Executed:** {len(plan['tasks'])}")
                        for task in plan['tasks']:
                            st.write(f"• {task.get('description', 'Unknown task')}")
                else:
                    st.error(f"❌ {response.get('error', 'Unknown error')}")
    
    async def run(self):
        """运行应用"""
        # 渲染页面
        self.render_header()
        
        # 初始化系统
        if await self.initialize_orchestrator():
            # 渲染侧边栏和主要内容
            self.render_sidebar()
            self.render_main_content()
        else:
            st.error("❌ Failed to initialize the system. Please check your configuration.")


# 主函数
async def main():
    """主函数"""
    app = DataAnalysisApp()
    await app.run()


if __name__ == "__main__":
    # 运行应用
    asyncio.run(main())