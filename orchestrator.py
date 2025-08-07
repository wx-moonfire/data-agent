import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from agents import (
    DataProcessorAgent, CodeGeneratorAgent, CodeExecutorAgent, 
    InsightGeneratorAgent, DataAnalysisTask, DataContext
)
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


class TaskType(Enum):
    """任务类型枚举"""
    DATA_LOADING = "data_loading"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    VISUALIZATION = "visualization"
    INSIGHT_GENERATION = "insight_generation"


@dataclass
class AnalysisRequest:
    """分析请求"""
    user_question: str
    file_path: Optional[str] = None
    analysis_type: str = "general"
    specific_columns: Optional[List[str]] = None
    chart_requirements: Optional[Dict[str, Any]] = None
    priority: int = 1


class DataAnalysisOrchestrator:
    """数据分析协调器 - 管理多智能体协作"""
    
    def __init__(self, name: str = "data_analysis_orchestrator"):
        self.name = name
        self.data_processor = DataProcessorAgent()
        self.code_generator = CodeGeneratorAgent()
        self.code_executor = CodeExecutorAgent()
        self.insight_generator = InsightGeneratorAgent()
        
        # 任务管理
        self.tasks: Dict[str, DataAnalysisTask] = {}
        self.task_counter = 0
        
        # 数据上下文
        self.data_context = DataContext()
        
        # 分析历史
        self.analysis_history: List[Dict[str, Any]] = []
        
        # 协调器智能体
        self.orchestrator_agent = None
        self.orchestrator_llm = None
    
    async def initialize(self):
        """初始化所有智能体"""
        print("🚀 Initializing Data Analysis Orchestrator...")
        
        # 初始化各个智能体
        await self.data_processor.initialize()
        await self.code_generator.initialize()
        await self.code_executor.initialize()
        await self.insight_generator.initialize()
        
        # 初始化协调器智能体
        self.orchestrator_agent = Agent(
            name="orchestrator",
            instruction="""You are the master orchestrator for a multi-agent data analysis system. 
            Your responsibilities include:
            1. Understanding user requests and breaking them down into specific tasks
            2. Determining the optimal sequence of operations
            3. Coordinating between different specialist agents
            4. Ensuring data flow and context sharing between agents
            5. Managing task dependencies and priorities
            6. Providing overall progress updates and final summaries
            
            You work with these specialist agents:
            - DataProcessor: Handles data loading, cleaning, and preprocessing
            - CodeGenerator: Creates Python code for analysis and visualization
            - CodeExecutor: Safely executes generated code
            - InsightGenerator: Analyzes results and provides business insights
            
            Always think step-by-step and coordinate efficiently.""",
            server_names=["filesystem"]
        )
        await self.orchestrator_agent.initialize()
        self.orchestrator_llm = await self.orchestrator_agent.attach_llm(OpenAIAugmentedLLM)
        
        print("✅ All agents initialized successfully!")
    
    async def process_analysis_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """处理分析请求的主入口"""
        print(f"📋 Processing analysis request: {request.user_question}")
        
        try:
            # 1. 生成执行计划
            execution_plan = await self._generate_execution_plan(request)
            
            # 2. 如果需要加载新数据
            if request.file_path and request.file_path != self.data_context.file_path:
                await self._load_data(request.file_path)
            
            # 3. 执行分析计划
            analysis_results = await self._execute_analysis_plan(execution_plan, request)
            
            # 4. 生成最终洞察
            final_insights = await self._generate_final_insights(analysis_results, request)
            
            # 5. 保存到历史记录
            self._save_to_history(request, analysis_results, final_insights)
            
            return {
                'status': 'success',
                'request': asdict(request),
                'execution_plan': execution_plan,
                'analysis_results': analysis_results,
                'insights': final_insights,
                'data_context': self._serialize_data_context()
            }
            
        except Exception as e:
            print(f"❌ Error processing request: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'request': asdict(request)
            }
    
    async def _generate_execution_plan(self, request: AnalysisRequest) -> Dict[str, Any]:
        """生成执行计划"""
        print("🧠 Generating execution plan...")
        
        # 构建上下文信息
        context_info = self._build_context_for_planning(request)
        
        planning_prompt = f"""
        Create a detailed execution plan for the following data analysis request:
        
        User Question: {request.user_question}
        Analysis Type: {request.analysis_type}
        File Path: {request.file_path}
        Specific Columns: {request.specific_columns}
        Chart Requirements: {request.chart_requirements}
        
        Current Data Context:
        {context_info}
        
        Please create a step-by-step execution plan that includes:
        1. Required preprocessing steps (if any)
        2. Specific analysis tasks to perform
        3. Visualization requirements
        4. Expected outputs
        
        Format your response as a structured plan with clear steps, priorities, and dependencies.
        Consider the user's question carefully and break it down into actionable tasks.
        """
        
        plan_response = await self.orchestrator_llm.generate_str(message=planning_prompt)
        
        # 解析计划并创建任务
        tasks = await self._parse_plan_into_tasks(plan_response, request)
        
        return {
            'plan_description': plan_response,
            'tasks': [asdict(task) for task in tasks],
            'total_tasks': len(tasks)
        }
    
    async def _parse_plan_into_tasks(self, plan_description: str, request: AnalysisRequest) -> List[DataAnalysisTask]:
        """将计划解析为具体任务"""
        tasks = []
        
        # 基于用户问题类型创建标准任务流程
        if "visualization" in request.user_question.lower() or "chart" in request.user_question.lower():
            # 可视化任务流程
            tasks.extend([
                self._create_task("data_analysis", "Analyze data structure and identify suitable columns for visualization", 1),
                self._create_task("code_generation", f"Generate visualization code for: {request.user_question}", 2, ["task_1"]),
                self._create_task("code_execution", "Execute visualization code and generate charts", 3, ["task_2"]),
                self._create_task("insight_generation", "Analyze visualization results and provide insights", 4, ["task_3"])
            ])
        elif "correlation" in request.user_question.lower() or "relationship" in request.user_question.lower():
            # 相关性分析任务流程
            tasks.extend([
                self._create_task("data_analysis", "Perform correlation analysis between variables", 1),
                self._create_task("code_generation", "Generate correlation analysis and heatmap code", 2, ["task_1"]),
                self._create_task("code_execution", "Execute correlation analysis", 3, ["task_2"]),
                self._create_task("visualization", "Create correlation heatmap and scatter plots", 4, ["task_3"]),
                self._create_task("insight_generation", "Interpret correlation results and provide insights", 5, ["task_4"])
            ])
        elif "trend" in request.user_question.lower() or "time" in request.user_question.lower():
            # 趋势分析任务流程
            tasks.extend([
                self._create_task("data_analysis", "Identify time-based columns and analyze temporal patterns", 1),
                self._create_task("code_generation", "Generate time series analysis code", 2, ["task_1"]),
                self._create_task("code_execution", "Execute trend analysis", 3, ["task_2"]),
                self._create_task("visualization", "Create time series plots and trend visualizations", 4, ["task_3"]),
                self._create_task("insight_generation", "Analyze trends and provide forecasting insights", 5, ["task_4"])
            ])
        elif "summary" in request.user_question.lower() or "overview" in request.user_question.lower():
            # 数据概览任务流程
            tasks.extend([
                self._create_task("data_analysis", "Generate comprehensive data summary and statistics", 1),
                self._create_task("code_generation", "Generate descriptive statistics code", 2, ["task_1"]),
                self._create_task("code_execution", "Execute summary analysis", 3, ["task_2"]),
                self._create_task("visualization", "Create overview charts (distributions, box plots, etc.)", 4, ["task_3"]),
                self._create_task("insight_generation", "Provide data overview insights and recommendations", 5, ["task_4"])
            ])
        else:
            # 通用分析任务流程
            tasks.extend([
                self._create_task("data_analysis", f"Analyze data to answer: {request.user_question}", 1),
                self._create_task("code_generation", f"Generate analysis code for: {request.user_question}", 2, ["task_1"]),
                self._create_task("code_execution", "Execute analysis code", 3, ["task_2"]),
                self._create_task("insight_generation", "Generate insights based on analysis results", 4, ["task_3"])
            ])
        
        return tasks
    
    def _create_task(self, task_type: str, description: str, priority: int, dependencies: List[str] = None) -> DataAnalysisTask:
        """创建分析任务"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        task = DataAnalysisTask(
            task_id=task_id,
            description=description,
            task_type=task_type,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.tasks[task_id] = task
        return task
    
    async def _execute_analysis_plan(self, execution_plan: Dict[str, Any], request: AnalysisRequest) -> List[Dict[str, Any]]:
        """执行分析计划"""
        print("⚡ Executing analysis plan...")
        
        tasks = [DataAnalysisTask(**task_data) for task_data in execution_plan['tasks']]
        results = []
        
        # 按优先级和依赖关系执行任务
        for task in sorted(tasks, key=lambda t: t.priority):
            print(f"🔄 Executing task: {task.description}")
            
            # 检查依赖是否完成
            if not self._check_dependencies_completed(task, results):
                print(f"⏳ Waiting for dependencies: {task.dependencies}")
                continue
            
            # 执行任务
            task_result = await self._execute_single_task(task, request)
            task_result['task_id'] = task.task_id
            task_result['task_description'] = task.description
            
            results.append(task_result)
            
            # 更新任务状态
            if task_result['status'] == 'success':
                task.status = 'completed'
                task.result = task_result
                print(f"✅ Task completed: {task.description}")
            else:
                task.status = 'failed'
                task.error = task_result.get('error', 'Unknown error')
                print(f"❌ Task failed: {task.description} - {task.error}")
        
        return results
    
    async def _execute_single_task(self, task: DataAnalysisTask, request: AnalysisRequest) -> Dict[str, Any]:
        """执行单个任务"""
        try:
            if task.task_type == "data_analysis":
                return await self._execute_data_analysis_task(task, request)
            elif task.task_type == "code_generation":
                return await self._execute_code_generation_task(task, request)
            elif task.task_type == "code_execution":
                return await self._execute_code_execution_task(task, request)
            elif task.task_type == "visualization":
                return await self._execute_visualization_task(task, request)
            elif task.task_type == "insight_generation":
                return await self._execute_insight_generation_task(task, request)
            else:
                return {
                    'status': 'error',
                    'error': f'Unknown task type: {task.task_type}'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _execute_data_analysis_task(self, task: DataAnalysisTask, request: AnalysisRequest) -> Dict[str, Any]:
        """执行数据分析任务"""
        return await self.code_generator.generate_analysis_code(task.description, self.data_context)
    
    async def _execute_code_generation_task(self, task: DataAnalysisTask, request: AnalysisRequest) -> Dict[str, Any]:
        """执行代码生成任务"""
        if "visualization" in task.description.lower():
            # 确定图表类型和列
            chart_type = self._determine_chart_type(request.user_question)
            columns = request.specific_columns or self._suggest_columns_for_visualization()
            return await self.code_generator.generate_visualization_code(
                chart_type, self.data_context, columns, request.user_question
            )
        else:
            return await self.code_generator.generate_analysis_code(task.description, self.data_context)
    
    async def _execute_code_execution_task(self, task: DataAnalysisTask, request: AnalysisRequest) -> Dict[str, Any]:
        """执行代码执行任务"""
        # 查找最近生成的代码
        latest_code_result = self._find_latest_code_result()
        if latest_code_result and 'code' in latest_code_result:
            return await self.code_executor.execute_code(latest_code_result['code'], self.data_context)
        else:
            return {
                'status': 'error',
                'error': 'No code found to execute'
            }
    
    async def _execute_visualization_task(self, task: DataAnalysisTask, request: AnalysisRequest) -> Dict[str, Any]:
        """执行可视化任务"""
        chart_type = self._determine_chart_type(request.user_question)
        columns = request.specific_columns or self._suggest_columns_for_visualization()
        
        # 生成可视化代码
        viz_result = await self.code_generator.generate_visualization_code(
            chart_type, self.data_context, columns, request.user_question
        )
        
        # 执行可视化代码
        if viz_result['status'] == 'success':
            exec_result = await self.code_executor.execute_code(viz_result['code'], self.data_context)
            return {
                'status': 'success',
                'visualization_code': viz_result,
                'execution_result': exec_result
            }
        else:
            return viz_result
    
    async def _execute_insight_generation_task(self, task: DataAnalysisTask, request: AnalysisRequest) -> Dict[str, Any]:
        """执行洞察生成任务"""
        # 收集所有分析结果
        analysis_results = [result for result in self.analysis_history if result.get('status') == 'success']
        return await self.insight_generator.generate_insights(analysis_results, self.data_context, request.user_question)
    
    async def _load_data(self, file_path: str) -> Dict[str, Any]:
        """加载数据"""
        print(f"📂 Loading data from: {file_path}")
        result = await self.data_processor.load_data(file_path)
        
        if result['status'] == 'success':
            print(f"✅ Data loaded successfully: {result['shape']}")
        else:
            print(f"❌ Failed to load data: {result['error']}")
        
        return result
    
    async def _generate_final_insights(self, analysis_results: List[Dict[str, Any]], request: AnalysisRequest) -> Dict[str, Any]:
        """生成最终洞察"""
        print("🔍 Generating final insights...")
        return await self.insight_generator.generate_insights(analysis_results, self.data_context, request.user_question)
    
    def _check_dependencies_completed(self, task: DataAnalysisTask, completed_results: List[Dict[str, Any]]) -> bool:
        """检查任务依赖是否已完成"""
        if not task.dependencies:
            return True
        
        completed_task_ids = {result.get('task_id') for result in completed_results if result.get('status') == 'success'}
        return all(dep in completed_task_ids for dep in task.dependencies)
    
    def _find_latest_code_result(self) -> Optional[Dict[str, Any]]:
        """查找最新的代码生成结果"""
        for result in reversed(self.analysis_history):
            if result.get('status') == 'success' and 'code' in result:
                return result
        return None
    
    def _determine_chart_type(self, user_question: str) -> str:
        """根据用户问题确定图表类型"""
        question_lower = user_question.lower()
        
        if any(word in question_lower for word in ['line', 'trend', 'time', 'over time']):
            return 'line_chart'
        elif any(word in question_lower for word in ['bar', 'column', 'compare', 'comparison']):
            return 'bar_chart'
        elif any(word in question_lower for word in ['scatter', 'relationship', 'correlation']):
            return 'scatter_chart'
        elif any(word in question_lower for word in ['pie', 'proportion', 'percentage', 'share']):
            return 'pie_chart'
        elif any(word in question_lower for word in ['histogram', 'distribution']):
            return 'histogram'
        elif any(word in question_lower for word in ['box', 'boxplot', 'outlier']):
            return 'boxplot'
        elif any(word in question_lower for word in ['heatmap', 'correlation matrix']):
            return 'heatmap'
        else:
            return 'bar_chart'  # 默认
    
    def _suggest_columns_for_visualization(self) -> List[str]:
        """为可视化建议列"""
        if not self.data_context.dataframe is not None:
            return []
        
        df = self.data_context.dataframe
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # 返回前几个数值列和分类列
        suggested = []
        if categorical_columns:
            suggested.append(categorical_columns[0])
        if numeric_columns:
            suggested.extend(numeric_columns[:2])
        
        return suggested[:3]  # 最多返回3列
    
    def _build_context_for_planning(self, request: AnalysisRequest) -> str:
        """为计划生成构建上下文信息"""
        if not self.data_context.dataframe is not None:
            return "No data loaded yet."
        
        df = self.data_context.dataframe
        context = f"""
        Current Dataset:
        - Shape: {df.shape}
        - Columns: {df.columns.tolist()}
        - Data Types: {df.dtypes.to_dict()}
        - Missing Values: {df.isnull().sum().to_dict()}
        """
        
        if self.data_context.summary_stats:
            context += f"\nSummary Statistics Available: Yes"
        
        if self.analysis_history:
            context += f"\nPrevious Analyses: {len(self.analysis_history)} completed"
        
        return context
    
    def _serialize_data_context(self) -> Dict[str, Any]:
        """序列化数据上下文"""
        if not self.data_context.dataframe is not None:
            return {'status': 'no_data'}
        
        df = self.data_context.dataframe
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'file_path': self.data_context.file_path,
            'file_type': self.data_context.file_type
        }
    
    def _save_to_history(self, request: AnalysisRequest, analysis_results: List[Dict[str, Any]], insights: Dict[str, Any]):
        """保存到分析历史"""
        history_entry = {
            'timestamp': asyncio.get_event_loop().time(),
            'request': asdict(request),
            'analysis_results': analysis_results,
            'insights': insights,
            'status': 'completed'
        }
        
        self.analysis_history.append(history_entry)
        
        # 保持历史记录在合理范围内
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """获取分析历史"""
        return self.analysis_history.copy()
    
    def get_current_data_info(self) -> Dict[str, Any]:
        """获取当前数据信息"""
        return self._serialize_data_context()