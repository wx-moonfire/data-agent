import pandas as pd
import numpy as np
import json
import io
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams


@dataclass
class DataAnalysisTask:
    """数据分析任务定义"""
    task_id: str
    description: str
    task_type: str  # 'data_processing', 'analysis', 'visualization', 'insight'
    priority: int
    dependencies: List[str]
    status: str = 'pending'  # 'pending', 'in_progress', 'completed', 'failed'
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class DataContext:
    """数据上下文信息"""
    dataframe: Optional[pd.DataFrame] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    columns_info: Optional[Dict[str, str]] = None
    summary_stats: Optional[Dict] = None
    data_quality_report: Optional[Dict] = None


class DataProcessorAgent:
    """数据处理智能体 - 负责数据上传、清洗和预处理"""
    
    def __init__(self, name: str = "data_processor"):
        self.name = name
        self.agent = None
        self.llm = None
        self.data_context = DataContext()
    
    async def initialize(self):
        """初始化智能体"""
        self.agent = Agent(
            name=self.name,
            instruction="""You are a data processing expert. Your responsibilities include:
            1. Loading and parsing CSV/Excel files
            2. Analyzing data structure and types
            3. Detecting and handling missing values
            4. Data quality assessment
            5. Basic data cleaning and preprocessing
            
            Always provide detailed analysis of the data structure, quality issues, and recommendations.""",
            server_names=["filesystem", "python"]
        )
        await self.agent.initialize()
        self.llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
    
    async def load_data(self, file_path: str) -> Dict[str, Any]:
        """加载数据文件"""
        try:
            # 检测文件类型
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                file_type = 'csv'
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
                file_type = 'excel'
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # 更新数据上下文
            self.data_context.dataframe = df
            self.data_context.file_path = file_path
            self.data_context.file_type = file_type
            
            # 分析数据结构
            analysis_result = await self._analyze_data_structure(df)
            
            return {
                'status': 'success',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'analysis': analysis_result
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据结构"""
        # 基本统计信息
        summary_stats = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # 列类型分析
        columns_info = {}
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }
            
            # 数值型列的统计信息
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            
            # 文本型列的信息
            elif df[col].dtype == 'object':
                col_info.update({
                    'most_common': df[col].value_counts().head(3).to_dict() if not df[col].empty else {},
                    'avg_length': df[col].astype(str).str.len().mean() if not df[col].empty else 0
                })
            
            columns_info[col] = col_info
        
        self.data_context.columns_info = columns_info
        self.data_context.summary_stats = summary_stats
        
        # 使用LLM生成数据质量报告
        quality_prompt = f"""
        Based on the following data analysis, provide a comprehensive data quality assessment:
        
        Dataset Shape: {summary_stats['shape']}
        Missing Values: {summary_stats['missing_values']}
        Duplicate Rows: {summary_stats['duplicate_rows']}
        
        Column Information:
        {json.dumps(columns_info, indent=2, default=str)}
        
        Please provide:
        1. Overall data quality score (1-10)
        2. Key data quality issues identified
        3. Recommendations for data cleaning
        4. Suggested data types for each column
        5. Potential analysis opportunities
        """
        
        quality_report = await self.llm.generate_str(message=quality_prompt)
        
        return {
            'summary_stats': summary_stats,
            'columns_info': columns_info,
            'quality_report': quality_report
        }


class CodeGeneratorAgent:
    """代码生成智能体 - 负责生成数据分析和可视化代码"""
    
    def __init__(self, name: str = "code_generator"):
        self.name = name
        self.agent = None
        self.llm = None
    
    async def initialize(self):
        """初始化智能体"""
        self.agent = Agent(
            name=self.name,
            instruction="""You are an expert Python data analyst and code generator. Your responsibilities include:
            1. Generating executable Python code for data analysis
            2. Creating visualization code using matplotlib, seaborn, and plotly
            3. Writing statistical analysis and machine learning code
            4. Ensuring code is safe, efficient, and well-documented
            5. Providing code explanations and comments
            
            Always generate complete, executable code with proper imports and error handling.
            Focus on pandas, numpy, matplotlib, seaborn, plotly, and scikit-learn libraries.""",
            server_names=["python", "filesystem"]
        )
        await self.agent.initialize()
        self.llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
    
    async def generate_analysis_code(self, task_description: str, data_context: DataContext) -> Dict[str, Any]:
        """生成数据分析代码"""
        try:
            # 构建上下文信息
            context_info = self._build_context_info(data_context)
            
            code_prompt = f"""
            Generate Python code for the following data analysis task:
            
            Task: {task_description}
            
            Data Context:
            {context_info}
            
            Requirements:
            1. Use pandas DataFrame named 'df' (assume it's already loaded)
            2. Include proper imports
            3. Add comments explaining each step
            4. Handle potential errors
            5. Return results in a structured format
            6. Use appropriate visualization libraries if needed
            
            Generate complete, executable Python code:
            """
            
            generated_code = await self.llm.generate_str(message=code_prompt)
            
            # 提取代码块
            code = self._extract_code_block(generated_code)
            
            return {
                'status': 'success',
                'code': code,
                'explanation': generated_code,
                'task': task_description
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def generate_visualization_code(self, chart_type: str, data_context: DataContext, 
                                        columns: List[str], title: str = "") -> Dict[str, Any]:
        """生成可视化代码"""
        try:
            context_info = self._build_context_info(data_context)
            
            viz_prompt = f"""
            Generate Python visualization code for:
            
            Chart Type: {chart_type}
            Columns to use: {columns}
            Title: {title}
            
            Data Context:
            {context_info}
            
            Requirements:
            1. Use pandas DataFrame named 'df'
            2. Create professional-looking charts
            3. Include proper labels, titles, and legends
            4. Use appropriate color schemes
            5. Handle missing values appropriately
            6. Save the chart as both PNG and HTML (for interactive charts)
            
            Generate complete Python code using matplotlib, seaborn, or plotly:
            """
            
            generated_code = await self.llm.generate_str(message=viz_prompt)
            code = self._extract_code_block(generated_code)
            
            return {
                'status': 'success',
                'code': code,
                'explanation': generated_code,
                'chart_type': chart_type,
                'columns': columns
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _build_context_info(self, data_context: DataContext) -> str:
        """构建数据上下文信息字符串"""
        if not data_context.dataframe is not None:
            return "No data context available"
        
        df = data_context.dataframe
        info = f"""
        Dataset Shape: {df.shape}
        Columns: {df.columns.tolist()}
        Data Types: {df.dtypes.to_dict()}
        """
        
        if data_context.columns_info:
            info += f"\nColumn Details: {json.dumps(data_context.columns_info, indent=2, default=str)}"
        
        return info
    
    def _extract_code_block(self, text: str) -> str:
        """从生成的文本中提取代码块"""
        # 查找代码块标记
        import re
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # 如果没有找到代码块标记，尝试提取整个文本
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else text


class CodeExecutorAgent:
    """代码执行智能体 - 负责安全执行生成的代码"""
    
    def __init__(self, name: str = "code_executor"):
        self.name = name
        self.agent = None
        self.llm = None
        self.execution_context = {}
    
    async def initialize(self):
        """初始化智能体"""
        self.agent = Agent(
            name=self.name,
            instruction="""You are a code execution specialist. Your responsibilities include:
            1. Safely executing Python code in a controlled environment
            2. Capturing and handling execution results and errors
            3. Managing execution context and variables
            4. Providing detailed execution reports
            5. Debugging and error analysis
            
            Always prioritize safety and provide comprehensive execution feedback.""",
            server_names=["python", "filesystem"]
        )
        await self.agent.initialize()
        self.llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
    
    async def execute_code(self, code: str, data_context: DataContext) -> Dict[str, Any]:
        """执行代码"""
        try:
            # 准备执行环境
            execution_globals = self._prepare_execution_environment(data_context)
            execution_locals = {}
            
            # 捕获输出
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                
                # 执行代码
                exec(code, execution_globals, execution_locals)
                
                # 获取输出
                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()
                
                # 提取结果变量
                results = self._extract_results(execution_locals)
                
                return {
                    'status': 'success',
                    'stdout': stdout_output,
                    'stderr': stderr_output,
                    'results': results,
                    'execution_locals': {k: str(v) for k, v in execution_locals.items() if not k.startswith('_')}
                }
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _prepare_execution_environment(self, data_context: DataContext) -> Dict[str, Any]:
        """准备代码执行环境"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        from scipy import stats
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        execution_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'stats': stats,
            'train_test_split': train_test_split,
            'StandardScaler': StandardScaler,
            'LabelEncoder': LabelEncoder,
            'LinearRegression': LinearRegression,
            'LogisticRegression': LogisticRegression,
            'RandomForestRegressor': RandomForestRegressor,
            'RandomForestClassifier': RandomForestClassifier,
            'accuracy_score': accuracy_score,
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score
        }
        
        # 添加数据框
        if data_context.dataframe is not None:
            execution_globals['df'] = data_context.dataframe.copy()
        
        return execution_globals
    
    def _extract_results(self, execution_locals: Dict[str, Any]) -> Dict[str, Any]:
        """提取执行结果"""
        results = {}
        
        for key, value in execution_locals.items():
            if not key.startswith('_'):
                try:
                    # 尝试序列化结果
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        results[key] = value
                    elif isinstance(value, pd.DataFrame):
                        results[key] = {
                            'type': 'DataFrame',
                            'shape': value.shape,
                            'columns': value.columns.tolist(),
                            'head': value.head().to_dict('records')
                        }
                    elif isinstance(value, pd.Series):
                        results[key] = {
                            'type': 'Series',
                            'length': len(value),
                            'head': value.head().tolist()
                        }
                    else:
                        results[key] = str(value)
                except Exception:
                    results[key] = f"<{type(value).__name__} object>"
        
        return results


class InsightGeneratorAgent:
    """洞察生成智能体 - 负责生成业务洞察和建议"""
    
    def __init__(self, name: str = "insight_generator"):
        self.name = name
        self.agent = None
        self.llm = None
    
    async def initialize(self):
        """初始化智能体"""
        self.agent = Agent(
            name=self.name,
            instruction="""You are a business intelligence and data insights expert. Your responsibilities include:
            1. Analyzing data analysis results to extract meaningful insights
            2. Identifying trends, patterns, and anomalies
            3. Generating actionable business recommendations
            4. Providing clear, non-technical explanations of findings
            5. Suggesting next steps and follow-up analyses
            
            Always focus on practical, actionable insights that can drive business decisions.""",
            server_names=["filesystem"]
        )
        await self.agent.initialize()
        self.llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
    
    async def generate_insights(self, analysis_results: List[Dict[str, Any]], 
                              data_context: DataContext, user_question: str) -> Dict[str, Any]:
        """生成数据洞察"""
        try:
            # 构建分析结果摘要
            results_summary = self._summarize_analysis_results(analysis_results)
            
            # 构建数据上下文摘要
            context_summary = self._summarize_data_context(data_context)
            
            insight_prompt = f"""
            Based on the following data analysis results, generate comprehensive business insights:
            
            User Question: {user_question}
            
            Data Context:
            {context_summary}
            
            Analysis Results:
            {results_summary}
            
            Please provide:
            1. Key Findings: 3-5 most important discoveries from the analysis
            2. Trends and Patterns: Notable trends, correlations, or patterns identified
            3. Anomalies and Outliers: Any unusual data points or unexpected findings
            4. Business Implications: What these findings mean for the business
            5. Actionable Recommendations: Specific, practical next steps
            6. Risk Assessment: Potential risks or concerns identified
            7. Opportunities: New opportunities or areas for improvement
            8. Follow-up Questions: Suggested additional analyses or investigations
            
            Format your response in clear sections with bullet points for easy reading.
            """
            
            insights = await self.llm.generate_str(message=insight_prompt)
            
            return {
                'status': 'success',
                'insights': insights,
                'user_question': user_question,
                'analysis_count': len(analysis_results)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _summarize_analysis_results(self, results: List[Dict[str, Any]]) -> str:
        """总结分析结果"""
        summary = []
        
        for i, result in enumerate(results, 1):
            if result.get('status') == 'success':
                summary.append(f"Analysis {i}:")
                if 'task' in result:
                    summary.append(f"  Task: {result['task']}")
                if 'results' in result:
                    summary.append(f"  Results: {json.dumps(result['results'], indent=2, default=str)[:500]}...")
                if 'stdout' in result and result['stdout']:
                    summary.append(f"  Output: {result['stdout'][:300]}...")
            else:
                summary.append(f"Analysis {i}: Failed - {result.get('error', 'Unknown error')}")
            summary.append("")
        
        return "\n".join(summary)
    
    def _summarize_data_context(self, data_context: DataContext) -> str:
        """总结数据上下文"""
        if not data_context.dataframe is not None:
            return "No data context available"
        
        df = data_context.dataframe
        summary = f"""
        Dataset Overview:
        - Shape: {df.shape}
        - Columns: {df.columns.tolist()}
        - Data Types: {df.dtypes.value_counts().to_dict()}
        - Missing Values: {df.isnull().sum().sum()} total
        """
        
        if data_context.summary_stats:
            summary += f"\nSummary Statistics: {json.dumps(data_context.summary_stats, indent=2, default=str)[:500]}..."
        
        return summary