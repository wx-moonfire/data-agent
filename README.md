# Multi-Agent Data Analysis Assistant

🤖 **A sophisticated AI-powered data analysis system featuring multiple specialized agents working together to provide intelligent data processing, analysis, visualization, and business insights.**

## 🌟 Features

### 🔹 **Intelligent Data Processing**
- **Multi-format Support**: CSV, Excel (.xlsx, .xls) file uploads
- **Automatic Field Recognition**: Smart detection of data types (text, numeric, datetime, categorical)
- **Data Quality Assessment**: Automatic identification of missing values, outliers, and data quality issues
- **Unified Data Structure**: Standardized data representation for seamless agent collaboration

### 🔹 **AI-Powered Code Generation**
- **Natural Language to Code**: Convert user questions into executable Python analysis code
- **Multi-library Support**: Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn integration
- **Context-Aware Generation**: Code generation based on actual data structure and content
- **Self-Documenting Code**: Generated code includes explanatory comments

### 🔹 **Safe Code Execution**
- **Sandboxed Environment**: Secure execution of generated analysis code
- **Error Handling**: Robust error detection and recovery mechanisms
- **Resource Management**: Memory and execution time limits for safety
- **Result Validation**: Automatic validation of execution results

### 🔹 **Advanced Visualizations**
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Chart Type Intelligence**: Automatic selection of appropriate chart types
- **Customizable Styling**: Professional-grade chart styling and formatting
- **Export Capabilities**: Save charts in multiple formats

### 🔹 **Business Intelligence**
- **Automated Insights**: AI-generated business insights and recommendations
- **Trend Analysis**: Identification of patterns, trends, and anomalies
- **Correlation Discovery**: Automatic detection of relationships between variables
- **Actionable Recommendations**: Clear, executable business suggestions

### 🔹 **Multi-Agent Coordination**
- **Task Decomposition**: Intelligent breaking down of complex analysis requests
- **Agent Specialization**: Each agent focuses on specific analysis aspects
- **Dynamic Workflow**: Adaptive execution plans based on data and requirements
- **Error Recovery**: Self-healing capabilities when individual tasks fail

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ Chat        │ │ Data        │ │ Visualiza-  │ │ History  │ │
│  │ Analysis    │ │ Explorer    │ │ tions       │ │          │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Data Analysis Orchestrator                     │
│                    (Task Coordination)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Specialized Agents                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ Data        │ │ Code        │ │ Code        │ │ Insight  │ │
│  │ Processor   │ │ Generator   │ │ Executor    │ │ Generator│ │
│  │ Agent       │ │ Agent       │ │ Agent       │ │ Agent    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Framework                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ Filesystem  │ │ Chart       │ │ Python      │ │ Fetch    │ │
│  │ Server      │ │ Server      │ │ Server      │ │ Server   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 16+** (for MCP servers)
- **OpenAI API Key**

### 1. Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/lastmile-ai/mcp-agent.git
cd mcp-agent/examples/data-agent

# Install Python dependencies
pip install -r requirements.txt

# Install MCP servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @antv/mcp-server-chart
pip install mcp-server-fetch
pip install mcp-server-python
```

### 2. Configuration

```bash
# Copy secrets template
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml

# Edit secrets file and add your OpenAI API key
# mcp_agent.secrets.yaml:
openai:
  api_key: "your-openai-api-key-here"
```

### 3. System Test

```bash
# Test system components
python main.py test
```

### 4. Launch Application

```bash
# Start the Streamlit web interface
streamlit run app.py
```

### 5. Demo Analysis

```bash
# Run demo with sample data
python main.py demo
```

## 📖 Usage Guide

### Basic Workflow

1. **Upload Data**: Use the sidebar to upload CSV or Excel files
2. **Ask Questions**: Type natural language questions about your data
3. **Review Results**: Examine generated analysis, code, and visualizations
4. **Explore Further**: Use quick analysis buttons or ask follow-up questions

### Example Questions

```
💬 "Show me a comprehensive overview of this dataset"
💬 "What are the correlations between sales and marketing spend?"
💬 "Create a time series visualization of revenue trends"
💬 "Identify any outliers or anomalies in the customer data"
💬 "Analyze seasonal patterns in the sales data"
💬 "Generate insights about customer behavior segments"
💬 "Compare performance across different product categories"
💬 "What factors most influence customer satisfaction?"
```

### Interface Tabs

#### 🗨️ **Chat Analysis**
- Interactive chat interface for natural language queries
- Real-time analysis results and visualizations
- Quick analysis buttons for common tasks
- Message history and context preservation

#### 📊 **Data Explorer**
- Dataset overview and statistics
- Column information and data types
- Missing value analysis
- Data preview with pagination

#### 📈 **Visualizations**
- Interactive chart gallery
- Chart type selection and customization
- Export capabilities
- Responsive design for different screen sizes

#### 📚 **History**
- Analysis session history
- Previous questions and results
- Rerun capability for past analyses
- Export analysis reports

## 🔧 Advanced Configuration

### MCP Server Configuration

The system uses multiple MCP servers for different capabilities:

```yaml
# mcp_agent.config.yaml
servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/data"]
    
  chart:
    command: npx
    args: ["-y", "@antv/mcp-server-chart"]
    
  python:
    command: uvx
    args: ["mcp-server-python"]
    
  fetch:
    command: uvx
    args: ["mcp-server-fetch"]
```

### Agent Customization

Each agent can be customized for specific use cases:

```python
# Custom agent configuration
data_processor = DataProcessorAgent(
    max_file_size_mb=100,
    supported_formats=['.csv', '.xlsx', '.xls'],
    auto_detect_encoding=True
)

code_generator = CodeGeneratorAgent(
    max_code_length=5000,
    allowed_libraries=['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly'],
    include_comments=True
)
```

## 🛠️ Troubleshooting

### Common Issues

#### ❌ **"MCP server not found"**
```bash
# Reinstall MCP servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @antv/mcp-server-chart
```

#### ❌ **"OpenAI API key invalid"**
- Check your API key in `mcp_agent.secrets.yaml`
- Ensure the key has sufficient credits
- Verify the key format (starts with 'sk-')

#### ❌ **"File upload failed"**
- Check file size (max 200MB)
- Verify file format (CSV, XLSX, XLS)
- Ensure file is not corrupted

#### ❌ **"Code execution timeout"**
- Reduce dataset size for complex operations
- Simplify analysis requests
- Check system resources

### Debug Mode

```bash
# Enable debug logging
export MCP_AGENT_DEBUG=1
streamlit run app.py
```

### Performance Optimization

```python
# Optimize for large datasets
config = {
    'chunk_size': 10000,
    'max_memory_usage': '2GB',
    'parallel_processing': True,
    'cache_results': True
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone for development
git clone https://github.com/lastmile-ai/mcp-agent.git
cd mcp-agent/examples/data-agent

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MCP Framework**: [Model Context Protocol](https://github.com/modelcontextprotocol)
- **Chart Server**: [@antv/mcp-server-chart](https://www.modelscope.cn/mcp/servers/@antvis/mcp-server-chart)
- **OpenAI**: GPT-4 language model
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library

## 📞 Support

For questions, issues, or feature requests:

- 📧 **Email**: support@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/lastmile-ai/mcp-agent/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/lastmile-ai/mcp-agent/discussions)
- 📖 **Documentation**: [Full Documentation](https://docs.example.com)

---

**Built with ❤️ using MCP (Model Context Protocol) and AI**