# Universal MCP Orchestrator

## 🚀 Enhanced Features

Your Universal MCP Server has been transformed into a powerful **MCP Orchestrator** that allows you to:

- **Select AI Models**: Choose from OpenAI, Google Gemini, or Anthropic Claude models
- **Connect MCP Servers**: Integrate with various MCP servers for enhanced capabilities
- **Interactive Chat**: Real-time chat interface with tool integration visualization
- **Web Search**: Search the web without API keys using the integrated web search MCP server
- **File Operations**: Secure file system operations via MCP
- **Git Integration**: Repository management and code analysis tools

## 🎯 What's New

### Model Selection Dropdown
- Choose from all available models across providers
- View model specifications (context length, cost, capabilities)
- Real-time provider status monitoring

### MCP Server Integration
- **Web Search**: Free Google search without API keys required
- **Filesystem**: Secure file operations with configurable access
- **Git**: Repository operations and code analysis
- **Extensible**: Easy to add more MCP servers

### Enhanced Chat Interface
- Interactive chat with AI models
- Real-time tool usage visualization
- System prompt configuration
- Chat history management

## 🛠️ Quick Setup

### Prerequisites
- Python 3.8+
- Node.js and npm (for MCP servers)
- At least one AI provider API key

### Installation

1. **Run the setup script**:
   ```bash
   python setup_mcp_orchestrator.py
   ```

2. **Install enhanced dependencies**:
   ```bash
   pip install -r requirements_enhanced.txt
   ```

3. **Configure your API keys** in `.env` file:
   ```env
   # At least one of these is required
   OPENAI_API_KEY=your-openai-key-here
   GEMINI_API_KEY=your-gemini-key-here
   ANTHROPIC_API_KEY=your-anthropic-key-here
   ```

4. **Start the orchestrator**:
   ```bash
   python run_orchestrator.py
   ```

5. **Open your browser** to `http://localhost:8080`

## 🎮 How to Use

### 1. Select Your AI Model
- Choose a provider (OpenAI, Gemini, or Anthropic)
- Select a specific model from the dropdown
- View model information (context length, cost)

### 2. Choose MCP Server (Optional)
- Select "Web Search" for internet search capabilities
- Choose "Filesystem" for file operations
- Pick "Git" for repository management
- Or select "No MCP Server" for direct chat

### 3. Configure System Prompt (Optional)
- Add custom instructions for the AI
- Predefined templates available

### 4. Start Chatting
- Type your message and press Enter
- Watch as the AI uses tools when needed
- View tool usage in real-time

## 💡 Example Use Cases

### Web Search Integration
**User**: "What are the latest developments in AI in 2025?"
**AI**: *Uses web search tool to find current information*
**Result**: Comprehensive, up-to-date response with real search results

### File Operations
**User**: "Create a Python script that analyzes log files"
**AI**: *Uses filesystem tools to create and manage files*
**Result**: Script created and saved to your system

### Git Repository Analysis
**User**: "Explain the recent changes in this repository"
**AI**: *Uses git tools to analyze commits and changes*
**Result**: Detailed analysis of repository history and changes

## 🔧 Advanced Configuration

### Adding New MCP Servers

Edit `orchestrator_config.json` to add new MCP servers:

```json
{
  "mcp_servers": {
    "your-server": {
      "name": "Your Custom Server",
      "description": "Description of your server",
      "command": ["command", "to", "run", "server"],
      "tools": ["tool1", "tool2"],
      "auto_install": false
    }
  }
}
```

### Custom System Prompts

Create role-specific prompts:

```
Research Assistant: "You are a research assistant that searches for and synthesizes information from multiple sources."

Code Reviewer: "You are a senior developer who reviews code for best practices, security issues, and optimization opportunities."

Data Analyst: "You analyze data and provide insights, using tools to access files and generate reports."
```

## 🌟 Available MCP Servers

### Web Search Server
- **No API Key Required**: Uses free Google search
- **Tools**: `search`
- **Use Cases**: Current events, research, fact-checking

### Filesystem Server
- **Secure Access**: Configurable permissions
- **Tools**: `read_file`, `write_file`, `create_directory`, `list_directory`
- **Use Cases**: File management, script generation, data processing

### Git Server
- **Repository Operations**: Local git repositories
- **Tools**: `git_log`, `git_diff`, `git_show`, `search_files`
- **Use Cases**: Code analysis, version control, repository management

## 🚀 Running the Orchestrator

### Development Mode
```bash
python run_orchestrator.py --host localhost --port 8080
```

### Production Mode
```bash
python run_orchestrator.py --host 0.0.0.0 --port 80
```

### With Custom Configuration
```bash
python run_orchestrator.py --config custom_config.json
```

## 🔍 Troubleshooting

### Common Issues

**"No providers available"**
- Check your API keys in `.env`
- Verify internet connectivity
- Ensure API keys are valid

**"MCP server failed to start"**
- Check Node.js and npm installation
- Verify MCP server dependencies
- Check server logs for errors

**"Web search not working"**
- Ensure Node.js is installed
- Check if the web search server compiled successfully
- Verify internet connectivity

### Debug Mode

Enable debug logging:
```bash
python run_orchestrator.py --log-level DEBUG
```

## 📊 Monitoring

The web interface provides real-time monitoring:
- Connected providers count
- Available tools count
- Active MCP server connections
- Chat message history
- Tool usage statistics

## 🔐 Security

- API keys stored in environment variables
- MCP servers run in isolated processes
- Configurable file system permissions
- Audit logging for all operations
- Rate limiting protection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your MCP server integration
4. Test with the orchestrator
5. Submit a pull request

### Adding New MCP Servers

1. Add server configuration to `mcp_orchestrator.py`
2. Update the UI dropdown options
3. Test the integration
4. Document the new functionality

## 📝 License

MIT License - see LICENSE file for details

## 🆘 Support

- Check the troubleshooting section above
- Review server logs in `logs/` directory
- Open an issue on GitHub
- Join our community discussions

---

**Happy orchestrating!** 🎵🤖

Enjoy the power of connecting AI models with real-world tools through the standardized MCP protocol.